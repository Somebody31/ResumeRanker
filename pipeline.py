"""
pipeline.py — High-quality resume ranking dataset generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single provider: Cerebras Qwen 3 235B for everything.

Key improvements over v1:
  - Pre-computed skill audit injected into every candidate block
    (eliminates hallucinated missing/present skills entirely)
  - Structured JD metadata injected above prose
  - Rubric gap contradiction resolved: 10pt minimum everywhere
  - Math + cap consistency validation added
  - MIN_BEST_OVERLAP_RATIO raised to 0.60
  - MAX_RESUME_REUSE lowered to 2
  - Stage 5 balances on score spread, not best-candidate bucket
  - Hard negatives filtered on domain-specific skills only
  - Robust JSON repair in extract_object
  - More writing styles, sampled with replacement

Validation philosophy:
  We check the RANKING OUTPUT only — not absolute score values.
  What matters for training:
    1. Ranking order is correct (best candidate first)
    2. Score gaps exist between ranks (model learns to differentiate)
    3. Hard negative is last (model learns domain mismatch)
    4. Reasons reference actual skills (model learns to explain)
    5. Math in reason strings is internally consistent
    6. Cap values match assigned scores

Usage:
    python pipeline.py                  # default samples
    python pipeline.py --samples 500
    python pipeline.py --dry-run        # 20 samples
    python pipeline.py --skip-gen       # re-pair + re-rank only
"""

import os
import re
import json
import hashlib
import time
import urllib.request
import random
import argparse
import sys
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Terminal Aesthetics
# ---------------------------------------------------------------------------

class Color:
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    DIM     = "\033[2m"

def status_msg(msg: str, color: str = Color.CYAN, bold: bool = True):
    prefix = f"{Color.BOLD}{color}●{Color.RESET} " if bold else ""
    print(f"{prefix}{color}{msg}{Color.RESET}")

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.stages = {}
        self.current_stage = None
        self._stage_start = None

    def start_stage(self, name: str):
        self.current_stage = name
        self._stage_start = time.time()

    def end_stage(self):
        if self.current_stage:
            self.stages[self.current_stage] = time.time() - self._stage_start
            self.current_stage = None

    def total(self) -> float:
        return time.time() - self.start_time

    def format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL            = "qwen-3-235b-a22b-instruct-2507"
BASE_URL         = "https://api.cerebras.ai/v1"
MAX_RETRIES      = 4
RATE_WAIT        = 10
RESUME_MIN_WORDS       = 80
JD_MIN_WORDS           = 100

# Raised from 0.40 → 0.60: ensures the best candidate in each pair has
# genuine overlap with the JD, producing cleaner training signal.
MIN_BEST_OVERLAP_RATIO = 0.60

# Lowered from 4 → 2: prevents resume memorisation across pairs.
MAX_RESUME_REUSE       = 2

DATA_DIR     = Path("data")
JDS_FILE     = DATA_DIR / "jds_raw.jsonl"
RESUMES_FILE = DATA_DIR / "resumes_raw.jsonl"
PAIRS_FILE   = DATA_DIR / "pairs.jsonl"

DOMAIN_SKILLS = {
    "backend_engineering": [
        "Python", "Go", "Java", "Node.js", "FastAPI", "Django", "Spring Boot",
        "PostgreSQL", "MySQL", "Redis", "REST APIs", "GraphQL", "gRPC",
        "Microservices", "Kafka", "RabbitMQ", "Docker", "Linux",
        "Distributed Systems", "WebSockets", "Celery", "SQLAlchemy"
    ],
    "frontend_engineering": [
        "React", "Next.js", "TypeScript", "JavaScript", "Vue.js", "Angular",
        "CSS", "Tailwind", "Webpack", "Vite", "Web Performance", "Accessibility",
        "Jest", "Cypress", "Playwright", "GraphQL", "REST APIs", "Figma",
        "Storybook", "Zustand", "Redux", "Web Components", "WCAG"
    ],
    "devops_platform": [
        "Kubernetes", "Docker", "Terraform", "Ansible", "AWS", "GCP", "Azure",
        "CI/CD", "Jenkins", "GitHub Actions", "ArgoCD", "Prometheus", "Grafana",
        "Helm", "Bash", "Linux", "Istio", "Vault", "CloudFormation", "Pulumi",
        "EKS", "GKE"
    ],
    "data_engineering": [
        "Apache Spark", "Airflow", "dbt", "SQL", "Python", "Kafka", "Flink",
        "Snowflake", "BigQuery", "Redshift", "Delta Lake", "Iceberg", "ETL/ELT",
        "Data Modeling", "Databricks", "Pandas", "Great Expectations",
        "DuckDB", "Trino", "Hive", "Fivetran"
    ],
    "ml_engineering": [
        "PyTorch", "TensorFlow", "CUDA", "Triton", "MLflow", "Kubeflow",
        "Model Serving", "TorchServe", "ONNX", "Quantization", "Fine-tuning",
        "LLMs", "RAG", "Vector DBs", "Transformers", "Feature Stores",
        "vLLM", "LangChain", "Pinecone", "Weaviate", "BentoML"
    ],
    "full_stack": [
        "React", "Node.js", "Python", "PostgreSQL", "REST APIs", "TypeScript",
        "Docker", "AWS", "Redis", "CI/CD", "GraphQL", "MongoDB",
        "Next.js", "FastAPI", "Prisma", "Nginx", "Auth/OAuth"
    ],
}

SENIORITY_CONFIG = {
    "Junior":    {"min_years": 1,  "max_years": 3,  "weight": 0.20},
    "Mid-level": {"min_years": 3,  "max_years": 6,  "weight": 0.35},
    "Senior":    {"min_years": 5,  "max_years": 9,  "weight": 0.30},
    "Staff":     {"min_years": 8,  "max_years": 15, "weight": 0.15},
}

DOMAIN_WEIGHTS = {
    "backend_engineering":  0.22,
    "frontend_engineering": 0.22,
    "data_engineering":     0.15,
    "ml_engineering":       0.15,
    "devops_platform":      0.13,
    "full_stack":           0.13,
}

SAMPLE_COMPOSITIONS = {
    "three_same_domain": {
        "weight":      0.35,
        "n_domain":    3,
        "n_hardneg":   0,
        "description": "3 same-domain resumes — nuanced discrimination",
    },
    "two_plus_hardneg": {
        "weight":      0.30,
        "n_domain":    2,
        "n_hardneg":   1,
        "description": "2 same-domain + 1 hard negative",
    },
    "four_same_domain": {
        "weight":      0.20,
        "n_domain":    4,
        "n_hardneg":   0,
        "description": "4 same-domain — harder multi-way ranking",
    },
    "three_plus_hardneg": {
        "weight":      0.15,
        "n_domain":    3,
        "n_hardneg":   1,
        "description": "3 same-domain + 1 hard negative",
    },
}

HARD_NEGATIVE_PROFILES = [
    {"type": "Mobile developer",    "skills": ["React Native", "Swift", "Kotlin", "iOS", "Xcode"]},
    {"type": "Marketing engineer",  "skills": ["HubSpot", "Salesforce", "SEO", "Google Analytics"]},
    {"type": "Academic researcher", "skills": ["LaTeX", "R", "MATLAB", "Jupyter", "grant writing"]},
    {"type": "Support engineer",    "skills": ["Zendesk", "ticket triage", "SLA management", "JIRA"]},
    {"type": "QA engineer",         "skills": ["Selenium", "manual testing", "test cases", "bug reports"]},
    {"type": "Data analyst",        "skills": ["Tableau", "Power BI", "Excel", "SQL reporting"]},
]

PREFIXES = [
    "Nova", "Apex", "Lumina", "Stellar", "Forge", "Arc", "Nexus", "Vanta",
    "Prism", "Zenith", "Aether", "Byte", "Cloud", "Data", "Edge", "Flux",
    "Giga", "Helix", "Infra", "Jet", "Kinet", "Logic", "Meta", "Nabla",
    "Orbit", "Pulse", "Quantum", "Ray", "Scale", "Terra", "Unit", "Vector",
    "Wave", "Xenon", "Yield", "Zetta", "Alpha", "Beta", "Gamma", "Delta",
    "Echo", "Falcon", "Grid", "Halo", "Ion", "Jump", "Kite", "Link", "Mode", "Node",
    "Omni", "Peak", "Quest", "Root", "Shift", "Tier", "Ultra", "Vise", "Warp"
]

SUFFIXES = [
    "Systems", "Labs", "Technologies", "Solutions", "Data", "Cloud", "AI", "Works",
    "Dynamics", "Flow", "Grid", "Hub", "Index", "Joint", "Key", "Layer",
    "Matrix", "Networks", "Ops", "Platform", "Queue", "Route", "Stack", "Tech",
    "Utility", "Vault", "Wire", "X", "Yard", "Zone", "Base", "Core", "Engine", "Front",
    "Gear", "Hard", "Internal", "Logic", "Main", "Open", "Path", "Rapid", "Soft", "Trust",
    "Ultra", "View", "Web", "Sync", "Link", "Swift", "Fast", "Smart", "Secure", "Direct"
]

# Extended writing styles — sampled with replacement so batches are genuinely varied
WRITING_STYLES = [
    "bullet points under each role with quantified outcomes",
    "paragraph prose narrative describing career progression",
    "metric-driven bullet points emphasizing numbers and impact",
    "concise paragraphs with a separate skills section at the end",
    "mixed format: short intro paragraph then bullet points per role",
    "reverse-chronological with a two-line summary at the top",
    "achievement-first format: lead each role with the biggest win",
    "technical deep-dive style: emphasize stack decisions and trade-offs",
    "storytelling format: each role described as a problem-solution arc",
    "minimalist format: role, company, one-line impact per bullet",
]

def get_random_companies(count: int = 10) -> list[str]:
    companies = set()
    while len(companies) < count:
        companies.add(f"{random.choice(PREFIXES)}{random.choice(SUFFIXES)}")
    return list(companies)

INFERENCE_SYSTEM_PROMPT = (
    "You are a meticulous technical recruiter specializing in elite engineering talent. "
    "Your task is to rank candidates with absolute precision based on a strict rubric. "
    "You must apply mathematical deductions without bias. "
    "The skill audit for each candidate is PRE-COMPUTED and provided in the candidate block — "
    "do NOT re-derive skills from resume text. Trust the metadata. "
    "Return ONLY a valid JSON object."
)

# ---------------------------------------------------------------------------
# Skill aliases for fuzzy matching (pairing/overlap only — not validation)
# ---------------------------------------------------------------------------

SKILL_ALIASES: dict[str, list[str]] = {
    "LLMs":            ["llm", "large language model", "gpt", "claude", "llama"],
    "RAG":             ["retrieval augmented", "retrieval-augmented", "vector search"],
    "TypeScript":      ["typescript", " ts ", "tsx"],
    "REST APIs":       ["rest api", "restful", "http api"],
    "ETL/ELT":         ["etl", "elt", "data pipeline"],
    "CI/CD":           ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "Auth/OAuth":      ["oauth", "authentication", "jwt", "authorization", "saml"],
    "Vector DBs":      ["vector database", "vector store", "pinecone", "milvus", "weaviate", "chroma"],
    "Apache Spark":    ["pyspark", "spark sql", "apache spark"],
    "Fine-tuning":     ["fine-tuning", "finetuning", "lora", "qlora", "adapter"],
    "Transformers":    ["hugging face", "huggingface", "attention mechanism", "bert", "gpt"],
    "Next.js":         ["nextjs", "next js"],
    "Node.js":         ["nodejs", "node js"],
    "Vue.js":          ["vuejs", "vue js"],
    "Web Components":  ["web component", "custom element"],
    "GraphQL":         ["graphql", "apollo", "relay"],
    "Accessibility":   ["wcag", "a11y", "aria", "inclusive design"],
    "WCAG":            ["accessibility", "a11y", "section 508"],
    "Delta Lake":      ["delta lake", "delta table"],
    "Spring Boot":     ["springboot", "spring framework", "spring mvc"],
    "GitHub Actions":  ["github actions", "gh actions", "gha"],
    "CloudFormation":  ["cloudformation", "cfn", "aws infrastructure"],
    "PostgreSQL":      ["postgres", "postgresql", "psql"],
    "Kubernetes":      ["k8s", "k8", "kube", "kubectl", "helm"],
    "Terraform":       ["tf", "hcl", "infrastructure as code", "iac"],
    "Docker":          ["containerization", "dockerfile", "docker-compose"],
    "TensorFlow":      ["tf", "keras", "tensorflow"],
    "PyTorch":         ["torch", "pytorch"],
}


def skill_in_text(skill: str, text: str) -> bool:
    """Fuzzy match — used for pairing/overlap scoring only."""
    t = text.lower()
    if skill.lower() in t:
        return True
    for alias in SKILL_ALIASES.get(skill, []):
        if alias.lower() in t:
            return True
    return False


def skill_explicitly_in_text(skill: str, text: str) -> bool:
    """
    Strict match — used for pre-computing the audit and for validation.
    Requires the skill name itself (or a primary alias) to appear as a
    word boundary match, not just any substring.
    """
    t = text.lower()
    skill_lower = skill.lower()
    if re.search(r'\b' + re.escape(skill_lower) + r'\b', t):
        return True
    # Allow a small set of well-known canonical aliases (not the full fuzzy list)
    canonical = {
        "Apache Spark": ["pyspark", "spark sql"],
        "PostgreSQL":   ["postgres"],
        "Kubernetes":   ["k8s"],
        "ETL/ELT":      ["etl", "elt"],
        "CI/CD":        ["ci/cd", "cicd"],
        "REST APIs":    ["rest api", "restful"],
        "Next.js":      ["nextjs"],
        "Node.js":      ["nodejs"],
        "PyTorch":      ["pytorch", "torch"],
        "TensorFlow":   ["tensorflow", "keras"],
    }
    for alias in canonical.get(skill, []):
        if re.search(r'\b' + re.escape(alias) + r'\b', t):
            return True
    return False


def skills_present_in(skill_list: list, text: str) -> list[str]:
    return [s for s in skill_list if skill_in_text(s, text)]


def skill_overlap_ratio(resume: dict, required_skills: list) -> float:
    """Fraction of required_skills present in resume text (fuzzy, for pairing)."""
    if not required_skills:
        return 1.0
    text = resume.get("text", "")
    found = sum(1 for s in required_skills if skill_in_text(s, text))
    return found / len(required_skills)


# ---------------------------------------------------------------------------
# Pre-computed audit — the key anti-hallucination mechanism
# ---------------------------------------------------------------------------

def compute_audit(candidate: dict, required_skills: list, nice_to_have: list) -> dict:
    """
    Compute the skill audit at pair-build time using the strict matcher.
    This is injected into the prompt so the model never has to derive
    skill presence/absence from prose — eliminating the main hallucination source.
    """
    text = candidate.get("text", "")
    present  = [s for s in required_skills if skill_explicitly_in_text(s, text)]
    missing  = [s for s in required_skills if s not in present]
    nice_hit = [s for s in nice_to_have    if skill_explicitly_in_text(s, text)]
    return {
        "required_present": present,
        "required_missing":  missing,
        "nice_to_have_present": nice_hit,
    }


def build_candidate_block(label: str, candidate: dict,
                           required_skills: list, nice_to_have: list) -> str:
    """
    Builds the structured candidate block injected into the ranking prompt.
    The pre-computed audit is the authoritative source — the model must not
    re-derive skill presence from the resume text.
    """
    audit = compute_audit(candidate, required_skills, nice_to_have)

    # Store audit on candidate dict so validation can access it without recomputing
    candidate["_audit"] = audit

    missing_str  = ", ".join(audit["required_missing"])  or "none"
    present_str  = ", ".join(audit["required_present"])  or "none"
    nice_str     = ", ".join(audit["nice_to_have_present"]) or "none"

    return (
        f"Candidate {label}:\n"
        f"  [METADATA — treat as ground truth]\n"
        f"  years_experience        : {candidate.get('years_experience', 'unknown')}\n"
        f"  seniority               : {candidate.get('seniority', 'unknown')}\n"
        f"  domain                  : {candidate.get('domain', 'unknown')}\n"
        f"  required_skills_present : {present_str}\n"
        f"  required_skills_missing : {missing_str}\n"
        f"  nice_to_have_present    : {nice_str}\n"
        f"  [RESUME TEXT]\n"
        f"{candidate['text']}\n"
    )


def build_jd_block(jd: dict) -> str:
    """Structured JD header injected above prose so the model has unambiguous constraints."""
    return (
        f"JOB DESCRIPTION:\n"
        f"  [METADATA — treat as ground truth]\n"
        f"  seniority       : {jd.get('seniority', 'unknown')}\n"
        f"  min_years       : {jd.get('min_years', 'unknown')}\n"
        f"  max_years       : {jd.get('max_years', 'unknown')}\n"
        f"  required_skills : {jd.get('required_skills', [])}\n"
        f"  [JD TEXT]\n"
        f"{jd['text']}\n"
    )


# ---------------------------------------------------------------------------
# Rubric — gap contradiction resolved, Step 1 updated, floor handling added
# ---------------------------------------------------------------------------

RANKING_RUBRIC = """
Rank ALL candidates based on their fit for the job description.

SCORING PROCESS (MANDATORY STEPS):
  Step 1. Skill Audit: Use ONLY the pre-computed [METADATA] in each candidate block.
          The fields required_skills_present and required_skills_missing are AUTHORITATIVE.
          Do NOT re-read the resume text to derive skill presence. Do NOT override the metadata.
  Step 2. Deductions: List all deductions with point values.
  Step 3. Math: 100 - sum(deductions). Floor at 0.
  Step 4. Cap: Apply ceiling if required. The cap is a MAXIMUM — never raise a score to meet it.
  Step 5. MANDATORY DIFFERENTIATION:
          If any two candidates have scores within 10 points of each other, you MUST adjust.
          Keep the better candidate's score unchanged.
          Subtract 10 from the lower candidate's score.
          If subtraction would push below 0, set to max(0, lower - 10) but never below 0.
          Use nice_to_have_present count, then years_experience to decide who is "better".
          Repeat until ALL adjacent gaps are >= 10 points.

SCORING RUBRIC:
  - Missing REQUIRED skill (each):             -15 pts
  - Under minimum experience (per year short): -10 pts per year
  - Over max experience by 1-3 years:          -10 pts total
  - Over max experience by 3+ years:           -25 pts total
  - Seniority Violation (Staff/Principal):     -20 pts ALWAYS
  - Domain mismatch:                           -35 pts

ABSOLUTE CAPS:
  - Missing ANY required skill:  final score <= 70 (hard ceiling)
  - Domain mismatch:             final score <= 40 (hard ceiling)
  If a candidate hits multiple caps, apply the LOWEST applicable ceiling.

NO INFERENCE RULE:
  Skill presence is determined solely by the pre-computed metadata.
  The words "assumed", "guessed", "likely has", "probably has" are FORBIDDEN
  when referring to skills. You MAY use "likely" or "implied" for seniority
  or experience reasoning only.

ABSOLUTE RULES:
  1. Scores strictly descending in ranking order.
  2. MINIMUM GAP: 10 points between every adjacent rank, no exceptions.
  3. NO TIED SCORES under any circumstances.
  4. Total spread (highest - lowest) >= 20 points.
  5. Use years_experience and seniority from metadata. Do not infer from prose.

REASON FORMAT (mandatory for every candidate):
  "[Audit: <skill>: P/M, ...]. [Deductions: <item: pts>, ...]. [Math: 100 - <sum> = <val>]. [Cap: MIN(<val>, <ceiling>) = <capped>]. [Diff: <adjustment or 'None'>]. [Summary: <one sentence>]."

  The Math line must be arithmetically correct.
  The Cap line must reflect the score you actually assign.

Return ONLY valid JSON. Start with { end with }. No other text.
{
  "reasons": { "A": "...", "B": "..." },
  "scores":  { "A": 75, "B": 60 },
  "ranking": ["A", "B"]
}
"""

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self._lock                    = threading.Lock()
        self.api_calls                = 0
        self.jds_generated            = 0
        self.resumes_generated        = 0
        self.pairs_built              = 0
        self.rankings_done            = 0
        self.samples_written          = 0
        self.composition_counts       = defaultdict(int)
        self.validation_failures      = defaultdict(int)
        self.generation_failures      = defaultdict(int)
        # Per-composition failure tracking
        self.composition_failures     = defaultdict(lambda: defaultdict(int))

    def inc(self, field: str, amount: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def inc_composition(self, name: str):
        with self._lock:
            self.composition_counts[name] += 1

    def fail_validation(self, reason: str, composition: str = "unknown"):
        with self._lock:
            self.validation_failures[reason] += 1
            self.composition_failures[composition][reason] += 1

    def fail_generation(self, stage: str):
        with self._lock:
            self.generation_failures[stage] += 1

    def summary(self, timer: Timer = None) -> str:
        border = f"{Color.DIM}{'=' * 62}{Color.RESET}"
        lines = [
            "\n" + border,
            f"  {Color.BOLD}{Color.CYAN}PIPELINE EXECUTION SUMMARY{Color.RESET}",
            border,
            f"  API calls made         : {Color.BOLD}{self.api_calls}{Color.RESET}",
            f"  JDs generated          : {Color.GREEN}{self.jds_generated}{Color.RESET}",
            f"  Resumes generated      : {Color.GREEN}{self.resumes_generated}{Color.RESET}",
            f"  Pairs built            : {Color.GREEN}{self.pairs_built}{Color.RESET}",
            f"  Rankings completed     : {Color.GREEN}{self.rankings_done}{Color.RESET}",
            f"  Samples written        : {Color.BOLD}{Color.GREEN}{self.samples_written}{Color.RESET}",
        ]

        if timer:
            lines.append(f"  Total time elapsed     : {Color.YELLOW}{timer.format_duration(timer.total())}{Color.RESET}")
            if timer.stages:
                lines.append(f"\n  {Color.BOLD}Timing per stage:{Color.RESET}")
                for name, duration in timer.stages.items():
                    lines.append(f"    {name:<25} : {timer.format_duration(duration)}")

        if self.composition_counts:
            lines.append(f"\n  {Color.BOLD}Composition breakdown:{Color.RESET}")
            for k, v in sorted(self.composition_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:<25} : {v}")

        if self.generation_failures:
            lines.append(f"\n  {Color.BOLD}{Color.YELLOW}Generation failures:{Color.RESET}")
            for k, v in sorted(self.generation_failures.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:<45} : {v}")

        if self.validation_failures:
            lines.append(f"\n  {Color.BOLD}{Color.RED}Validation rejections (global):{Color.RESET}")
            for k, v in sorted(self.validation_failures.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:<55} : {v}")

        if self.composition_failures:
            lines.append(f"\n  {Color.BOLD}{Color.RED}Validation rejections by composition:{Color.RESET}")
            for comp, failures in sorted(self.composition_failures.items()):
                total_fails = sum(failures.values())
                lines.append(f"    {comp} ({total_fails} total):")
                for reason, count in sorted(failures.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"      {reason:<50} : {count}")

        lines.append(border)
        return "\n".join(lines)


stats = Stats()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

class CerebrasClient:
    def __init__(self):
        self.keys = []
        if os.getenv("CEREBRAS_API_KEYS"):
            self.keys.extend([k.strip() for k in os.getenv("CEREBRAS_API_KEYS").split(",") if k.strip()])

        base_key = os.getenv("CEREBRAS_API_KEY")
        if base_key and base_key not in self.keys:
            self.keys.append(base_key)
        for i in range(1, 21):
            key = os.getenv(f"CEREBRAS_API_KEY_{i}")
            if not key:
                if i > 4: break
                continue
            if key not in self.keys:
                self.keys.append(key)

        if not self.keys:
            raise ValueError("No CEREBRAS_API_KEY or CEREBRAS_API_KEYS found in .env")

        print(f"  CerebrasClient initialized with {len(self.keys)} keys.")
        self.clients = [OpenAI(base_url=BASE_URL, api_key=k) for k in self.keys]
        self._current_index = 0
        self._lock = threading.Lock()

    def get_client(self):
        with self._lock:
            client = self.clients[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.clients)
            return client

    def call(self, messages: list, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        last_err = None
        for attempt in range(MAX_RETRIES):
            client = self.get_client()
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Empty response")
                stats.inc("api_calls")
                return content.strip()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    wait_time = RATE_WAIT / len(self.keys) if len(self.keys) > 1 else RATE_WAIT
                    tqdm.write(f"  Rate limit (key {self._current_index}): waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                if attempt == MAX_RETRIES - 1:
                    print(f"  Cerebras Error: {str(e)}")
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Cerebras call failed after {MAX_RETRIES} retries. Last: {last_err}")


class GeminiClient:
    def __init__(self):
        self.keys = []
        if os.getenv("GEMINI_API_KEYS"):
            self.keys.extend([k.strip() for k in os.getenv("GEMINI_API_KEYS").split(",") if k.strip()])

        i = 1
        while True:
            suffix = f"_{i}" if i > 1 else ""
            key = os.getenv(f"GEMINI_API_KEY{suffix}")
            if not key:
                if i > 1: break
                else: i += 1; continue
            if key not in self.keys:
                self.keys.append(key)
            i += 1
            if i > 20: break

        if not self.keys:
            raise ValueError("No GEMINI_API_KEY or GEMINI_API_KEYS found in .env")

        print(f"  GeminiClient initialized with {len(self.keys)} keys.")
        self._current_index = 0
        self._lock = threading.Lock()
        GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
        self.model = GEMINI_MODEL

    def get_key(self):
        with self._lock:
            key = self.keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.keys)
            return key

    def call(self, messages: list, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        system_text = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user_text   = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        full_text   = f"{system_text}\n\n{user_text}"

        for attempt in range(MAX_RETRIES):
            key = self.get_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": full_text}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            }
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    content = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    if not content:
                        raise ValueError("Empty response")
                    stats.inc("api_calls")
                    return content.strip()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  Gemini Error after {MAX_RETRIES} attempts: {str(e)}")
                    raise
                time.sleep(2 ** attempt)
        return ""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """
    Attempt to repair truncated JSON using a stack-based brace counter.
    Handles nested objects/arrays, not just single-level truncation.
    """
    stack = []
    in_string = False
    escape_next = False
    last_valid = 0

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch in ('}', ']'):
            if stack:
                stack.pop()
                if not stack:
                    last_valid = i + 1

    if not stack:
        return text  # Already balanced

    # Close all open structures in reverse order
    closers = {'{': '}', '[': ']'}
    repair = text[:last_valid] if last_valid else text
    for opener in reversed(stack):
        repair += closers[opener]
    return repair


def extract_object(text: str) -> dict | None:
    """Finds the last occurring JSON object. Handles CoT output and truncation."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown fenced block first
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    # Find all top-level { } blocks, try the last one
    matches = list(re.finditer(r"\{.*\}", text, re.DOTALL))
    if not matches:
        return None

    json_str = matches[-1].group().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Attempt stack-based repair
    try:
        repaired = _repair_json(json_str)
        return json.loads(repaired)
    except Exception:
        return None


def extract_array(text: str) -> list | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        try:
            repaired = _repair_json(match.group())
            return json.loads(repaired)
        except Exception:
            return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Stage 1: Generate JDs
# ---------------------------------------------------------------------------

JD_SYSTEM = """You are a senior engineering hiring manager writing realistic job descriptions.
Your output trains an AI technical recruiter.
Write professionally but directly — no buzzwords like 'rockstar', 'ninja', 'synergy', 'fast-paced'.
Required skills must appear naturally inside responsibilities, not just listed at the end.
Return ONLY valid JSON array. No markdown. No explanation."""


def build_jd_batch_prompt(domain: str, count: int = 3) -> str:
    skills      = DOMAIN_SKILLS[domain]
    seniorities = random.choices(
        list(SENIORITY_CONFIG.keys()),
        weights=[v["weight"] for v in SENIORITY_CONFIG.values()],
        k=count
    )
    configs = []
    for seniority in seniorities:
        cfg  = SENIORITY_CONFIG[seniority]
        req  = random.sample(skills, min(6, len(skills)))
        nice = random.sample(
            [s for s in skills if s not in req],
            min(3, len(skills) - len(req))
        )
        company = get_random_companies(1)[0]
        ceiling = ""
        if seniority in ("Senior", "Mid-level"):
            ceiling = (
                f'End with: "Strict Seniority Ceiling: Do not apply if you exceed '
                f'{cfg["max_years"] + 1} years or hold a Staff or Principal title."'
            )
        configs.append({
            "company": company, "seniority": seniority, "cfg": cfg,
            "req": req, "nice": nice, "ceiling": ceiling,
        })

    items = [
        f'{{"company":"{c["company"]}","seniority":"{c["seniority"]}",'
        f'"min_years":{c["cfg"]["min_years"]},"max_years":{c["cfg"]["max_years"]},'
        f'"required_skills":{json.dumps(c["req"])},"nice_to_have":{json.dumps(c["nice"])},'
        f'"ceiling_instruction":"{c["ceiling"]}"}}'
        for c in configs
    ]

    return f"""Generate a JSON array of exactly {count} job descriptions for {domain.replace('_', ' ')} roles.

Configs (one JD per config):
[{", ".join(items)}]

For EACH config:
- Company context: 1 sentence what the company builds
- Role summary: 2-3 sentences on what this engineer owns
- Responsibilities: 3-4 bullets, each mentioning a required skill IN CONTEXT
- Required skills section: list all required_skills BY EXACT NAME as given in the config
- Nice-to-have section: list nice_to_have
- Experience: state exact range "min_years-max_years years"
- Apply ceiling_instruction if not empty
- Length: 150-220 words

Return JSON array, each item:
{{
  "text": "<full job description>",
  "domain": "{domain}",
  "seniority": "<from config>",
  "min_years": <number>,
  "max_years": <number>,
  "required_skills": ["skill1", ...],
  "nice_to_have": ["skill1", ...]
}}

Start with [ end with ]. No other text."""


def generate_jd_batch(client, domain, count: int = 3, temperature: float = 0.8) -> list[dict]:
    prompt   = build_jd_batch_prompt(domain, count)
    messages = [
        {"role": "system", "content": JD_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    raw    = client.call(messages, max_tokens=3000, temperature=temperature)
    parsed = extract_array(raw)
    if not parsed:
        stats.fail_generation("jd_batch_json_parse")
        return []

    valid = []
    for item in parsed:
        if not all(k in item for k in ["text", "domain", "seniority", "required_skills"]):
            stats.fail_generation("jd_missing_fields")
            continue
        if len(item["text"].split()) < JD_MIN_WORDS:
            stats.fail_generation("jd_too_short")
            continue
        if len(item.get("required_skills", [])) < 3:
            stats.fail_generation("jd_insufficient_skills")
            continue
        # Ensure nice_to_have exists
        if "nice_to_have" not in item:
            item["nice_to_have"] = []
        valid.append(item)

    stats.inc("jds_generated", len(valid))
    return valid


# ---------------------------------------------------------------------------
# Stage 2: Generate resumes
# ---------------------------------------------------------------------------

RESUME_SYSTEM = """You are a professional resume writer creating realistic software engineering resumes.
Your output trains an AI technical recruiter.
Each resume must be distinct in writing style, skill depth, and career trajectory.
Skills must appear in work history through concrete achievements — not just listed at the bottom.
IMPORTANT: When you list a skill, use its EXACT canonical name (e.g., "Apache Spark" not "Spark",
"ETL/ELT" not "ETL", "REST APIs" not "REST"). This is critical for downstream processing.
Return ONLY valid JSON array. No markdown. No explanation."""


def build_resume_batch_prompt(domain: str, seniority: str, count: int = 5) -> str:
    skills = DOMAIN_SKILLS[domain]
    cfg    = SENIORITY_CONFIG[seniority]
    # Sample with replacement so styles are independent per resume
    styles = [random.choice(WRITING_STYLES) for _ in range(count)]

    skill_subsets = [
        random.sample(skills, random.randint(4, min(8, len(skills))))
        for _ in range(count)
    ]
    config_strs = [
        f"Resume {i+1}: focus_skills={json.dumps(subset)}, style={styles[i]}"
        for i, subset in enumerate(skill_subsets)
    ]

    return f"""Generate a JSON array of exactly {count} realistic software engineering resumes.

Domain: {domain.replace('_', ' ')}
Seniority: {seniority} ({cfg['min_years']}–{cfg['max_years']} years experience)
Companies: {', '.join(get_random_companies(10))}

Per-resume config:
{chr(10).join(config_strs)}

RULES:
1. Show each focus skill used at a specific role — not just listed
2. Use EXACT canonical skill names as given in focus_skills (e.g. "Apache Spark" not "Spark")
3. State years of experience explicitly within {cfg['min_years']}–{cfg['max_years']}
4. 2–3 past roles at these companies
5. At least 2 measurable outcomes — varied percentages (23%, 41%, 67%)
6. Each resume distinct — different companies, career paths, emphasis
7. Length: 140–220 words
8. Follow the writing style specified for each resume

Return JSON array, each item:
{{
  "text": "<full resume>",
  "domain": "{domain}",
  "seniority": "{seniority}",
  "years_experience": <number within {cfg['min_years']}-{cfg['max_years']}>,
  "skills": ["skill1", ...]
}}

Start with [ end with ]. No other text."""


def build_hard_negative_batch_prompt(domain: str, count: int = 5) -> str:
    profiles = random.choices(HARD_NEGATIVE_PROFILES, k=count)
    styles   = [random.choice(WRITING_STYLES) for _ in range(count)]

    # Use ONLY domain-specific skills for the exclusion list — not all required skills
    # from any particular JD. This avoids over-filtering valid hard negatives that
    # happen to mention generic terms like SQL.
    domain_specific = [
        s for s in DOMAIN_SKILLS[domain]
        if s not in {"SQL", "Python", "Docker", "Linux", "Git"}  # exclude cross-domain generics
    ]
    exclusion_sample = random.sample(domain_specific, min(8, len(domain_specific)))

    config_strs = [
        f"Resume {i+1}: type={p['type']}, skills={json.dumps(p['skills'])}, style={styles[i]}"
        for i, p in enumerate(profiles)
    ]

    return f"""Generate a JSON array of exactly {count} HARD NEGATIVE resumes.
These are completely irrelevant candidates for {domain.replace('_', ' ')} jobs.

Per-resume configs:
{chr(10).join(config_strs)}

Companies: {', '.join(get_random_companies(10))}

RULES:
1. Each candidate works in their specified type — NOT software engineering
2. These domain-specific skills must NOT appear anywhere:
   {', '.join(exclusion_sample)}
3. Domain mismatch must be obvious to a recruiter
4. Realistic, professionally written
5. 2–3 roles at these companies
6. At least 1 measurable outcome in their actual domain
7. Length: 130–200 words
8. Follow the writing style specified for each resume

Return JSON array, each item:
{{
  "text": "<full resume>",
  "domain": "other",
  "seniority": "Mid-level",
  "years_experience": <number 2-8>,
  "skills": ["their actual non-engineering skills"]
}}

Start with [ end with ]. No other text."""


def _parse_batch(raw: str, domain_override: str = None) -> list[dict]:
    parsed = extract_array(raw)
    if not parsed:
        stats.fail_generation("batch_json_parse")
        return []
    valid = []
    for item in parsed:
        if len(item.get("text", "").split()) < RESUME_MIN_WORDS:
            stats.fail_generation("resume_too_short")
            continue
        if not item.get("skills"):
            stats.fail_generation("resume_no_skills")
            continue
        if domain_override:
            item["domain"] = domain_override
        valid.append(item)
    stats.inc("resumes_generated", len(valid))
    return valid


def generate_resume_batch(client, domain, seniority, count=5, temperature=0.82):
    prompt   = build_resume_batch_prompt(domain, seniority, count)
    messages = [{"role": "system", "content": RESUME_SYSTEM},
                {"role": "user",   "content": prompt}]
    raw = client.call(messages, max_tokens=4000, temperature=temperature)
    return _parse_batch(raw)


def generate_hard_negative_batch(client, domain, count=5, temperature=0.8):
    prompt   = build_hard_negative_batch_prompt(domain, count)
    messages = [{"role": "system", "content": RESUME_SYSTEM},
                {"role": "user",   "content": prompt}]
    raw = client.call(messages, max_tokens=4000, temperature=temperature)
    return _parse_batch(raw, domain_override="other")


# ---------------------------------------------------------------------------
# Stage 3: Pair JDs with resumes
# ---------------------------------------------------------------------------

def pick_composition() -> tuple[str, dict]:
    names   = list(SAMPLE_COMPOSITIONS.keys())
    weights = [SAMPLE_COMPOSITIONS[n]["weight"] for n in names]
    name    = random.choices(names, weights=weights, k=1)[0]
    return name, SAMPLE_COMPOSITIONS[name]


def pair_jds_and_resumes(jds: list[dict], resumes: list[dict], target: int,
                          min_overlap: float = MIN_BEST_OVERLAP_RATIO,
                          max_reuse: int = MAX_RESUME_REUSE) -> list[dict]:
    by_domain: dict[str, list] = defaultdict(list)
    hard_negs: list            = []

    for r in resumes:
        if r.get("domain") == "other":
            hard_negs.append(r)
        else:
            by_domain[r["domain"]].append(r)

    resume_usage: dict[int, int] = defaultdict(int)
    pairs = []
    skipped_overlap = 0
    skipped_pool    = 0
    attempts        = 0
    max_attempts    = target * 3

    while len(pairs) < target and attempts < max_attempts:
        random.shuffle(jds)
        found_in_round = 0

        for jd in jds:
            if len(pairs) >= target:
                break
            attempts += 1

            domain          = jd["domain"]
            required_skills = jd.get("required_skills", [])
            raw_pool        = by_domain.get(domain, [])
            pool            = [r for r in raw_pool if resume_usage[id(r)] < max_reuse]

            pool_with_scores = [
                (r, skill_overlap_ratio(r, required_skills)) for r in pool
            ]
            pool_with_scores.sort(key=lambda x: x[1], reverse=True)

            if not pool_with_scores or pool_with_scores[0][1] < min_overlap:
                skipped_overlap += 1
                continue

            threshold     = pool_with_scores[0][1] * 0.5
            eligible_pool = [r for r, score in pool_with_scores if score >= threshold]

            if len(eligible_pool) < 2:
                skipped_pool += 1
                continue

            comp_name, comp = pick_composition()
            n_domain  = comp["n_domain"]
            n_hardneg = comp["n_hardneg"]

            if len(eligible_pool) < n_domain:
                n_domain  = min(len(eligible_pool), 3)
                n_hardneg = 0
                comp_name = "three_same_domain"

            # Hard negative filter: use domain-specific skills only
            domain_specific_skills = [
                s for s in required_skills
                if s in DOMAIN_SKILLS.get(domain, [])
                and s not in {"SQL", "Python", "Docker", "Linux", "Git"}
            ]
            clean_hardnegs = [
                r for r in hard_negs
                if len(skills_present_in(domain_specific_skills, r.get("text", ""))) < 2
                and resume_usage[id(r)] < max_reuse
            ]

            if n_hardneg > 0 and len(clean_hardnegs) < n_hardneg:
                n_hardneg = 0
                comp_name = "three_same_domain"

            n_to_pick     = min(n_domain, len(eligible_pool))
            domain_picks  = random.sample(eligible_pool, n_to_pick)
            hardneg_picks = random.choices(clean_hardnegs, k=n_hardneg) if n_hardneg > 0 else []
            candidates    = domain_picks + hardneg_picks

            random.shuffle(candidates)
            for r in candidates:
                resume_usage[id(r)] += 1

            pairs.append({
                "jd":          jd,
                "candidates":  candidates,
                "composition": comp_name,
            })
            found_in_round += 1

        if found_in_round == 0:
            status_msg(
                f"  Yield exhausted: built {len(pairs)} pairs before hitting reuse caps.",
                Color.YELLOW
            )
            break

    if skipped_overlap or skipped_pool:
        status_msg(
            f"  Pairing quality gates: {skipped_overlap} JDs skipped (low overlap), "
            f"{skipped_pool} skipped (pool too small after filter)",
            Color.YELLOW, bold=False
        )

    stats.inc("pairs_built", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Stage 4: Rank
# ---------------------------------------------------------------------------

def rank_pair(client, pair, temperature=0.0) -> dict | None:
    jd         = pair["jd"]
    candidates = pair["candidates"]
    labels     = [chr(65 + i) for i in range(len(candidates))]
    nice_to_have = jd.get("nice_to_have", [])

    # Build structured JD block
    jd_block = build_jd_block(jd)

    # Build candidate blocks with pre-computed audits
    candidate_section = ""
    for label, candidate in zip(labels, candidates):
        candidate_section += build_candidate_block(label, candidate, jd.get("required_skills", []), nice_to_have) + "\n"

    user_content = (
        f"{RANKING_RUBRIC}\n"
        f"{jd_block}\n"
        f"RESUMES:\n{candidate_section}"
    ).strip()

    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    raw    = client.call(messages, max_tokens=2000, temperature=temperature)
    parsed = extract_object(raw)
    if parsed:
        stats.inc("rankings_done")
    return parsed


# ---------------------------------------------------------------------------
# Stage 5: Validation
# ---------------------------------------------------------------------------

def validate_strict(output: dict, labels: list[str],
                    candidates: list[dict], jd: dict,
                    composition: str = "unknown") -> tuple[bool, str]:
    """
    Validates ranking output. Key additions over v1:
    - Math consistency: verifies 100 - deductions = stated result
    - Cap consistency: verifies cap value matches assigned score
    - hallucinated_missing_skill now uses pre-computed audit, not fuzzy match
    - Composition passed through for per-composition failure logging
    """
    ranking         = output.get("ranking", [])
    scores          = output.get("scores",  {})
    reasons         = output.get("reasons", {})
    expected        = set(labels)
    required_skills = jd.get("required_skills", [])
    label_to_cand   = {chr(65 + i): c for i, c in enumerate(candidates)}

    # 1. Structure
    if set(ranking) != expected:
        return False, "ranking_labels_mismatch"
    if set(scores.keys()) != expected:
        return False, "score_keys_mismatch"
    if set(reasons.keys()) != expected:
        return False, "reason_keys_mismatch"

    # 2. Score range
    for label, score in scores.items():
        if not isinstance(score, (int, float)) or not (-100 <= score <= 100):
            return False, f"score_out_of_range_{label}_{score}"

    vals = list(scores.values())

    # 3. No ties
    if len(set(vals)) != len(vals):
        return False, "duplicate_scores"

    # 4. Descending order
    ranked_scores = [scores[c] for c in ranking]
    if ranked_scores != sorted(ranked_scores, reverse=True):
        return False, "scores_not_descending"

    # 5. Minimum spread
    spread = max(vals) - min(vals)
    if spread < 20:
        return False, f"spread_too_low_{spread}"

    # 6. Adjacent gap >= 10 (aligned with rubric — was 5 in v1)
    for i in range(len(ranked_scores) - 1):
        gap = ranked_scores[i] - ranked_scores[i + 1]
        if gap < 10:
            return False, f"adjacent_gap_too_small_rank{i+1}_{i+2}_{gap}pts"

    # 7. Reason quality, format, and math consistency
    for label, reason in reasons.items():
        if len(reason.split()) < 15:
            return False, f"reason_too_short_{label}"

        reason_lower = reason.lower()

        # Mandatory brackets
        for bracket in ["[audit:", "[deductions:", "[math:", "[cap:", "[summary:"]:
            if bracket not in reason_lower:
                return False, f"reason_missing_bracket_{label}_{bracket}"

        # Math consistency check: [Math: 100 - X = Y] must be correct
        math_match = re.search(r"\[math:\s*100\s*-\s*(\d+)\s*=\s*(\d+)\]", reason_lower)
        if math_match:
            deductions    = int(math_match.group(1))
            stated_result = int(math_match.group(2))
            expected_result = max(0, 100 - deductions)
            if abs(expected_result - stated_result) > 1:  # allow 1pt rounding
                return False, f"math_inconsistency_{label}_100-{deductions}={stated_result}_expected_{expected_result}"

        # Cap consistency check: score assigned must be <= any stated cap
        cap_match = re.search(r"\[cap:\s*min\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)\]", reason_lower)
        if cap_match:
            stated_cap_result = int(cap_match.group(3))
            actual_score      = scores[label]
            if abs(stated_cap_result - actual_score) > 1:
                return False, f"cap_score_mismatch_{label}_stated_{stated_cap_result}_actual_{actual_score}"

        # Skill reference check
        cand_skills   = [s.lower() for s in label_to_cand[label].get("skills", [])]
        req_lower     = [s.lower() for s in required_skills]
        all_skills    = req_lower + cand_skills
        found_skill   = any(s in reason_lower for s in all_skills)
        if not found_skill:
            for skill in required_skills + label_to_cand[label].get("skills", []):
                for alias in SKILL_ALIASES.get(skill, []):
                    if alias.lower() in reason_lower:
                        found_skill = True
                        break
        if not found_skill:
            return False, f"reason_no_skill_reference_{label}"

        # Hallucination check: uses pre-computed audit, not fuzzy re-derivation
        # If the model says a skill is missing but our audit says it's present, reject.
        audit = label_to_cand[label].get("_audit", {})
        audit_present = [s.lower() for s in audit.get("required_present", [])]
        audit_missing = [s.lower() for s in audit.get("required_missing", [])]

        for skill in required_skills:
            skill_lower = skill.lower()
            # Model claims missing — but audit says present
            negative_contexts = [
                f"missing {skill_lower}",
                f"not find {skill_lower}",
                f"no {skill_lower} experience",
                f"lack of {skill_lower}",
                f"{skill_lower}: m",   # catches "[Audit: Airflow: M]" style
            ]
            model_says_missing = any(ctx in reason_lower for ctx in negative_contexts)
            if model_says_missing and skill_lower in audit_present:
                return False, f"hallucinated_missing_skill_{label}_{skill}"

            # Model claims present — but audit says missing
            positive_contexts = [
                f"{skill_lower}: p",   # catches "[Audit: Airflow: P]" style
                f"present: {skill_lower}",
            ]
            model_says_present = any(ctx in reason_lower for ctx in positive_contexts)
            if model_says_present and skill_lower in audit_missing:
                return False, f"hallucinated_present_skill_{label}_{skill}"

    # 8. Logic checks
    for label, candidate in label_to_cand.items():
        score = scores[label]
        audit = candidate.get("_audit", {})

        # Domain mismatch
        if candidate.get("domain") == "other":
            if label == ranking[0]:
                return False, "hard_negative_ranked_first"
            if score > 40:
                return False, f"hard_negative_score_too_high_{label}_{score}"

        # Missing skill cap — use pre-computed audit
        if audit.get("required_missing") and score > 70:
            return False, f"cap_violation_missing_skill_{label}_{score}_expected_le_70"

        # YOE metadata check
        cand_yoe = candidate.get("years_experience", 0)
        min_yoe  = jd.get("min_years", 0)
        if cand_yoe and min_yoe and cand_yoe < min_yoe and score > 70:
            return False, f"metadata_yoe_underqualified_{label}_{cand_yoe}_vs_min_{min_yoe}"

        # Seniority ceiling
        if "seniority ceiling" in jd.get("text", "").lower() or jd.get("seniority") in ("Senior", "Mid-level"):
            cand_seniority = candidate.get("seniority", "")
            if cand_seniority in ("Staff", "Principal", "Director") and score > 70:
                return False, f"metadata_overqualified_ceiling_violation_{label}_{cand_seniority}"

    # 9. Best candidate floor
    if scores[ranking[0]] < 20:
        return False, f"best_candidate_score_too_low_{ranking[0]}_{scores[ranking[0]]}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Process one pair
# ---------------------------------------------------------------------------

def process_pair(client, pair: dict,
                 out_path: Path, write_lock: threading.Lock,
                 dedup_hashes: set = None, temperature=0.0) -> bool:
    jd          = pair["jd"]
    candidates  = pair["candidates"]
    composition = pair.get("composition", "unknown")
    labels      = [chr(65 + i) for i in range(len(candidates))]
    nice_to_have = jd.get("nice_to_have", [])

    # Deduplication
    cand_texts_sorted = tuple(sorted(c.get("text", "")[:200] for c in candidates))
    dedup_key = hashlib.sha256(
        (jd.get("text", "")[:300] + "||" + "||".join(cand_texts_sorted)).encode()
    ).hexdigest()
    if dedup_hashes is not None:
        with write_lock:
            if dedup_key in dedup_hashes:
                stats.fail_validation("duplicate_pair", composition)
                return False
            dedup_hashes.add(dedup_key)

    ranking_output = rank_pair(client, pair, temperature=temperature)
    if not ranking_output:
        stats.fail_generation("ranking_json_parse_failed")
        return False

    is_valid, reason = validate_strict(ranking_output, labels, candidates, jd, composition)
    if not is_valid:
        stats.fail_validation(reason, composition)
        failure_record = {
            "val_reason":    reason,
            "composition":   composition,
            "ranking_output": ranking_output,
            "jd_required":   jd.get("required_skills", []),
            "candidates": [
                {
                    "label":   labels[i],
                    "domain":  c.get("domain"),
                    "skills":  c.get("skills"),
                    "audit":   c.get("_audit", {}),
                    "yoe":     c.get("years_experience"),
                    "seniority": c.get("seniority"),
                }
                for i, c in enumerate(candidates)
            ]
        }
        with write_lock:
            append_jsonl(DATA_DIR / "failures.jsonl", failure_record)
        return False

    # Build the saved training record using the same structured prompt
    jd_block = build_jd_block(jd)
    candidate_section = ""
    for label, candidate in zip(labels, candidates):
        candidate_section += build_candidate_block(label, candidate, jd.get("required_skills", []), nice_to_have) + "\n"

    user_content = (
        f"{RANKING_RUBRIC}\n"
        f"{jd_block}\n"
        f"RESUMES:\n{candidate_section}"
    ).strip()

    record = {
        "pair_hash": pair.get("pair_hash"),
        "messages": [
            {"role": "system",    "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(ranking_output)},
        ]
    }

    with write_lock:
        append_jsonl(out_path, record)

    stats.inc("samples_written")
    stats.inc_composition(composition)
    return True


# ---------------------------------------------------------------------------
# Pipeline stage runners
# ---------------------------------------------------------------------------

def pick_domain(override: str = None) -> str:
    if override:
        return override
    return random.choices(
        list(DOMAIN_WEIGHTS.keys()),
        weights=list(DOMAIN_WEIGHTS.values()),
        k=1
    )[0]


def pick_seniority() -> str:
    return random.choices(
        list(SENIORITY_CONFIG.keys()),
        weights=[v["weight"] for v in SENIORITY_CONFIG.values()],
        k=1
    )[0]


def run_stage1(client, target_jds, domain_override, workers, timer: Timer, temperature=0.8):
    timer.start_stage("Stage 1: JD Generation")
    existing = load_jsonl(JDS_FILE)
    needed   = max(0, target_jds - len(existing))
    if needed == 0:
        status_msg(f"Stage 1: {len(existing)} JDs already generated, skipping.", Color.GREEN)
        timer.end_stage()
        return existing

    status_msg(f"Stage 1: Generating {needed} JDs (~{(needed+2)//3} calls)...")
    write_lock = threading.Lock()
    calls      = (needed + 2) // 3

    def gen(_):
        domain = pick_domain(domain_override)
        batch  = generate_jd_batch(client, domain, count=3, temperature=temperature)
        with write_lock:
            for jd in batch:
                append_jsonl(JDS_FILE, jd)
        return len(batch)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen, i) for i in range(calls)]
        for f in tqdm(as_completed(futs), total=calls, desc="  JD batches"):
            f.result()

    all_jds = load_jsonl(JDS_FILE)
    timer.end_stage()
    status_msg(f"Stage 1 done: {len(all_jds)} JDs | {stats.api_calls} calls so far", Color.GREEN)
    return all_jds


def run_stage2(client, jds, target_resumes, workers, timer: Timer, temperature=0.8):
    timer.start_stage("Stage 2: Resume Generation")
    existing = load_jsonl(RESUMES_FILE)
    needed   = max(0, target_resumes - len(existing))
    if needed == 0:
        status_msg(f"Stage 2: {len(existing)} resumes already generated, skipping.", Color.GREEN)
        timer.end_stage()
        return existing

    status_msg(f"Stage 2: Generating {needed} resumes (batch=5)...")
    write_lock    = threading.Lock()
    jds_by_domain = defaultdict(list)
    for jd in jds:
        jds_by_domain[jd["domain"]].append(jd)

    tasks     = []
    n_domains = len(jds_by_domain)
    n_batches = max(1, needed // (n_domains * 2 * 5))

    for domain in jds_by_domain:
        seniority = pick_seniority()
        for _ in range(n_batches):
            tasks.append(("resume",        domain, seniority))
            tasks.append(("hard_negative", domain, seniority))

    random.shuffle(tasks)

    def gen_batch(task):
        rtype, domain, seniority = task
        if rtype == "resume":
            batch = generate_resume_batch(client, domain, seniority, count=5, temperature=temperature)
        else:
            batch = generate_hard_negative_batch(client, domain, count=5, temperature=temperature)
        with write_lock:
            for r in batch:
                append_jsonl(RESUMES_FILE, r)
        return len(batch)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen_batch, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(tasks), desc="  Resume batches"):
            f.result()

    all_resumes = load_jsonl(RESUMES_FILE)
    timer.end_stage()
    status_msg(f"Stage 2 done: {len(all_resumes)} resumes | {stats.api_calls} calls so far", Color.GREEN)
    return all_resumes


def run_stage3(jds, resumes, target, timer: Timer,
               min_overlap: float = MIN_BEST_OVERLAP_RATIO,
               max_reuse: int = MAX_RESUME_REUSE):
    timer.start_stage("Stage 3: Pairing")
    existing = load_jsonl(PAIRS_FILE)
    if len(existing) >= target:
        status_msg(f"Stage 3: {len(existing)} pairs already built, skipping.", Color.GREEN)
        timer.end_stage()
        return existing[:target]

    status_msg(f"Stage 3: Pairing (overlap≥{min_overlap:.0%}, max-reuse={max_reuse})...")
    pairs = pair_jds_and_resumes(jds, resumes, target,
                                  min_overlap=min_overlap, max_reuse=max_reuse)
    for pair in pairs:
        append_jsonl(PAIRS_FILE, pair)

    status_msg(f"Stage 3 done: {len(pairs)} pairs", Color.GREEN)
    timer.end_stage()
    return pairs


def run_stage4(client, pairs, out_path, workers, timer: Timer, temperature=0.0):
    timer.start_stage("Stage 4: Ranking")

    ranked_hashes = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    h = obj.get("pair_hash")
                    if h:
                        ranked_hashes.add(h)
                except Exception:
                    continue

    remaining = []
    for p in pairs:
        cand_texts_sorted = tuple(sorted(c.get("text", "")[:400] for c in p["candidates"]))
        h = hashlib.sha256(
            (p["jd"].get("text", "")[:500] + "||" + "||".join(cand_texts_sorted)).encode()
        ).hexdigest()
        p["pair_hash"] = h
        if h not in ranked_hashes:
            remaining.append(p)

    if not remaining:
        status_msg(f"Stage 4: All {len(pairs)} samples already ranked, skipping.", Color.GREEN)
        timer.end_stage()
        return

    status_msg(f"Stage 4: Ranking {len(remaining)} pairs (temp={temperature}, workers={workers})...")
    write_lock   = threading.Lock()
    dedup_hashes = set()
    success = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_pair, client, p, out_path, write_lock,
                      dedup_hashes=dedup_hashes, temperature=temperature): p
            for p in remaining
        }
        with tqdm(total=len(remaining), desc="Ranking", unit="pair") as pbar:
            for future in as_completed(futs):
                ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                pbar.set_postfix(
                    ok=f"{Color.GREEN}{success}{Color.RESET}",
                    fail=f"{Color.RED}{failed}{Color.RESET}",
                    api=stats.api_calls
                )
                pbar.update(1)

    timer.end_stage()
    status_msg(
        f"Stage 4 done: {success} written, {failed} rejected",
        Color.GREEN if failed == 0 else Color.YELLOW
    )


# ---------------------------------------------------------------------------
# Stage 5: Score spread balancing (spread-based, not bucket-based)
# ---------------------------------------------------------------------------

def run_stage5_balance(out_path: Path, timer: Timer):
    """
    Trim samples with low score spread — these are low-signal pairs where
    candidates are too similar to produce useful training signal.
    Target: no more than 20% of samples should have spread < 25.
    """
    timer.start_stage("Stage 5: Spread Balancing")
    if not out_path.exists():
        status_msg("Stage 5: No output file found, skipping.", Color.YELLOW)
        timer.end_stage()
        return

    records = load_jsonl(out_path)
    if not records:
        timer.end_stage()
        return

    def get_spread(rec):
        try:
            ranking = json.loads(rec["messages"][-1]["content"])
            scores  = ranking.get("scores", {})
            if scores:
                return max(scores.values()) - min(scores.values())
        except Exception:
            pass
        return 0

    spreads       = [get_spread(r) for r in records]
    total         = len(records)
    low_spread    = [(i, s) for i, s in enumerate(spreads) if s < 25]
    low_pct       = len(low_spread) / total * 100 if total else 0

    # Spread distribution report
    buckets = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 101)]
    status_msg("Score spread distribution:")
    for lo, hi in buckets:
        count = sum(1 for s in spreads if lo <= s < hi)
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2)
        print(f"    {lo:3d}–{hi-1:3d}  {bar}  {count:4d} ({pct:.1f}%)")

    MAX_LOW_SPREAD_RATIO = 0.20
    if low_pct <= MAX_LOW_SPREAD_RATIO * 100:
        status_msg(f"Spread distribution healthy ({low_pct:.1f}% low-spread) — no trimming needed.", Color.GREEN)
        timer.end_stage()
        return

    # Trim the excess low-spread samples randomly
    target_low    = int(total * MAX_LOW_SPREAD_RATIO)
    excess        = len(low_spread) - target_low
    to_remove     = set(i for i, _ in random.sample(low_spread, excess))
    kept          = [rec for i, rec in enumerate(records) if i not in to_remove]

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    status_msg(f"Trimmed {len(to_remove)} low-spread samples → {len(kept)} remaining", Color.YELLOW)
    timer.end_stage()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timer = Timer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",        type=int,   default=800)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--out",            type=str,   default="train_verified_final.jsonl")
    parser.add_argument("--min-overlap",    type=float, default=MIN_BEST_OVERLAP_RATIO, dest="min_overlap")
    parser.add_argument("--max-reuse",      type=int,   default=MAX_RESUME_REUSE,       dest="max_reuse")
    parser.add_argument("--domain",         type=str,   choices=list(DOMAIN_SKILLS.keys()))
    parser.add_argument("--provider",       type=str,   choices=["cerebras"])
    parser.add_argument("--gen-provider",   type=str,   choices=["cerebras", "gemini"], default="cerebras")
    parser.add_argument("--rank-provider",  type=str,   choices=["cerebras", "gemini"], default="cerebras")
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--gen-temperature",  type=float, default=0.7)
    parser.add_argument("--rank-temperature", type=float, default=0.0)
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--skip-gen",       action="store_true")
    parser.add_argument("--force",          action="store_true")
    args = parser.parse_args()

    if args.provider:
        args.gen_provider  = args.provider
        args.rank_provider = args.provider

    gen_temp  = args.temperature if "--temperature" in sys.argv else args.gen_temperature
    rank_temp = args.temperature if "--temperature" in sys.argv else args.rank_temperature

    target   = 20 if args.dry_run else args.samples
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    if args.force:
        status_msg("Force mode: clearing old pairings and output file...", Color.YELLOW)
        if PAIRS_FILE.exists():  PAIRS_FILE.unlink()
        if out_path.exists():    out_path.unlink()

    def get_client(provider):
        return GeminiClient() if provider == "gemini" else CerebrasClient()

    gen_client  = get_client(args.gen_provider)
    rank_client = get_client(args.rank_provider)

    print(f"\n{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}")
    print(f"  {Color.BOLD}{Color.MAGENTA}RESUME RANKING DATASET PIPELINE v2{Color.RESET}")
    print(f"{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}")
    print(f"  {Color.BOLD}GEN PROVIDER       :{Color.RESET} {args.gen_provider.upper()} (temp={gen_temp})")
    print(f"  {Color.BOLD}RANK PROVIDER      :{Color.RESET} {args.rank_provider.upper()} (temp={rank_temp})")
    print(f"  {Color.BOLD}TARGET SAMPLES     :{Color.RESET} {target}")
    print(f"  {Color.BOLD}WORKERS            :{Color.RESET} {args.workers}")
    print(f"  {Color.BOLD}MIN OVERLAP        :{Color.RESET} {args.min_overlap:.0%}")
    print(f"  {Color.BOLD}MAX REUSE          :{Color.RESET} {args.max_reuse}")
    print(f"  {Color.BOLD}OUTPUT             :{Color.RESET} {out_path}")
    if args.skip_gen:
        print(f"  {Color.BOLD}MODE               :{Color.RESET} {Color.YELLOW}skip-gen{Color.RESET}")
    if args.domain:
        print(f"  {Color.BOLD}DOMAIN LOCKED      :{Color.RESET} {args.domain}")
    print(f"{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}\n")

    if args.skip_gen:
        jds     = load_jsonl(JDS_FILE)
        resumes = load_jsonl(RESUMES_FILE)
        status_msg(f"Skip-gen: {len(jds)} JDs and {len(resumes)} resumes loaded", Color.YELLOW)
    else:
        jds     = run_stage1(gen_client, target, args.domain, args.workers, timer, temperature=gen_temp)
        resumes = run_stage2(gen_client, jds, target * 3, args.workers, timer, temperature=gen_temp)

    pairs = run_stage3(jds, resumes, target, timer,
                       min_overlap=args.min_overlap, max_reuse=args.max_reuse)
    run_stage4(rank_client, pairs, out_path, args.workers, timer, temperature=rank_temp)
    run_stage5_balance(out_path, timer)

    print(stats.summary(timer))
    final = sum(1 for _ in open(out_path, encoding="utf-8")) if out_path.exists() else 0
    status_msg(f"Final samples in {out_path}: {final}\n", Color.GREEN, bold=True)


if __name__ == "__main__":
    main()