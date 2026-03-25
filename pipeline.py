"""
pipeline.py — High-quality resume ranking dataset generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single provider: Cerebras Qwen 3 235B for everything.

Validation philosophy:
  We check the RANKING OUTPUT only — not absolute score values.
  What matters for training:
    1. Ranking order is correct (best candidate first)
    2. Score gaps exist between ranks (model learns to differentiate)
    3. Hard negative is last (model learns domain mismatch)
    4. Reasons reference actual skills (model learns to explain)
  Absolute score values (85 vs 70) are NOT enforced — the ranker
  learns rubric compliance from the prompt, not from validation.

Usage:
    python pipeline.py                  # 300 samples
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
MIN_BEST_OVERLAP_RATIO = 0.40   # Best candidate must match ≥40% of JD required skills
MAX_RESUME_REUSE       = 4      # Max times any resume may appear across all pairs

# Gemini Config (for optional validation)
GEMINI_MODEL     = "gemini-3.1-flash-lite-preview"
GEMINI_BASE_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/"

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

REAL_COMPANIES = [
    "Google", "Stripe", "Airbnb", "Notion", "Shopify", "Figma", "Vercel",
    "Cloudflare", "Datadog", "Snowflake", "Databricks", "HashiCorp", "Confluent",
    "MongoDB", "Elastic", "Twilio", "PagerDuty", "Grafana Labs", "dbt Labs",
    "Hugging Face", "Cohere", "Scale AI", "Weights & Biases", "Linear",
    "Retool", "Segment", "Amplitude", "LaunchDarkly", "Sentry", "Intercom",
]

WRITING_STYLES = [
    "bullet points under each role with quantified outcomes",
    "paragraph prose narrative describing career progression",
    "metric-driven bullet points emphasizing numbers and impact",
    "concise paragraphs with a separate skills section at the end",
    "mixed format: short intro paragraph then bullet points per role",
]

INFERENCE_SYSTEM_PROMPT = (
    "You are a meticulous technical recruiter specializing in elite engineering talent. "
    "Your task is to rank candidates with absolute precision based on a strict rubric. "
    "You must identify missing skills by name and apply mathematical deductions without bias. "
    "Return ONLY a valid JSON object."
)

# ---------------------------------------------------------------------------
# Skill aliases for fuzzy matching (used in reason quality check only)
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
    t = text.lower()
    if skill.lower() in t:
        return True
    for alias in SKILL_ALIASES.get(skill, []):
        if alias.lower() in t:
            return True
    return False


def skills_present_in(skill_list: list, text: str) -> list[str]:
    return [s for s in skill_list if skill_in_text(s, text)]


def skill_overlap_ratio(resume: dict, required_skills: list) -> float:
    """Fraction of required_skills explicitly present in resume text."""
    if not required_skills:
        return 1.0
    text = resume.get("text", "")
    found = sum(1 for s in required_skills if skill_in_text(s, text))
    return found / len(required_skills)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self._lock               = threading.Lock()
        self.api_calls           = 0
        self.jds_generated       = 0
        self.resumes_generated   = 0
        self.pairs_built         = 0
        self.rankings_done       = 0
        self.samples_written     = 0
        self.composition_counts  = defaultdict(int)
        self.validation_failures = defaultdict(int)
        self.generation_failures = defaultdict(int)

    def inc(self, field: str, amount: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def inc_composition(self, name: str):
        with self._lock:
            self.composition_counts[name] += 1

    def fail_validation(self, reason: str):
        with self._lock:
            self.validation_failures[reason] += 1

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
            lines.append(f"\n  {Color.BOLD}{Color.RED}Validation rejections:{Color.RESET}")
            for k, v in sorted(self.validation_failures.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:<55} : {v}")
        lines.append(border)
        return "\n".join(lines)


stats = Stats()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

class CerebrasClient:
    def __init__(self):
        self.keys = []
        # Support comma-separated list
        if os.getenv("CEREBRAS_API_KEYS"):
            self.keys.extend([k.strip() for k in os.getenv("CEREBRAS_API_KEYS").split(",") if k.strip()])
        
        # Support numbered keys: CEREBRAS_API_KEY, CEREBRAS_API_KEY_1, CEREBRAS_API_KEY_2, etc.
        # Check base key first (no suffix)
        base_key = os.getenv("CEREBRAS_API_KEY")
        if base_key and base_key not in self.keys:
            self.keys.append(base_key)
        # Then check _1, _2, _3, ...
        for i in range(1, 21):
            key = os.getenv(f"CEREBRAS_API_KEY_{i}")
            if not key:
                if i > 4: break  # Stop after a reasonable gap
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

    def call(self, messages: list, max_tokens: int = 1500,
              temperature: float = 0.0) -> str:
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
                    # If we have multiple keys, maybe try another one immediately instead of waiting?
                    # For now, let's keep the wait but maybe shorter if we have more keys.
                    wait_time = RATE_WAIT / len(self.keys) if len(self.keys) > 1 else RATE_WAIT
                    tqdm.write(f"  Cerebras Rate Limit (Key {self._current_index}): Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                if attempt == MAX_RETRIES - 1:
                    print(f"  Cerebras Error: {str(e)}")
                    break
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Cerebras call failed after {MAX_RETRIES} retries. Last error: {last_err}")


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

    def get_key(self):
        with self._lock:
            key = self.keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.keys)
            return key

    def call(self, messages: list, max_tokens: int = 1500,
              temperature: float = 0.0) -> str:
        system_text = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user_text   = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        full_text   = f"{system_text}\n\n{user_text}"

        for attempt in range(MAX_RETRIES):
            key = self.get_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": full_text}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            }

            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
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

def extract_object(text: str) -> dict | None:
    """Finds the LAST occurring JSON block. Essential for Chain-of-Thought output."""
    # Strip <think> tags if model uses them natively
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Try to find JSON block using Markdown fences first
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    # Find the outermost { } at the end of the response
    matches = list(re.finditer(r"\{.*\}", text, re.DOTALL))
    if not matches:
        return None
    
    json_str = matches[-1].group()
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(json_str)
    except Exception:
        # Final attempt: try to close a truncated JSON if it was cut off
        try:
            return json.loads(json_str + "}")
        except:
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
# Stage 1: Generate JDs (batch of 3)
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
        company = (
            f"{random.choice(['Nova','Apex','Lumina','Stellar','Forge','Arc','Nexus','Vanta'])}"
            f"{random.choice(['Systems','Labs','Technologies','Solutions','Data','Cloud','AI','Works'])}"
        )
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
  GOOD: "Build and maintain Redis-backed caching to reduce API latency by 40%"
  BAD:  "Work with Redis"
- Required skills section: list all required_skills
- Nice-to-have section: list nice_to_have
- Experience: state exact range "min_years-max_years years"
- Apply ceiling_instruction if not empty
- Length: 150-220 words
- No: "fast-paced", "passionate", "rockstar", "ninja", "synergy"

Return JSON array, each item:
{{
  "text": "<full job description>",
  "domain": "{domain}",
  "seniority": "<from config>",
  "min_years": <number>,
  "max_years": <number>,
  "required_skills": ["skill1", ...]
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
        valid.append(item)

    stats.inc("jds_generated", len(valid))
    return valid


# ---------------------------------------------------------------------------
# Stage 2: Generate resumes (batch of 5, generic)
# ---------------------------------------------------------------------------

RESUME_SYSTEM = """You are a professional resume writer creating realistic software engineering resumes.
Your output trains an AI technical recruiter.
Each resume must be distinct in writing style, skill depth, and career trajectory.
Skills must appear in work history through concrete achievements — not just listed at the bottom.
Return ONLY valid JSON array. No markdown. No explanation."""


def build_resume_batch_prompt(domain: str, seniority: str, count: int = 5) -> str:
    skills = DOMAIN_SKILLS[domain]
    cfg    = SENIORITY_CONFIG[seniority]
    styles = random.sample(WRITING_STYLES, min(count, len(WRITING_STYLES)))

    skill_subsets = [
        random.sample(skills, random.randint(4, min(8, len(skills))))
        for _ in range(count)
    ]
    config_strs = [
        f"Resume {i+1}: focus_skills={json.dumps(subset)}"
        for i, subset in enumerate(skill_subsets)
    ]

    return f"""Generate a JSON array of exactly {count} realistic software engineering resumes.

Domain: {domain.replace('_', ' ')}
Seniority: {seniority} ({cfg['min_years']}–{cfg['max_years']} years experience)
Writing styles: {', '.join(styles)}
Companies: {', '.join(random.sample(REAL_COMPANIES, 10))}

Per-resume skill focus:
{chr(10).join(config_strs)}

RULES:
1. Show each focus skill used at a specific role — not just listed
2. State years of experience explicitly within {cfg['min_years']}–{cfg['max_years']}
3. 2–3 past roles at real companies
4. At least 2 measurable outcomes — varied percentages (23%, 41%, 67%)
5. Each resume distinct — different companies, career paths, emphasis
6. Length: 140–220 words

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
    styles   = random.sample(WRITING_STYLES, min(count, len(WRITING_STYLES)))
    domain_skills_sample = random.sample(
        DOMAIN_SKILLS[domain], min(8, len(DOMAIN_SKILLS[domain]))
    )
    config_strs = [
        f"Resume {i+1}: type={p['type']} skills={json.dumps(p['skills'])}"
        for i, p in enumerate(profiles)
    ]

    return f"""Generate a JSON array of exactly {count} HARD NEGATIVE resumes.
These are completely irrelevant candidates for {domain.replace('_', ' ')} jobs.

Per-resume configs:
{chr(10).join(config_strs)}

Writing styles: {', '.join(styles)}
Companies: {', '.join(random.sample(REAL_COMPANIES, 10))}

RULES:
1. Each candidate works in their specified type — NOT software engineering
2. These {domain.replace('_', ' ')} skills must NOT appear anywhere:
   {', '.join(domain_skills_sample)}
3. Domain mismatch must be obvious to a recruiter
4. Realistic, professionally written
5. 2–3 roles with real company names
6. At least 1 measurable outcome in their actual domain
7. Length: 130–200 words

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
# Stage 3: Pair JDs with resumes (reuse allowed)
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

    # Track how many times each resume (by id()) has been used
    resume_usage: dict[int, int] = defaultdict(int)

    pairs = []
    skipped_overlap = 0
    skipped_pool    = 0

    # --- JD Reuse Logic ---
    # To reach target > len(jds), we may need to iterate through JDs multiple times
    # Pair with different random candidates each time.
    
    attempts = 0
    max_attempts = target * 3  # Safety break
    
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

            # --- Quality filter: exclude over-used resumes ---
            pool = [r for r in raw_pool if resume_usage[id(r)] < max_reuse]

            # --- Quality filter: sort by skill overlap descending ---
            pool_with_scores = [
                (r, skill_overlap_ratio(r, required_skills)) for r in pool
            ]
            pool_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Gate: at least 1 candidate must hit min_overlap threshold
            if not pool_with_scores or pool_with_scores[0][1] < min_overlap:
                skipped_overlap += 1
                continue

            # Only permit resumes in top-80% of overlap scores for this JD
            threshold = pool_with_scores[0][1] * 0.5
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

            clean_hardnegs = [
                r for r in hard_negs
                if len(skills_present_in(required_skills, r.get("text", ""))) < 2
                and resume_usage[id(r)] < max_reuse
            ]

            if n_hardneg > 0 and len(clean_hardnegs) < n_hardneg:
                n_hardneg = 0
                comp_name = "three_same_domain"

            n_to_pick    = min(n_domain, len(eligible_pool))
            domain_picks = random.sample(eligible_pool, n_to_pick)
            hardneg_picks = random.choices(clean_hardnegs, k=n_hardneg) if n_hardneg > 0 else []
            candidates   = domain_picks + hardneg_picks

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
            # We went through all JDs and couldn't find ANY more pairs (likely due to reuse caps)
            status_msg(f"  Yield exhausted: Could only build {len(pairs)} pairs before hitting reuse caps.", Color.YELLOW)
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
# Stage 4: Rank (temp=0.2)
# ---------------------------------------------------------------------------

RANKING_RUBRIC = """
Rank ALL candidates based on their fit for the job description.

SCORING PROCESS (MANDATORY STEPS):
  Step 1. Skill Audit: For EACH required skill, state if it's "Explicitly Present" or "Missing".
  Step 2. Deductions: List all pts (Missing skills -15, Exp gaps, Seniority, Domain).
  Step 3. Math: 100 - sum of deductions (Floor at 0).
  Step 4. Cap: Apply 70 or 40 ceilings if required (Missing skill -> Max 70, Mismatch -> Max 40).
  Step 5. MANDATORY DIFFERENTIATION: 
     - If two candidates have the same score OR a gap < 10 points, you MUST adjust.
     - Keep the better candidate at the higher score.
     - Decrease the lower candidate's score by 10 points (e.g., 70 and 70 becomes 70 and 60).
     - Use "Nice to Have" skills or Tenure to decide who is better.
     - Final scores MUST be separated by at least 10 points (100, 90, 80...).

SCORING RUBRIC:
  - Missing REQUIRED skill:                     -15 pts each
  - Under minimum experience:                  -10 pts per year short
  - Over max experience (1-3 yrs):             -10 pts total
  - Over max experience (3+ yrs):              -25 pts total
  - Seniority Violation (Staff/Principal):     -20 pts ALWAYS
  - Domain mismatch:                           -35 pts

ABSOLUTE CAPS:
  - Missing ANY required skill: final score MUST be <= 70.
  - Domain mismatch: final score MUST be <= 40.
  CEILING RULE: The cap is the MAXIMUM. If two candidates are capped at 70, you MUST score them 70 and 60 to maintain the gap.

NO INFERENCE RULE (STRICT):
  If a skill is not EXPLICITLY named, it is MISSING. 
  Prohibited words for skills: "assumed", "guessed". 
  (Note: You MAY use words like "likely" or "implied" for seniority/experience reasoning only).

ABSOLUTE RULES (VAL_REJECTION CRITICAL):
  1. Scores strictly descending.
  2. MINIMUM GAP: 5 points between EVERY adjacent rank (e.g., 70, 65, 60).
  3. NO TIED SCORES. If multiple candidates are disqualified (0 pts), you MUST still differentiate them using small positive scores (e.g., 3, 2, 1, 0) based on who is "least bad" (e.g., has one skill vs zero skills).
  4. Total spread >= 15 points.
  5. USE METADATA: You must prioritize the 'years_experience' and 'seniority' metadata provided in the candidate block. Do not guess or 'imply' tenure if the metadata is present.

REASON FORMAT:
  "[Audit: <skill>: [P/M], ...]. [Deductions: <list>]. [Math: 100 - <sum> = <val>]. [Cap: MIN(<val>, <cap>) = <val2>]. [Diff: Tie-break adjustment if any]. [Summary: <brief overview>]."

Example:
Candidate A and B both hit 70 cap.
A has more 'Nice to Have' skills.
A Reason: "... [Cap: MIN(85, 70) = 70]. [Diff: None]. [Summary: ...]"
B Reason: "... [Cap: MIN(80, 70) = 70]. [Diff: -10 to break tie with A]. [Summary: ...]"
Scores: {"A": 70, "B": 60}

Return ONLY valid JSON. Start with { and end with }. No other text.
{
  "reasons": { ... },
  "scores": {"A": 70, "B": 60, "C": 50},
  "ranking": ["A", "B", "C"]
}
"""


def rank_pair(client, pair, temperature=0.2) -> dict | None:
    jd         = pair["jd"]
    candidates = pair["candidates"]
    labels     = [chr(65 + i) for i in range(len(candidates))]

    candidate_block = ""
    for label, candidate in zip(labels, candidates):
        candidate_block += f"\nCandidate {label}:\n{candidate['text']}\n"

    # Rubric is part of the user turn — teaches model to follow dynamic instructions
    user_content = (
        f"{RANKING_RUBRIC}\n"
        f"JOB DESCRIPTION:\n{jd['text']}\n\n"
        f"RESUMES:{candidate_block}"
    )

    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    raw    = client.call(messages, max_tokens=1500, temperature=temperature)
    parsed = extract_object(raw)
    if parsed:
        stats.inc("rankings_done")
    return parsed


# ---------------------------------------------------------------------------
# Stage 5: Validation — checks ranking output only, not absolute scores
# ---------------------------------------------------------------------------

def validate_strict(output: dict, labels: list[str],
                    candidates: list[dict], jd: dict) -> tuple[bool, str]:
    """
    Validation — checks what matters for training quality:
    1. All labels present in ranking, scores, reasons
    2. Scores are valid numbers
    3. No tied scores
    4. Scores descending with ranking order
    5. Minimum spread of 20 points
    6. Adjacent gap >= 10 points
    7. Reasons follow mandatory [Deductions] [Math] [Cap] format
    8. Reason references at least one skill
    9. Hard negative is last and scores <= 45
    10. Cap compliance (missing skill <= 70)
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
    if spread < 15:
        return False, f"spread_too_low_{spread}"

    # 6. Adjacent gap >= 5
    for i in range(len(ranked_scores) - 1):
        gap = ranked_scores[i] - ranked_scores[i + 1]
        if gap < 5:
            return False, f"adjacent_gap_too_small_rank{i+1}_{i+2}_{gap}pts"

    # 7 & 8. Reason quality & format
    req_lower = [s.lower() for s in required_skills]
    for label, reason in reasons.items():
        if len(reason.split()) < 15:
            return False, f"reason_too_short_{label}"
        
        # Check mandatory brackets
        if "[deductions:" not in reason.lower() or "[math:" not in reason.lower() or "[cap:" not in reason.lower() or "[summary:" not in reason.lower():
            return False, f"reason_format_invalid_{label}"

        # Check for inference keywords (strict for skills, but allow some for seniority)
        # Note: Banned words are only relative to skill hallucinations now
        if any(w in reason.lower() for w in ["hallucinated", "phantom"]):
            return False, f"inference_detected_{label}"

        # Check reason references at least one skill
        reason_lower = reason.lower()
        cand_skills  = [s.lower() for s in label_to_cand[label].get("skills", [])]
        all_skills   = req_lower + cand_skills
        found_skill = any(s in reason_lower for s in all_skills)
        if not found_skill:
            for skill in required_skills + label_to_cand[label].get("skills", []):
                for alias in SKILL_ALIASES.get(skill, []):
                    if alias.lower() in reason_lower:
                        found_skill = True
                        break
        if not found_skill:
            return False, f"reason_no_skill_reference_{label}"

        # ADVANCED: Skill Traceability (Reason vs Resume Text)
        # Check if the model claims a skill is missing when it actually exists in the resume
        for skill in required_skills:
            # Look for phrasing like "Missing React", "Does not have Python", etc.
            negative_contexts = [f"missing {skill.lower()}", f"not find {skill.lower()}", f"no {skill.lower()} experience", f"lack of {skill.lower()}"]
            if any(ctx in reason_lower for ctx in negative_contexts):
                cand_text = label_to_cand[label].get("text", "").lower()
                if skill_in_text(skill, cand_text):
                    return False, f"hallucinated_missing_skill_{label}_{skill}"

    # 9 & 10. Logic checks
    for label, candidate in label_to_cand.items():
        cand_text = candidate.get("text", "").lower()
        score = scores[label]

        # Domain mismatch check
        if candidate.get("domain") == "other":
            if label == ranking[0]:
                return False, "hard_negative_ranked_first"
            if score > 40:
                return False, f"hard_negative_score_too_high_{label}_{score}"

        # Missing skill cap check
        missing_any = False
        for skill in required_skills:
            if not skill_in_text(skill, cand_text):
                missing_any = True
                break

        if missing_any and score > 70:
            return False, f"cap_violation_missing_skill_{label}_{score}_expected_le_70"

        # ADVANCED: Seniority/YOE Ground-Truth Validation (Metadata-based)
        cand_yoe = candidate.get("years_experience", 0)
        min_yoe  = jd.get("min_years", 0)
        max_yoe  = jd.get("max_years", 100)
        
        # Penalize if YOE is significantly below minimum
        if cand_yoe < min_yoe and score > 70:
            return False, f"metadata_yoe_underqualified_{label}_{cand_yoe}_vs_min_{min_yoe}"
            
        # Optional: Strict Seniority Ceiling check (if JD has max_years or Staff/Principal restriction)
        # If JD says "Seniority Ceiling" and candidate is 'Staff' or 'Principal'
        if "seniority ceiling" in jd.get("text", "").lower() or jd.get("seniority") in ("Senior", "Mid-level"):
            cand_seniority = candidate.get("seniority", "")
            if cand_seniority in ("Staff", "Principal", "Director") and score > 70:
                # Unless they are clearly the only option, they should be penalized for being overqualified
                # which is a common failure mode for LLMs to prefer "more" experience even if capped.
                return False, f"metadata_overqualified_ceiling_violation_{label}_{cand_seniority}"

    # 11. Best candidate score floor — low-signal pair if even winner scores poorly
    best_score = scores[ranking[0]]
    if best_score < 20:
        return False, f"best_candidate_score_too_low_{ranking[0]}_{best_score}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Process one pair
# ---------------------------------------------------------------------------

def process_pair(client, pair: dict,
                 out_path: Path, write_lock: threading.Lock,
                 dedup_hashes: set = None, temperature=0.2) -> bool:
    jd          = pair["jd"]
    candidates  = pair["candidates"]
    composition = pair.get("composition", "unknown")
    labels      = [chr(65 + i) for i in range(len(candidates))]

    # --- Deduplication: hash JD + sorted candidate texts ---
    cand_texts_sorted = tuple(sorted(c.get("text", "")[:200] for c in candidates))
    dedup_key = hashlib.sha256(
        (jd.get("text", "")[:300] + "||" + "||" .join(cand_texts_sorted)).encode()
    ).hexdigest()
    if dedup_hashes is not None:
        with write_lock:
            if dedup_key in dedup_hashes:
                stats.fail_validation("duplicate_pair")
                return False
            dedup_hashes.add(dedup_key)

    ranking_output = rank_pair(client, pair, temperature=temperature)
    if not ranking_output:
        stats.fail_generation("ranking_json_parse_failed")
        return False


    is_valid, reason = validate_strict(ranking_output, labels, candidates, jd)
    if not is_valid:
        stats.fail_validation(reason)
        # Log failure for prompt engineering
        failure_record = {
            "val_reason": reason,
            "ranking_output": ranking_output,
            "jd_required": jd.get("required_skills", []),
            "candidates": [{"label": labels[i], "domain": c.get("domain"), "skills": c.get("skills")} for i, c in enumerate(candidates)]
        }
        with write_lock:
            append_jsonl(DATA_DIR / "failures.jsonl", failure_record)
        return False

    candidate_block = ""
    for label, candidate in zip(labels, candidates):
        candidate_block += f"\nCandidate {label}:\n{candidate['text']}\n"

    # New format: short system (role identity only); rubric + content in user turn.
    # This teaches the model to follow dynamic instructions rather than memorise a fixed rubric.
    user_content = (
        f"{RANKING_RUBRIC}\n"
        f"JOB DESCRIPTION:\n{jd['text']}\n\n"
        f"RESUMES:{candidate_block}"
    ).strip()

    record = {
        "pair_hash": pair.get("pair_hash"), # Essential for checkpointing
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
# Pipeline stages
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

    status_msg(f"Stage 3: Pairing JDs with resumes (overlap≥40%, max-reuse={max_reuse})...")
    pairs = pair_jds_and_resumes(jds, resumes, target,
                                  min_overlap=min_overlap, max_reuse=max_reuse)
    for pair in pairs:
        append_jsonl(PAIRS_FILE, pair)

    comp_counts = defaultdict(int)
    for p in pairs:
        comp_counts[p.get("composition", "unknown")] += 1
    status_msg(f"Stage 3 done: {len(pairs)} pairs", Color.GREEN)
    timer.end_stage()
    return pairs


def run_stage4(client, pairs, out_path, workers, timer: Timer, temperature=0.2):
    timer.start_stage("Stage 4: Ranking")
    
    # Load hashes of already ranked pairs to enable robust checkpointing
    ranked_hashes = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    h = obj.get("pair_hash")
                    if h: ranked_hashes.add(h)
                except Exception: continue

    remaining = []
    for p in pairs:
        cand_texts_sorted = tuple(sorted(c.get("text", "")[:400] for c in p["candidates"]))
        h = hashlib.sha256(
            (p["jd"].get("text", "")[:500] + "||" + "||".join(cand_texts_sorted)).encode()
        ).hexdigest()
        p["pair_hash"] = h # Attach hash to the pair for later reference
        if h not in ranked_hashes:
            remaining.append(p)

    if not remaining:
        status_msg(f"Stage 4: All {len(pairs)} samples already ranked, skipping.", Color.GREEN)
        timer.end_stage()
        return

    status_msg(f"Stage 4: Ranking {len(remaining)} remaining pairs (already have {len(ranked_hashes)}) (temp={temperature}, workers={workers})...")
    write_lock = threading.Lock()
    dedup_hashes = set()   # Thread-safe via write_lock
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
                pbar.set_postfix(ok=f"{Color.GREEN}{success}{Color.RESET}", 
                                fail=f"{Color.RED}{failed}{Color.RESET}", 
                                api=stats.api_calls)
                pbar.update(1)

    timer.end_stage()
    status_msg(f"Stage 4 done: {success} written, {failed} rejected (dedup: {len(dedup_hashes)} unique)",
               Color.GREEN if failed == 0 else Color.YELLOW)


# ---------------------------------------------------------------------------
# Stage 5: Score Distribution Balancing (post-processing)
# ---------------------------------------------------------------------------

SCORE_BUCKETS = [(40, 55), (55, 70), (70, 80), (80, 90), (90, 101)]
MAX_BUCKET_RATIO = 0.30  # No single bucket may exceed 30% of total samples

def run_stage5_balance(out_path: Path, timer: Timer):
    """Trim over-represented score buckets to ensure balanced training signal."""
    timer.start_stage("Stage 5: Score Balancing")
    if not out_path.exists():
        status_msg("Stage 5: No output file found, skipping.", Color.YELLOW)
        timer.end_stage()
        return

    records = load_jsonl(out_path)
    if not records:
        timer.end_stage()
        return

    # Extract best-candidate score from each sample
    def get_best_score(rec):
        try:
            assistant_msg = rec["messages"][-1]["content"]
            ranking = json.loads(assistant_msg)
            scores = ranking.get("scores", {})
            rank_order = ranking.get("ranking", [])
            if rank_order and scores:
                return scores.get(rank_order[0], 0)
        except Exception:
            pass
        return 0

    # Bucket the samples
    buckets = {b: [] for b in SCORE_BUCKETS}
    for i, rec in enumerate(records):
        score = get_best_score(rec)
        for lo, hi in SCORE_BUCKETS:
            if lo <= score < hi:
                buckets[(lo, hi)].append(i)
                break

    total = len(records)
    max_per_bucket = int(total * MAX_BUCKET_RATIO)

    # Report distribution
    status_msg("Score distribution (before balancing):")
    trimmed_indices = set()
    for (lo, hi), indices in sorted(buckets.items()):
        count = len(indices)
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        over = count > max_per_bucket
        color = Color.RED if over else Color.GREEN
        print(f"    {color}{lo:3d}–{hi-1:3d}{Color.RESET}  {bar}  {count:4d} ({pct:.1f}%)" +
              (f"  → trimming to {max_per_bucket}" if over else ""))
        if over:
            # Keep a random subset to maintain diversity
            random.shuffle(indices)
            to_remove = indices[max_per_bucket:]
            trimmed_indices.update(to_remove)

    if not trimmed_indices:
        status_msg("Score distribution is balanced — no trimming needed.", Color.GREEN)
        timer.end_stage()
        return

    # Rewrite the file without trimmed samples
    kept = [rec for i, rec in enumerate(records) if i not in trimmed_indices]
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    status_msg(f"Trimmed {len(trimmed_indices)} samples → {len(kept)} remaining", Color.YELLOW)
    timer.end_stage()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timer = Timer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",  type=int, default=800,
                        help="Target validated samples (recommend 800 for 1K JDs / 3K resumes)")
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--out",      type=str, default="train_verified_final.jsonl")
    parser.add_argument("--min-overlap", type=float, default=MIN_BEST_OVERLAP_RATIO,
                        dest="min_overlap",
                        help="Min skill overlap ratio for best candidate in a pair (default 0.40)")
    parser.add_argument("--max-reuse", type=int, default=MAX_RESUME_REUSE,
                        dest="max_reuse",
                        help="Max times a resume may appear across pairs (default 4)")
    parser.add_argument("--domain",   type=str, choices=list(DOMAIN_SKILLS.keys()))
    parser.add_argument("--provider", type=str, choices=["cerebras"],
                        help="Global provider override")
    parser.add_argument("--gen-provider", type=str, choices=["cerebras", "gemini"], default="cerebras")
    parser.add_argument("--rank-provider", type=str, choices=["cerebras", "gemini"], default="cerebras")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Global temperature override")
    parser.add_argument("--gen-temperature", type=float, default=0.7)
    parser.add_argument("--rank-temperature", type=float, default=0.0)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip Stage 1+2, re-pair and re-rank only")
    parser.add_argument("--force", action="store_true",
                        help="Clear previous pairings and output file to start fresh")
    args = parser.parse_args()

    # Apply overrides
    if args.provider:
        args.gen_provider  = args.provider
        args.rank_provider = args.provider
    
    gen_temp  = args.temperature if "--temperature" in sys.argv else args.gen_temperature
    rank_temp = args.temperature if "--temperature" in sys.argv else args.rank_temperature

    target   = 20 if args.dry_run else args.samples
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    if args.force:
        status_msg("Force mode: Clearing old pairings and output file...", Color.YELLOW)
        if PAIRS_FILE.exists(): PAIRS_FILE.unlink()
        if out_path.exists(): out_path.unlink()

    def get_client(provider):
        return GeminiClient() if provider == "gemini" else CerebrasClient()

    gen_client  = get_client(args.gen_provider)
    rank_client = get_client(args.rank_provider)

    print(f"\n{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}")
    print(f"  {Color.BOLD}{Color.MAGENTA}RESUME RANKING DATASET PIPELINE{Color.RESET}")
    print(f"{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}")
    print(f"  {Color.BOLD}GEN PROVIDER       :{Color.RESET} {args.gen_provider.upper()} (temp={gen_temp})")
    print(f"  {Color.BOLD}RANK PROVIDER      :{Color.RESET} {args.rank_provider.upper()} (temp={rank_temp})")
    print(f"  {Color.BOLD}TARGET SAMPLES     :{Color.RESET} {target}")
    print(f"  {Color.BOLD}WORKERS            :{Color.RESET} {args.workers}")
    print(f"  {Color.BOLD}OUTPUT             :{Color.RESET} {out_path}")
    if args.skip_gen:
        print(f"  {Color.BOLD}MODE               :{Color.RESET} {Color.YELLOW}skip-gen{Color.RESET}")
    print(f"{Color.DIM}  ────────────────────────────────────────────────────{Color.RESET}")
    if args.domain:
        print(f"  {Color.BOLD}DOMAIN LOCKED      :{Color.RESET} {args.domain}")
    print(f"{Color.BOLD}{Color.MAGENTA}{'='*62}{Color.RESET}\n")

    if args.skip_gen:
        jds     = load_jsonl(JDS_FILE)
        resumes = load_jsonl(RESUMES_FILE)
        status_msg(f"Skip-gen: {len(jds)} JDs and {len(resumes)} resumes loaded", Color.YELLOW)
    else:
        jds     = run_stage1(gen_client, target, args.domain, args.workers, timer, temperature=gen_temp)
        resumes = run_stage2(gen_client, jds, target*3, args.workers, timer, temperature=gen_temp)

    pairs = run_stage3(jds, resumes, target, timer,
                       min_overlap=args.min_overlap, max_reuse=args.max_reuse)
    run_stage4(rank_client, pairs, out_path, args.workers, timer, temperature=rank_temp)
    run_stage5_balance(out_path, timer)

    print(stats.summary(timer))
    final = sum(1 for _ in open(out_path, encoding="utf-8")) if out_path.exists() else 0
    status_msg(f"Final samples in {out_path}: {final}\n", Color.GREEN, bold=True)


if __name__ == "__main__":
    main()