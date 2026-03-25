"""
verify_and_correct.py — Verification and Correction tool for ranking dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses Cerebras Qwen 3 235B to audit and fix existing samples in train.jsonl.
"""

import os
import re
import json
import time
import random
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Terminal Aesthetics (Copy from pipeline.py)
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
        self._stage_start = None
        self.current_stage = None

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
# Config & Clients (Copy from pipeline.py)
# ---------------------------------------------------------------------------

MODEL       = "qwen-3-235b-a22b-instruct-2507"
BASE_URL    = "https://api.cerebras.ai/v1"
MAX_RETRIES = 5
RATE_WAIT   = 10

class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_audited = 0
        self.perfect = 0
        self.corrected = 0
        self.discarded = 0
        self.corrections_failed = 0
        self.flaws_detected = defaultdict(int)

    def inc(self, field: str, amount: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def log_flaw(self, flaw: str):
        with self._lock:
            self.flaws_detected[flaw] += 1

    def summary(self, timer: Timer) -> str:
        border = f"{Color.DIM}{'=' * 62}{Color.RESET}"
        lines = [
            "\n" + border,
            f"  {Color.BOLD}{Color.CYAN}VERIFICATION & CORRECTION SUMMARY{Color.RESET}",
            border,
            f"  Total audited          : {Color.BOLD}{self.total_audited}{Color.RESET}",
            f"  Perfect (No correction): {Color.GREEN}{self.perfect}{Color.RESET}",
            f"  Corrected              : {Color.GREEN}{self.corrected}{Color.RESET}",
            f"  Discarded (Failed)     : {Color.RED}{self.discarded}{Color.RESET}",
            f"  Total duration         : {Color.YELLOW}{timer.format_duration(timer.total())}{Color.RESET}",
        ]
        if self.flaws_detected:
            lines.append(f"\n  {Color.BOLD}Flaws Detected & Fixed:{Color.RESET}")
            for k, v in sorted(self.flaws_detected.items(), key=lambda x: -x[1]):
                lines.append(f"    {k:<45} : {v}")
        lines.append(border)
        return "\n".join(lines)

stats = Stats()

class CerebrasClient:
    def __init__(self):
        self.keys = []
        if os.getenv("CEREBRAS_API_KEYS"):
            self.keys.extend([k.strip() for k in os.getenv("CEREBRAS_API_KEYS").split(",") if k.strip()])
        # Support numbered keys
        i = 1
        while True:
            suffix = f"_{i}" if i > 1 else ""
            key = os.getenv(f"CEREBRAS_API_KEY{suffix}")
            if not key:
                if i > 1: break
                else: i += 1; continue
            if key not in self.keys: self.keys.append(key)
            i += 1
            if i > 10: break
        
        if not self.keys: raise ValueError("No Cerebras keys found.")
        print(f"  CerebrasClient initialized with {len(self.keys)} keys.")
        self.clients = [OpenAI(base_url=BASE_URL, api_key=k) for k in self.keys]
        self._current = 0
        self._lock = threading.Lock()

    def call(self, messages, max_tokens=2500, temperature=0.0):
        for attempt in range(MAX_RETRIES):
            with self._lock:
                client = self.clients[self._current]
                self._current = (self._current + 1) % len(self.clients)
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature
                )
                content = resp.choices[0].message.content
                if not content: raise ValueError("Empty response")
                return content.strip()
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate limit" in err:
                    wait = RATE_WAIT / len(self.keys)
                    time.sleep(wait)
                    continue
                time.sleep(2**attempt)
        return None

def extract_object(text: str) -> dict | None:
    """Finds the LAST occurring JSON block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try: return json.loads(json_match.group(1))
        except: pass
    matches = list(re.finditer(r"\{.*\}", text, re.DOTALL))
    if not matches: return None
    json_str = matches[-1].group()
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try: return json.loads(json_str)
    except:
        try: return json.loads(json_str + "}")
        except: return None

# ---------------------------------------------------------------------------
# Skill aliases (Keep in sync with pipeline.py)
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
}

def skill_in_text(skill: str, text: str) -> bool:
    t = text.lower()
    if skill.lower() in t: return True
    for alias in SKILL_ALIASES.get(skill, []):
        if alias.lower() in t: return True
    return False

# ---------------------------------------------------------------------------
# Audit Logic
# ---------------------------------------------------------------------------

def audit_sample(sample: dict) -> tuple[bool, str | None]:
    """Checks for factual accuracy. Returns (is_perfect, flaw_description)."""
    try:
        user_msg = sample["messages"][1]["content"]
        assistant_json = json.loads(sample["messages"][2]["content"])
        
        # Extract JD text and resumes
        jd_match = re.search(r"JOB DESCRIPTION:\n(.*?)\n\nRESUMES:", user_msg, re.DOTALL)
        if not jd_match: return False, "parse_error_jd"
        jd_text = jd_match.group(1)
        
        # Skill check (Factual audit)
        # Look for "Skills: ..." or "Required Skills: ..." in JD
        skills_match = re.search(r"(Skills|Required Skills|Technical Skills):\s*(.*?)(\n|$|Responsibilities|Role)", jd_text, re.IGNORECASE)
        required_skills = []
        if skills_match:
            skills_part = skills_match.group(2)
            required_skills = [s.strip() for s in skills_part.split(",")]
        
        # Check reasons vs reality
        reasons = assistant_json.get("reasons", {})
        scores  = assistant_json.get("scores", {})
        ranking = assistant_json.get("ranking", [])
        
        resumes_block = user_msg.split("RESUMES:\n")[1]
        # Split resumes. Labels are Candidate A:, Candidate B:, etc.
        parts = re.split(r"\nCandidate [A-D]:\n", resumes_block)
        candidates = [p.strip() for p in parts if p.strip()]
        labels = [chr(65+i) for i in range(len(candidates))]
        cand_map = dict(zip(labels, candidates))
        
        # 1. Audit skill mentions in reasons
        for label, reason in reasons.items():
            cand_text = cand_map.get(label, "").lower()
            for skill in required_skills:
                present = skill_in_text(skill, cand_text)
                # Hallucination: Claims skill is missing when it exists
                if f"missing {skill.lower()}" in reason.lower() and present:
                    return False, f"hallucinated_missing_skill_{label}_{skill}"
                # Hallucination: Claims skill is present when it doesn't exist
                if f"mention of {skill.lower()}" in reason.lower() and not present and "not" not in reason.lower():
                    return False, f"hallucinated_present_skill_{label}_{skill}"

        # 2. Audit ranking logic (Basic: if A has more skills than B, A should be >= B score)
        skill_counts = {}
        for label, text in cand_map.items():
            count = sum(1 for s in required_skills if skill_in_text(s, text))
            skill_counts[label] = count
            
        # Check adjacent pairs in ranking
        for i in range(len(ranking) - 1):
            curr_label = ranking[i]
            next_label = ranking[i+1]
            if skill_counts[curr_label] < skill_counts[next_label]:
                # This doesn't ALWAYS mean it's wrong (e.g., seniority), but it's a strong signal
                pass

        # 3. Structural integrity checks
        vals = list(scores.values())

        # 3a. No tied scores
        if len(set(vals)) != len(vals):
            return False, "duplicate_scores"

        # 3b. Scores must be strictly descending with ranking
        ranked_scores = [scores[c] for c in ranking if c in scores]
        if ranked_scores != sorted(ranked_scores, reverse=True):
            return False, "scores_not_descending"

        # 3c. Minimum spread >= 20
        if len(vals) >= 2 and (max(vals) - min(vals)) < 20:
            return False, f"spread_too_low_{max(vals) - min(vals)}"

        # 3d. Adjacent gap >= 5 (lenient; pipeline uses 10)
        for i in range(len(ranked_scores) - 1):
            gap = ranked_scores[i] - ranked_scores[i + 1]
            if gap < 5:
                return False, f"adjacent_gap_too_small_rank{i+1}_{i+2}_{gap}pts"

        # 4. Audit Math & Caps
        for label, reason in reasons.items():
            # Math: 100 - 30 = 70
            math_match = re.search(r"\[Math: 100 - (\d+) = (\d+)\]", reason)
            if math_match:
                deductions = int(math_match.group(1))
                result = int(math_match.group(2))
                expected_result = max(0, 100 - deductions)
                if expected_result != result:
                    return False, f"math_error_{label}_100-{deductions}_is_not_{result}"

            # Cap: MIN(70, 70) = 70
            cap_match = re.search(r"\[Cap: MIN\((\d+), (\d+)\) = (\d+)\]", reason)
            final_reason_score = None
            if cap_match:
                val = int(cap_match.group(1))
                cap = int(cap_match.group(2))
                res = int(cap_match.group(3))
                if min(val, cap) != res:
                    return False, f"cap_error_{label}_min({val},{cap})_is_not_{res}"
                final_reason_score = res
            elif math_match:
                final_reason_score = int(math_match.group(2))

            # Score consistency: reason result vs JSON score
            if final_reason_score is not None:
                json_score = scores.get(label)
                if json_score != final_reason_score:
                    has_diff = re.search(r"\[(Diff|Adjustment):.*?\]", reason, re.IGNORECASE)
                    if not has_diff or ("tie" not in reason.lower() and "adjustment" not in reason.lower()):
                        return False, f"score_mismatch_{label}_reason:{final_reason_score}_json:{json_score}"

        return True, None
    except Exception as e:
        return False, f"audit_crash_{str(e)}"

# ---------------------------------------------------------------------------
# Correction Logic
# ---------------------------------------------------------------------------

CORRECTION_SYSTEM = """You are an elite technical recruiter performing a quality audit.
The provided JSON output has errors in reasoning or adherence to the rubric.
Identify the errors mentioned in the user prompt and provide a CORRECTED version.
Follow the rubric EXACTLY. No inferred skills. Precise deductions.

ABSOLUTE RULES YOU MUST FOLLOW:
  1. NO TIED SCORES. Every candidate MUST have a unique score.
  2. Scores MUST be strictly descending in ranking order.
  3. MINIMUM GAP of 10 points between every adjacent rank.
  4. Total spread >= 20 points.
  5. MATH MUST BE EXACT:
     - [Math: 100 - <sum_of_deductions> = <result>]. The result MUST equal max(0, 100 - sum).
     - [Cap: MIN(<math_result>, <cap_value>) = <final>]. The final MUST equal min(math_result, cap_value).
     - VERIFY YOUR ARITHMETIC BEFORE RETURNING. Double-check every subtraction.
  6. SCORE-REASON CONSISTENCY:
     - The "scores" JSON value for each candidate MUST EXACTLY EQUAL the final value
       from that candidate's [Cap] block (or [Math] block if no cap applies).
     - The ONLY exception is a documented [Diff: -N to break tie...] adjustment.
     - If you apply a [Diff] adjustment, the JSON score = Cap result + Diff amount.
  7. Do NOT change the JD, resumes, or candidate labels. Only fix the ranking output.

Return ONLY valid JSON. Start with { and end with }. No other text."""

MAX_CORRECTION_ATTEMPTS = 2

def correct_sample(client: CerebrasClient, sample: dict, flaw: str) -> dict | None:
    """Attempt correction up to MAX_CORRECTION_ATTEMPTS times, re-auditing each attempt."""
    user_content = sample["messages"][1]["content"]
    original_json = sample["messages"][2]["content"]

    for attempt in range(MAX_CORRECTION_ATTEMPTS):
        prompt = f"""AUDIT FINDING: {flaw}

ORIGINAL OUTPUT:
{original_json}

Please provide an updated JSON that corrects the above audit finding.
REMINDER: All scores must be UNIQUE, strictly descending, with >= 10 point gaps.
The [Math] and [Cap] values in reasons MUST match the final JSON scores."""

        messages = [
            {"role": "system", "content": CORRECTION_SYSTEM},
            {"role": "user",   "content": user_content},
            {"role": "user",   "content": prompt}
        ]

        raw = client.call(messages, max_tokens=3000, temperature=0.0)
        if not raw: return None

        corrected_json = extract_object(raw)
        if not corrected_json: continue

        # Build candidate sample and re-audit it
        new_sample = sample.copy()
        new_sample["messages"] = list(sample["messages"])
        new_sample["messages"][2] = {"role": "assistant", "content": json.dumps(corrected_json)}

        is_ok, new_flaw = audit_sample(new_sample)
        if is_ok:
            return new_sample

        # If still flawed, retry with the NEW flaw as context
        flaw = new_flaw
        original_json = json.dumps(corrected_json)

    return None  # All attempts exhausted

# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="train.jsonl")
    parser.add_argument("--output", type=str, default="train_audited.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    timer = Timer()
    client = CerebrasClient()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists():
        status_msg(f"Input file {in_path} not found.", Color.RED)
        return

    with open(in_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    if args.limit:
        samples = samples[:args.limit]

    status_msg(f"Auditing {len(samples)} samples from {in_path}...")
    
    final_samples = []
    write_lock = threading.Lock()

    def process(sample):
        stats.inc("total_audited")
        is_perfect, flaw = audit_sample(sample)
        
        if is_perfect:
            stats.inc("perfect")
            return sample
        
        stats.log_flaw(flaw)
        corrected = correct_sample(client, sample, flaw)
        if corrected:
            stats.inc("corrected")
            return corrected
        else:
            stats.inc("discarded")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, s) for s in samples]
        with tqdm(total=len(samples), desc=f"  {Color.CYAN}Auditing{Color.RESET}", unit="sample") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with write_lock:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result) + "\n")
                pbar.update(1)

    print(stats.summary(timer))
    status_msg(f"Audited samples written to {out_path}", Color.GREEN)

if __name__ == "__main__":
    main()
