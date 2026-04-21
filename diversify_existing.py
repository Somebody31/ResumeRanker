import json
import re
import random
from pathlib import Path
# from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
ROOT_DIR = Path(".")

# Original list from pipeline.py to find and replace in resumes
REAL_COMPANIES = [
    "Google", "Stripe", "Airbnb", "Notion", "Shopify", "Figma", "Vercel",
    "Cloudflare", "Datadog", "Snowflake", "Databricks", "HashiCorp", "Confluent",
    "MongoDB", "Elastic", "Twilio", "PagerDuty", "Grafana Labs", "dbt Labs",
    "Hugging Face", "Cohere", "Scale AI", "Weights & Biases", "Linear",
    "Retool", "Segment", "Amplitude", "LaunchDarkly", "Sentry", "Intercom",
]

# Tech/Skill words that should NOT be replaced if they match a company name
# because they are frequently used as skills/technologies.
TECH_SKILLS = {
    "Snowflake", "MongoDB", "Elastic", "Databricks", "Confluent", "HashiCorp",
    "Weights & Biases", "Hugging Face", "Cohere", "Scale AI"
}

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

def generate_company_name():
    return f"{random.choice(PREFIXES)}{random.choice(SUFFIXES)}"

def surgical_replace(text, old_name, replacement_map):
    if not old_name: return text
    
    if old_name not in replacement_map:
        replacement_map[old_name] = generate_company_name()
    new_name = replacement_map[old_name]
    
    if old_name in TECH_SKILLS:
        # 1. Replace in experience headers: "Title\nCompany | Location"
        # We need to handle both real newlines and escaped ones in JSON strings
        pattern = rf'(\n|\\n)({re.escape(old_name)})\s*\|'
        text = re.sub(pattern, rf'\1{new_name} |', text)
        
        # 2. Replace if it's the very first word in the text (often JD starts with company)
        if text.startswith(old_name):
            text = new_name + text[len(old_name):]
            
        # 3. Replace if preceded by "At " or "at " (often used in experience descriptions)
        text = re.sub(rf'\b(at|At)\s+{re.escape(old_name)}\b', f'\\1 {new_name}', text)
        
        # 4. Replace if it's in a standalone "Company: Name" line
        text = re.sub(rf'(?i)Company:\s*{re.escape(old_name)}', f'Company: {new_name}', text)
        
        return text
    else:
        # For non-tech companies like "Stripe", "Figma", "Uber", we can be more aggressive
        # BUT we still avoid "Skills:" lines
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            if re.search(r'^(Skills|Required skills|Required Skills):', line, re.I):
                new_lines.append(line)
            else:
                # Also check for "Skills: " within the line if it's not at the start
                if "Skills:" in line or "Required skills:" in line:
                    # Only replace before the "Skills:" part
                    parts = re.split(r'(?i)(Skills|Required skills):', line, maxsplit=1)
                    if len(parts) > 1:
                        prefix = parts[0].replace(old_name, new_name)
                        new_lines.append(prefix + parts[1] + (parts[2] if len(parts) > 2 else ""))
                    else:
                        new_lines.append(line.replace(old_name, new_name))
                else:
                    new_lines.append(line.replace(old_name, new_name))
        return '\n'.join(new_lines)

def process_item(item):
    # A single record should have consistent naming
    replacement_map = {}
    
    # Process "company" field if it exists (JDs)
    if "company" in item:
        old_name = item["company"]
        if old_name in REAL_COMPANIES:
            new_name = generate_company_name()
            replacement_map[old_name] = new_name
            item["company"] = new_name
            if "text" in item:
                # Special care for tech skills even in JD text
                if old_name in TECH_SKILLS:
                     item["text"] = surgical_replace(item["text"], old_name, replacement_map)
                else:
                     item["text"] = item["text"].replace(old_name, new_name)

    # Process "text" field carefully (Resumes)
    if "text" in item:
        for c in REAL_COMPANIES:
            if c in item["text"]:
                item["text"] = surgical_replace(item["text"], c, replacement_map)

    # Nested structures
    if "jd" in item:
        jd_old = item["jd"].get("company")
        if jd_old and jd_old in REAL_COMPANIES:
            jd_new = generate_company_name()
            item["jd"]["company"] = jd_new
            item["jd"]["text"] = item["jd"]["text"].replace(jd_old, jd_new)
            
    if "candidates" in item:
        for cand in item["candidates"]:
            cand_text = cand.get("text", "")
            cand_map = {} # Each candidate might have different company history
            for c in REAL_COMPANIES:
                if c in cand_text:
                    cand_text = surgical_replace(cand_text, c, cand_map)
            cand["text"] = cand_text

    if "messages" in item:
        for msg in item["messages"]:
            msg_text = msg.get("content", "")
            msg_map = {}
            for c in REAL_COMPANIES:
                if c in msg_text:
                    msg_text = surgical_replace(msg_text, c, msg_map)
            msg["content"] = msg_text
            
    return item

def run_diversification():
    target_files = [
        DATA_DIR / "jds_raw.jsonl",
        DATA_DIR / "resumes_raw.jsonl",
        DATA_DIR / "pairs.jsonl",
        DATA_DIR / "failures.jsonl",
        ROOT_DIR / "train_verified_v3.jsonl",
        ROOT_DIR / "final_verification.jsonl"
    ]
    
    for i, path in enumerate(target_files):
        if not path.exists():
            print(f"Skipping {path}: not found")
            continue
            
        print(f"[{i+1}/{len(target_files)}] Diversifying {path}...")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    records.append(process_item(item))
                except Exception as e:
                    # print(f"Error parsing line in {path}: {e}")
                    records.append({"raw": line}) # Keep original if parse fails

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                if "raw" in r:
                    f.write(r["raw"])
                else:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    random.seed(42)
    run_diversification()
    print("\n[SUCCESS] Dataset diversification complete.")
