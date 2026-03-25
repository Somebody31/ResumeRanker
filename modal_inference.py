import modal

# Use the same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "unsloth",
    )
)

app = modal.App("qwen-inference")
@app.function(
    gpu="A10G", # Inference is lighter, A10G is enough
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("my-gguf-volume")},
)
def run_inference(job_description: str, resumes: list[str]):
    from unsloth import FastLanguageModel
    import torch

    model_name = "unsloth/Qwen3.5-4B"
    max_seq_length = 4096

    print("Loading model and LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, 
    )

    # Load the fine-tuned adapters
    model = FastLanguageModel.for_inference(model)
    model.load_adapter("/mnt/data/qwen35-4b-lora")

    # 3. Format the input using the chat template (matching training)
    SYSTEM_PROMPT = """You are a meticulous technical recruiter specializing in elite engineering talent. Your task is to rank candidates with absolute precision based on a strict rubric. You must identify missing skills by name and apply mathematical deductions without bias. Return ONLY a valid JSON object.


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
  Prohibited words: "implies", "suggests", "assumed", "likely", "probably".

ABSOLUTE RULES (VAL_REJECTION CRITICAL):
  1. Scores strictly descending.
  2. MINIMUM GAP: 10 points between EVERY adjacent rank (e.g., 70, 60, 50).
  3. NO TIED SCORES.
  4. Total spread >= 20 points.

REASON FORMAT:
  "[Audit: <skill>: [P/M], ...]. [Deductions: <list>]. [Math: 100 - <sum> = <val>]. [Cap: MIN(<val>, <cap>) = <val2>]. [Diff: Tie-break adjustment if any]. [Summary: <brief overview>]."

Example:
Candidate A and B both hit 70 cap.
A has more 'Nice to Have' skills.
A Reason: "... [Cap: MIN(85, 70) = 70]. [Diff: None]. [Summary: ...]"
B Reason: "... [Cap: MIN(80, 70) = 70]. [Diff: -10 to break tie with A]. [Summary: ...]"
Scores: {"A": 70, "B": 60}

Return ONLY valid JSON:
{
  "reasons": { ... },
  "scores": {"A": 70, "B": 60, "C": 50},
  "ranking": ["A", "B", "C"]
}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Job Description: {job_description}\n\nResumes:\n" + "\n".join([f"Resume {i+1}:\n{res}" for i, res in enumerate(resumes)])}
    ]
    
    # Explicitly tokenize to get tensors
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_tensors = "pt",
    ).to("cuda")

    print("Generating ranking...")
    outputs = model.generate(
        input_ids = input_ids,
        max_new_tokens = 1024,
        use_cache = True,
    )
    
    # Decode only the NEW tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

@app.local_entrypoint()
def main():
    # SAMPLE DATA FROM TEST CASE 1
    jd = "ApexTechnologies builds high-performance analytics platforms for financial services firms. As a Junior Data Engineer, you will own the ingestion and transformation pipelines that fuel our client-facing dashboards. You will work closely with senior staff to maintain data integrity and optimize storage costs while learning to manage complex analytical workloads. Responsibilities include: developing complex SQL queries to transform raw data into structured reporting tables for business users; utilizing Python to script automation tasks for our data orchestration workflows; implementing Flink for real-time stream processing to ensure low-latency data availability; and managing data storage using Iceberg to optimize query performance across our BigQuery and DuckDB environments. You will assist in debugging pipeline failures and documenting schema changes to support our growing data warehouse. We prioritize engineers who can write clean, maintainable code and communicate technical constraints effectively. Experience: 1-3 years. Nice to have: Trino, ETL/ELT, Fivetran."
    
    resumes = [
        # Candidate A
        "Mid-level data engineer with 4 years of experience designing analytics infrastructure at high-growth tech firms. Skilled in modern data lake technologies and workflow orchestration, with a focus on query performance and data quality.\n\nAirbnb – Data Engineer\n• Implemented Trino with Iceberg for faster SQL-like queries, reducing dashboard load times by 60% across key reporting tables\n• Migrated critical datasets to BigQuery with optimized partitioning, lowering monthly costs by $18k\n\nSnowflake – Data Engineer\n• Developed and maintained Airflow pipelines to sync operational data into Iceberg tables, improving freshness from hourly to near-real-time\n• Integrated Pandas-based data validation checks, reducing data errors by 75% before downstream consumption\n\nSkills: Iceberg, Trino, BigQuery, SQL, Airflow, Pandas, Delta Lake",
        
        # Candidate B
        "Results-oriented Marketing Operations Specialist with 5 years of experience driving lead generation and pipeline growth. Expert at aligning cross-functional teams to optimize funnel performance and increase brand visibility across digital channels.\n\nMarketing Operations Manager | HubSpot\n• Architected automated nurture sequences in HubSpot that increased qualified lead conversion rates by 22% year-over-year.\n• Managed Salesforce data integrity initiatives, reducing duplicate records by 40% and improving sales rep productivity.\n\nDigital Marketing Lead | MongoDB\n• Led SEO strategy overhaul, resulting in a 35% increase in organic traffic over 12 months.\n• Leveraged Google Analytics to identify high-intent user segments, informing $500k in annual ad spend allocation.\n\nSkills: HubSpot, Salesforce, SEO, Google Analytics, Lead Scoring, CRM Management, Email Marketing Automation.",
        
        # Candidate C
        "Mid-level data engineer with 4 years of experience in cloud data platforms, ETL development, and real-time analytics. Focused on scalable data infrastructure and cross-functional collaboration.\n\nData Engineer\nVercel, 2020–Present\n• Engineered BigQuery data models for product analytics, reducing report generation time by 70% across 15TB datasets\n• Developed Apache Spark pipelines to process streaming logs, improving data freshness from hours to under 5 minutes\n• Integrated Kafka with Spark Streaming, achieving 99.95% delivery reliability\n\nJunior Data Engineer\nConfluent, 2018–2020\n• Automated daily data ingestion workflows using Python and Fivetran, cutting manual effort by 15 hours/week\n• Enhanced data pipeline monitoring, reducing incident resolution time by 40%"
    ]
    
    print("Starting inference with FIXED prompt...")
    result = run_inference.remote(jd, resumes)
    print("\n--- Model Output (Case 1) ---\n")
    print(result)
