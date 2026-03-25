import modal

# Define the image with Unsloth and other dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=5.3.0",
        "datasets==4.3.0",
        "peft",
        "trl==0.22.2",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
        "unsloth",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen-finetune-v2")
volume = modal.Volume.from_name("qwen-finetune-storage-v2", create_if_missing=True)

@app.function(
    gpu="A100", # A10G might OOM with packing=True and 4096 context, stepping up to A100 to be safe
    image=image,
    volumes={"/mnt/data": volume},
    timeout=7200, 
)
def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import torch

    # 1. Load Model & Tokenizer
    model_name = "unsloth/Qwen3.5-4B" 
    max_seq_length = 4096
    
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False,        # As defined in the notebook    
        load_in_16bit = True,        # As defined in the notebook 
        use_gradient_checkpointing = "unsloth",
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 3. Load Dataset
    print("Loading dataset...")
    raw_dataset = load_dataset("json", data_files="/mnt/data/train_verified_v2.jsonl", split="train")
    
    # Split into train and validation (90/10)
    dataset = raw_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Format the dataset using the chat template (from the notebook)
    def format_and_tokenize(examples):
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in examples["messages"]
        ]
        return tokenizer(text=texts, truncation=True, max_length=max_seq_length, padding=True)

    print("Formatting datasets...")
    train_dataset = train_dataset.map(
        format_and_tokenize, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        format_and_tokenize, 
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # 4. Define Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 4,   
            gradient_accumulation_steps = 4, 
            per_device_eval_batch_size = 1,
            eval_strategy = "steps",
            eval_steps = 3,
            warmup_steps = 2,
            num_train_epochs = 3,
            learning_rate = 5e-5,
            logging_steps = 1,
            packing = True,          # Enable this for major speedup (from notebook)
            optim = "adamw_8bit",
            weight_decay = 0.01,     
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "/mnt/data/outputs",
            report_to = "none",     
            max_seq_length = max_seq_length,   
            dataset_text_field = "text",
        ),
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    # 6. Save Model
    print("Saving model...")
    model.save_pretrained("/mnt/data/qwen35-4b-lora")
    tokenizer.save_pretrained("/mnt/data/qwen35-4b-lora")
    print("Training complete! Model saved to /mnt/data/qwen35-4b-lora")

@app.local_entrypoint()
def main():
    print("Uploading dataset...")
    try:
        with volume.batch_upload() as batch:
            batch.put_file("train_verified_v2.jsonl", "train_verified_v2.jsonl")    
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Dataset already exists on volume. Skipping upload.")
        else:
            raise e
    train.remote()
