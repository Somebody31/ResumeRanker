import modal

# Image with Unsloth and llama.cpp dependencies
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
    .apt_install("git", "make", "cmake", "g++") # Required for llama.cpp compilation
)

app = modal.App("qwen-export-gguf")
volume = modal.Volume.from_name("qwen-finetune-storage-v2")

@app.function(
    gpu="A10G", 
    image=image,
    volumes={"/mnt/data": volume},
    timeout=3600,
)
def export_gguf():
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

    # Attach adapters
    model.load_adapter("/mnt/data/qwen35-4b-lora")

    print("Exporting to GGUF (Q4_K_M)...")
    # This will:
    # 1. Merge adapters with base model
    # 2. Automatically download/compile llama.cpp if needed
    # 3. Quantize to Q4_K_M
    # 4. Save to the volume
    model.save_pretrained_gguf(
        "/mnt/data/qwen35-4b-gguf", 
        tokenizer, 
        quantization_method = "q4_k_m"
    )
    
    print("Export complete! GGUF file is in /mnt/data/qwen35-4b-gguf on the volume.")

@app.local_entrypoint()
def main():
    print("Starting GGUF export on Modal...")
    export_gguf.remote()
    print("\nNext step: Download the GGUF file using:")
    print("modal volume get qwen-finetune-storage-v2 qwen35-4b-gguf .")
