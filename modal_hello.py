import modal

# Define the image with necessary dependencies
# Unsloth is great for Qwen fine-tuning on consumer GPUs like T4/A10G
image = modal.Image.debian_slim().pip_install("torch", "nvidia-ml-py3")

app = modal.App("gpu-check")

@app.function(gpu="T4", image=image)
def check_gpu():
    import torch
    import pynvml

    print("Checking GPU status...")
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Total VRAM: {info.total / 1024**3:.2f} GB")
        print(f"Used VRAM: {info.used / 1024**3:.2f} GB")
        print(f"Free VRAM: {info.free / 1024**3:.2f} GB")
    else:
        print("CUDA is NOT available.")

@app.local_entrypoint()
def main():
    check_gpu.remote()
