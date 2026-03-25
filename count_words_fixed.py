import json
from pathlib import Path

def calculate_avg_words(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return

    total_words = 0
    sample_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            sample_words = 0
            for msg in data.get("messages", []):
                content = msg.get("content", "")
                sample_words += len(content.split())
            
            total_words += sample_words
            sample_count += 1

    if sample_count == 0:
        print("No samples found.")
        return

    avg_words = total_words / sample_count
    print(f"File: {file_path}")
    print(f"Total Samples: {sample_count}")
    print(f"Total Words: {total_words}")
    print(f"Average Words per Sample: {avg_words:.2f}")

if __name__ == "__main__":
    calculate_avg_words("train_fixed.jsonl")
