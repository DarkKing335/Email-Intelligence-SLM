import csv
import json
import uuid
from pathlib import Path

csv_path = Path("e:/FPT/Email Intelligence SLM/data/samples/emails_sample.csv")
jsonl_path = Path("e:/FPT/Email Intelligence SLM/data/samples/emails_sample.jsonl")

jsonl_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, "r", encoding="utf-8") as f_csv:
    reader = csv.DictReader(f_csv)
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for row in reader:
            record = {
                "email_id": str(uuid.uuid4()),
                "thread_id": str(uuid.uuid4()),
                "sender": row["sender"],
                "subject": row["subject"],
                "body": row["body"],
                "category": row["category"]
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Converted CSV to JSONL at {jsonl_path}")
