import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from tqdm import tqdm
import logging
import transformers

transformers.logging.set_verbosity_warning()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class TestDataset(Dataset):
    def __init__(self, path: str, tokenizer: RobertaTokenizer, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                context = item.get("context", "").strip()
                question = item["question"].strip()
                options_dict = item["options"]
                options = list(options_dict.values())
                choice_keys = list(options_dict.keys())

                first_sentences = [f"{context} {question}".strip() if context else question for _ in options]
                second_sentences = options

                encoded = tokenizer(
                    first_sentences,
                    second_sentences,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                answer = item.get("answer", "").strip()
                label = choice_keys.index(answer) if answer in options_dict else None

                self.samples.append({
                    "id": item.get("id"),
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": label,
                    "choice_keys": choice_keys,
                    "gold": answer
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        label = item["label"] if item["label"] is not None else -100
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long)
        }


def predict(
    model_dir: str, 
    test_path: str, 
    output_path: str, 
    batch_size: int = 4
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForMultipleChoice.from_pretrained(model_dir).to(device)
    model.eval()

    dataset = TestDataset(test_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            pred = logits.argmax(dim=1).tolist()
            preds.extend(pred)
            labels.extend(batch["label"].tolist())

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, pred_idx in enumerate(preds):
            item = dataset.samples[i]
            pred_choice = item["choice_keys"][pred_idx]
            result = {
                "pred": pred_choice,
                "gold": item["gold"]
            }
            if item["id"] is not None:
                result["id"] = item["id"]
            else:
                result["index"] = i
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    valid_samples = [(p, l) for p, l in zip(preds, labels) if l != -100]
    if valid_samples:
        correct = sum(p == l for p, l in valid_samples)
        accuracy = correct / len(valid_samples)
        print(f"Valid samples: {len(valid_samples)} | Accuracy: {accuracy:.4f}")
    else:
        print("No labeled samples in test set. Accuracy not calculated.")


def collate_fn(batch: list) -> dict:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    predict(args.model_dir, args.test_path, args.output_path)