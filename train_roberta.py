import transformers
import logging
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForMultipleChoice, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

transformers.logging.set_verbosity_warning()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class MultipleChoiceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                context = item.get("context", "").strip()
                question = item["question"].strip()
                options = list(item["options"].values())
                option_keys = list(item["options"].keys())

                answer_key = item.get("answer")
                label = option_keys.index(answer_key) if answer_key in option_keys else None
                if label is None:
                    continue

                first_sentences = [f"{context} {question}".strip() if context else question for _ in options]
                second_sentences = options

                encoded = self.tokenizer(
                    first_sentences,
                    second_sentences,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                self.data.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": label
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': torch.tensor(item["label"], dtype=torch.long)
        }


def train_roberta_model(train_path, val_path, output_dir, model_name='roberta-base'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMultipleChoice.from_pretrained(model_name)

    train_dataset = MultipleChoiceDataset(train_path, tokenizer)
    val_dataset = MultipleChoiceDataset(val_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nBest Validation Accuracy: {metrics['eval_accuracy']:.4f}")

    model.save_pretrained(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "validation.jsonl")

    print(f"Training on dataset: {args.dataset}")
    print(f"Train file: {train_path}")
    print(f"Validation file: {val_path}")
    print(f"Using model: {args.model_dir}")
    print(f"Saving to: {args.output_dir}")

    train_roberta_model(
        train_path=train_path,
        val_path=val_path,
        output_dir=args.output_dir,
        model_name=args.model_dir
    )