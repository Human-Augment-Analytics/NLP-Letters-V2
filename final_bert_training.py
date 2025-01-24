import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from degender_pronoun import degenderizer

model_checkpoint = "keshavkumaresan/distilbert-base-uncased-debiased"
data_column = "s1_s2"
preprocessing_type = "all"
batch_size = 16
metric_name = "f1"
labels = ["female", "male"]
num_labels = 2
dataset_path = "data/sentence_sets_trimmed.csv"

degender_pronouns = {
    " mr ": " mx ",
    " mrs ": " mx ",
    " ms ": " mx ",
    " miss ": " mx ",
    " mister ": " mx ",
}

degender_nouns = {
    " man ": " person ",
    " men ": " persons ",
    " woman ": " person ",
    " women ": " persons ",
    " man's ": " person's ",
    " men's ": " person's ",
    " woman's ": " person's ",
    " women's ": " person's ",
    " gentleman ": " person ",
    " lady ": " person ",
    " gentleman's ": " person's ",
    " lady's ": " person's ",
}


def preprocess(df, data_col, p_type="none"):
    if p_type == "none":
        return df
    D = degenderizer()
    df[data_col] = df[data_col].apply(lambda x: D.degender(x) if len(x) > 5 else x)
    for old, new in degender_pronouns.items():
        df[data_col] = df[data_col].str.lower().replace(old, new)
    if p_type == "all":
        for old, new in degender_nouns.items():
            df[data_col] = df[data_col].str.lower().replace(old, new)
    return df


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([8.0, 1.0], device=model.device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


metric = evaluate.load(metric_name)
confusion_metric = evaluate.load("confusion_matrix")


def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    cm = confusion_metric.compute(predictions=preds, references=true_labels)
    print("Confusion Matrix:", cm)
    print(classification_report(true_labels, preds, target_names=labels))
    print("MCC:", matthews_corrcoef(true_labels, preds))
    print("Balanced Accuracy:", balanced_accuracy_score(true_labels, preds))
    return metric.compute(predictions=preds, references=true_labels, average="macro")


def preprocess_function(sample):
    return tokenizer(sample[data_column], truncation=True, padding=True)


df = pd.read_csv(dataset_path, encoding="unicode_escape")
df = preprocess(df, data_column, preprocessing_type)

dataset = (
    Dataset.from_pandas(df)
    .rename_column("applicant_gender", "label")
    .class_encode_column("label")
)
dataset = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)

encoded_dataset = dataset.map(preprocess_function, batched=True)
task = f"nlp-letters-{data_column}-{preprocessing_type}-class-weighted"
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
