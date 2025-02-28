import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
)

import torch
from torch import nn

import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import Dataset
from degender_pronoun import degenderizer

# Dictionaries for degendering
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
    " man's ": " person's",
    " men's ": " person's",
    " woman's ": " person's",
    " women's ": " person's",
    " gentleman ": " person ",
    " lady ": " person ",
    " gentleman's ": " person's ",
    " lady's ": " person's ",
}


def preprocess(df, data_column, preprocess_type):
    if preprocess_type == "none":
        return df
    D = degenderizer()
    df[data_column] = df[data_column].apply(
        lambda x: D.degender(x) if len(x) > 5 else x
    )
    for old, new in degender_pronouns.items():
        df[data_column] = df[data_column].str.lower().replace(old, new)
    if preprocess_type == "all":
        for old, new in degender_nouns.items():
            df[data_column] = df[data_column].str.lower().replace(old, new)
    return df


def train_distilbert(dataset, data_column):
    metric = evaluate.load("f1")
    confusion_metric = evaluate.load("confusion_matrix")
    model_checkpoint = "distilbert-base-uncased"
    num_labels = 2
    batch_size = 16

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor([8.0, 1.0], device=model.device)
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir="distilbert-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=True,
    )

    def preprocess_function(samples):
        return tokenizer(samples[data_column], truncation=True, padding=True)

    def compute_metrics(eval_pred):
        logits, references = eval_pred
        preds = np.argmax(logits, -1)
        print(confusion_metric.compute(predictions=preds, references=references))
        return metric.compute(predictions=preds, references=references, average="macro")

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


def _plot_confusion_matrix(cm, class_names, filename, save_path, model, text_column):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(cm)
    fig.colorbar(im, cax=cax, orientation="vertical")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}%", ha="center", va="center", color="black")
    ax.set_title(f"Confusion Matrix for {model}, column: {text_column}")
    fig.tight_layout()
    plt.savefig(Path(save_path) / filename, bbox_inches="tight")
    plt.clf()


def main(model, text_column, preprocess_type):
    dataset_path = "data/sentence_sets_trimmed.csv"
    random_state = 100
    save_path = "final_results"
    label_column = "applicant_gender"
    labels = ["female", "male"]
    Path(save_path).mkdir(exist_ok=True)

    df = pd.read_csv(dataset_path, encoding="unicode_escape")
    df.replace(to_replace=r"[^\w\s]", value="", regex=True, inplace=True)
    df = preprocess(df, text_column, preprocess_type)

    if model == "distilbert":
        ds = (
            Dataset.from_pandas(df)
            .rename_column(label_column, "label")
            .class_encode_column("label")
        )
        ds = ds.train_test_split(test_size=0.2)
        train_distilbert(ds, text_column)
        return

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_x = train_df[text_column]
    train_y = train_df[label_column]
    test_x = test_df[text_column]
    test_y = test_df[label_column]

    vectorizer = CountVectorizer()
    train_x_vec = vectorizer.fit_transform(train_x)
    test_x_vec = vectorizer.transform(test_x)

    clf = None
    if model == "svm":
        clf = SVC(kernel="rbf", C=1, class_weight={"female": 8, "male": 1})
    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=10, ccp_alpha=0.001, class_weight={"female": 8, "male": 1}
        )

    if clf:
        clf.fit(train_x_vec, train_y)
        preds = clf.predict(test_x_vec)
        f1 = f1_score(preds, test_y, average="macro")
        acc = accuracy_score(preds, test_y)
        mcc = matthews_corrcoef(preds, test_y)
        bal_acc = balanced_accuracy_score(preds, test_y)
        print(
            f"Model: {model} | Column: {text_column} | "
            f"Macro F1: {f1:.3f}, Accuracy: {acc:.3f}, "
            f"MCC: {mcc:.3f}, Balanced Acc: {bal_acc:.3f}"
        )
        print(classification_report(preds, test_y, target_names=labels))
        cm = confusion_matrix(test_y, preds, labels=clf.classes_).astype(np.int32)
        row_sums = cm.sum(axis=1)
        cm_percent = (np.around(cm / row_sums[:, np.newaxis], 2) * 100).astype(int)
        _plot_confusion_matrix(
            cm_percent,
            clf.classes_,
            f"{model}_cm_{text_column}.png",
            save_path,
            model,
            text_column,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="rf")
    parser.add_argument("-dataset", default="full_text")
    parser.add_argument("-preprocessing", default="none")
    args = parser.parse_args()
    main(args.model, args.dataset, args.preprocessing)
