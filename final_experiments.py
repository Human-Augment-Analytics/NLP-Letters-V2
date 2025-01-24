from degender_pronoun import degenderizer
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
)

from pathlib import Path
import pandas as pd
import numpy as np

import argparse
import matplotlib.pyplot as plt
import scienceplots
import functools
import warnings
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets import load_metric
from torch import nn
import torch


warnings.filterwarnings("ignore")
plt.style.use('ieee')


degender_pronouns = {
    ' mr ': ' mx ',
    ' mrs ': ' mx ',
    ' ms ': ' mx ',
    ' miss ': ' mx ',
    ' mister ': ' mx ',
}

degender_nouns = {
    ' man ': ' person ',
    ' men ': ' persons ',
    ' woman ': ' person ',
    ' women ': ' persons ',
    " man's ": " person's",
    " men's ": " person's",
    " woman's ": " person's",
    " women's ": " person's",
    " gentleman ": " person ",
    " lady ": " person ",
    " gentleman's ": " person's ",
    " lady's ": " person's ",
}


# - Works on Google Colab - L4 due to RAM and GPU requirements
# Runs out of resources on 16GB M1 Pro
def train_distilbert(data_column):
    metric = evaluate.load('f1')

    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16
    task = "nlp-letters-{}-degendered-class-weighted".format(data_column)
    metric_name = "f1"
#    data_column = "s1_s2"
    model_name = model_checkpoint.split("/")[-1]
    num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 2 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([8.0, 1.0], device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=True,
    )

    def preprocess_function(sample):
        return tokenizer(sample[data_column], truncation=True, padding=True)

    confusion_metric = evaluate.load("confusion_matrix")

    def compute_metrics(eval_pred):
        x, y = eval_pred
        preds = np.argmax(x, -1)
        print(confusion_metric.compute(predictions=preds, references=y))
        return metric.compute(predictions=preds, references=y, average="macro")

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


def preprocess(df, data_column, preprocess_type):
    if preprocess_type == 'none':
        return df

    D = degenderizer()
    df[data_column] = df[data_column].apply(lambda x: D.degender(x) if len(x) > 5 else x)

    for k, v in degender_pronouns.items():
        df[data_column] = df[data_column].str.lower().replace(k,v)

    if preprocess_type == 'all':
        for k, v in degender_nouns.items():
            df[data_column] = df[data_column].str.lower().replace(k,v)
    return df


def replace(df, data_column):
    D = degenderizer()
    df[data_column] = df[data_column].apply(lambda x: D.degender(x) if len(x) > 5 else x)

    for k, v in degender_map.items():
        df[data_column] = df[data_column].str.replace(k,v)
    return df


def main(model, text_column, preprocess_type):

    dataset_path = "data/charlotte_dataset_final.csv"
    random_state = 100
    save_path = "final_results"
    Path(save_path).mkdir(exist_ok=True)
    label_column = "APPLICANT_GENDER"
    labels = ['FEMALE','MALE']

    # Load dataset
    df = pd.read_csv(dataset_path, encoding='unicode_escape')

    # Preprocess
    df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
    df = preprocess(df, text_column, preprocess_type)

    if model == "distilbert":
        train_distilbert(text_column)
        # branch off since training pipeline is completely different
        return


    train, test = train_test_split(df, test_size=0.2, random_state=random_state)
    train_x = train[text_column]
    train_y = train[label_column]


    test_x = test[text_column]
    test_y = test[label_column]

    count_vectorizer = CountVectorizer()
    train_x_processed = count_vectorizer.fit_transform(train_x)
    test_x_processed = count_vectorizer.transform(test_x)
    clf = None
    if model == "svm":
        best_parameters = {
            "kernel": "rbf",
            "C": 1,
            "class_weight": {"FEMALE": 8, "MALE": 1},
        }
        clf = SVC(**best_parameters)
    if model == "rf":
        best_parameters = {
            "n_estimators": 10,
            "ccp_alpha": 0.001,
            "class_weight": {"FEMALE": 8, "MALE": 1},
        }
        clf = RandomForestClassifier(**best_parameters)
    clf.fit(
        train_x_processed, train_y
    )
    predicted = clf.predict(test_x_processed)
    f1 = f1_score(predicted, test_y, average="macro")
    accuracy = accuracy_score(predicted, test_y)
    mcc = matthews_corrcoef(predicted, test_y)
    balance_acc = balanced_accuracy_score(predicted, test_y)
    print(f"model: {model}, dataset: {text_column}, Macro F1: {f1}, Accuracy: {accuracy}, MCC: {mcc}, Balanced Accuracy: {balance_acc}")
    print(classification_report(predicted, test_y, target_names=labels))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    def plot_confusion_matrix(cm, class_names):

        fig, ax = plt.subplots()

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(cm)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))

        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f"{np.around(cm[i, j], decimals=2)}%",
                            ha="center", va="center", color="k")

        ax.set_title(f"Confusion Matrix for {model}, column: {text_column}")
        fig.tight_layout()
        plt.savefig(Path(save_path) / f"{model}_cm_{text_column}.png", bbox_inches="tight")
        plt.clf()

    cm = confusion_matrix(test_y, predicted, labels=clf.classes_).astype(np.int32)
    row_sums = cm.sum(axis=1)
    cm = (np.around(cm / row_sums[:, np.newaxis], decimals=2) * 100).astype(np.int32)
    plot_confusion_matrix(cm, clf.classes_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-model', help='Model Type', default='rf')
    parser.add_argument('-dataset', help='Data column', default='full_text')
    parser.add_argument('-preprocessing', help='Preprocessing steps', default='replace')

    args = parser.parse_args()

    main(args.model, args.dataset, args.preprocessing)
