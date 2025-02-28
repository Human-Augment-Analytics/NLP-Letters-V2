import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import numpy as np
import os
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all models, ensuring a consistent interface."""

    @abstractmethod
    def train(self, X, y):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Run inference on input data and return predictions."""
        pass

    @abstractmethod
    def test(self, X, y):
        """Evaluate the model on test data and return performance metrics."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model to a specified file."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the model from a saved file."""
        pass


class DistilBERTModel(BaseModel):
    """DistilBERT-based classifier implementing BaseModel."""

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=2,
        dropout_rate=0.2,
        class_weights=None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

        # Handle class weights for imbalanced datasets
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        X,
        y,
        epochs=3,
        batch_size=16,
        test_size=0.2,
        learning_rate=2e-5,
        weight_decay=0.01,
        output_dir="./distilbert_model",
    ):

        # Restructure, tokenize and split dataset
        dataset = Dataset.from_dict({"text": X, "labels": y})
        dataset = dataset.map(self._tokenize, batched=True)
        dataset = dataset.train_test_split(test_size)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

    def predict(self, X):
        self.model.eval()
        tokens = self._tokenize({"text": X})
        input_ids = torch.tensor(tokens["input_ids"]).to(self.device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return torch.argmax(outputs.logits, dim=1).cpu().numpy()

    def test(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="macro"
        )

        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def _tokenize(self, batch):
        """Helper function to tokenize input text."""
        return self.tokenizer(
            batch["text"], truncation=True, padding=True, max_length=512
        )

    def _compute_metrics(self, eval_pred):
        """Computes classification metrics."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )

        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
