import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from abc import ABC, abstractmethod


# Define the abstract base class
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model on provided data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for input data."""
        pass

    @abstractmethod
    def test(self, X, y):
        """Evaluate the model on test data and return performance metrics."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model to a specified path."""
        pass

    @abstractmethod
    def load(self, path):
        """Load a model from a specified path."""
        pass


# Define the DistilBERT classifier with extra flexibility
class DistilBERTClassifier(BaseModel):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=2,
        extra_layers=None,
        dropout_rate=0.2,
        class_weights=None,
        pooling="cls",
    ):
        # Load configuration and update with custom settings
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.pooling = pooling  # "cls" or "mean"

        # Load tokenizer and initialize the inner transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._DistilBERTClassifier(
            model_name, self.config, dropout_rate, extra_layers, class_weights
        )

    class _DistilBERTClassifier(nn.Module):
        def __init__(
            self, model_name, config, dropout_rate, extra_layers, class_weights
        ):
            super().__init__()
            # Load the transformer model with configuration
            self.transformer = AutoModel.from_pretrained(model_name, config=config)

            # Pre-classifier: transform transformer output to hidden space
            self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
            self.pre_classifier_act = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate)

            # Build extra dense layers if specified
            if extra_layers is None:
                extra_layers = []
            self.extra_layers = nn.ModuleList()
            in_features = config.hidden_size
            for layer_size in extra_layers:
                self.extra_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, layer_size),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_rate),
                    )
                )
                in_features = layer_size

            # Final classifier layer mapping to number of labels
            self.classifier = nn.Linear(in_features, config.num_labels)

            # Set up loss function with optional class weighting
            if class_weights is not None:
                self.loss_fn = nn.CrossEntropyLoss(
                    weight=torch.tensor(class_weights, dtype=torch.float32)
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss()
            self.pooling = config.pooling

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Pooling: choose between using the [CLS] token or mean pooling
            if self.pooling == "mean":
                pooled = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            x = self.pre_classifier(pooled)
            x = self.pre_classifier_act(x)
            x = self.dropout(x)
            for layer in self.extra_layers:
                x = layer(x)
            logits = self.classifier(x)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

    def train(
        self,
        X,
        y,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        output_dir="./distilbert_classifier",
    ):
        """Train the model using Hugging Face's Trainer API."""
        # Create a Dataset from the input data
        dataset = Dataset.from_dict({"text": X, "labels": y})
        dataset = dataset.map(self._tokenize, batched=True)
        dataset = dataset.remove_columns(["text"])

        # Split into training and evaluation sets (80/20 split)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        # Set up training arguments
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
        """Return model predictions for input texts."""
        self.model.eval()
        tokens = self.tokenizer(
            X,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.model.transformer.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
        preds = torch.argmax(outputs["logits"], dim=1)
        return preds.cpu().numpy()

    def test(self, X, y):
        """Evaluate the model on test data and return metrics."""
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="macro"
        )
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def save(self, path):
        """Save the transformer model and tokenizer."""
        self.model.transformer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        """Load the transformer model and tokenizer from a path."""
        self.model.transformer = AutoModel.from_pretrained(path, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def _tokenize(self, example):
        """Tokenize the input text."""
        return self.tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=512
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
