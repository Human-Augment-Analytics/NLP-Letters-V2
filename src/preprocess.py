import re
from sklearn.base import BaseEstimator, TransformerMixin

class GenderNeutralizer(BaseEstimator, TransformerMixin):
    def __init__(self, paths, lowercase=True):
        """
        paths: str or list of str
            Path(s) to the neutralization mapping file(s).
        lowercase: bool
            Whether to apply lowercase transformation to words.
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        self.lowercase = lowercase
        self.mapping = {}

    def fit(self, X=None, y=None):
        """Loads and parses mapping files."""
        self.mapping = self._load_mappings()
        return self

    def transform(self, X):
        """Replaces gendered words in text data."""
        if not isinstance(X, list):
            raise ValueError("Input X must be a list of strings.")

        return [self._replace_words(text) for text in X]

    def _load_mappings(self):
        """Parses neutral-gendered mapping files into a dictionary."""
        mapping = {}

        for path in self.paths:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    words = line.strip().split()
                    if len(words) < 2:
                        continue  # Ignore invalid lines
                    neutral_word, gendered_words = words[0], words[1:]

                    for gendered_word in gendered_words:
                        key = gendered_word.lower() if self.lowercase else gendered_word
                        mapping[key] = neutral_word

        return mapping

    def _replace_words(self, text):
        """Replaces gendered words in a given text."""
        words = text.split()
        return " ".join([self.mapping.get(word.lower() if self.lowercase else word, word) for word in words])