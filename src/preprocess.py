import re
from sklearn.base import BaseEstimator, TransformerMixin

class BasicDegenderizer(BaseEstimator, TransformerMixin):
    def __init__(self, paths, lowercase=True):
        """
        paths: str or list of str
            Path(s) to the neutralization mapping file(s).
        lowercase: bool
            Whether to use lowercase keys for mapping lookups.
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        self.lowercase = lowercase
        self.mapping = {}

    def fit(self, X=None, y=None):
        """Loads and parses mapping files."""
        self._load_mappings()
        return self

    def transform(self, X):
        """Replaces gendered words in text data while preserving case."""
        if not isinstance(X, list):
            raise ValueError("Input X must be a list of strings.")
        return [self._replace_words(text) for text in X]

    def _load_mappings(self):
        """Parses neutral-gendered mapping files into a dictionary.
        Expects each line to be in the format:
        neutral_word gendered_word gendered_word ...
        """
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
        self.mapping = mapping

    def _replace_words(self, text):
        """Replaces gendered words in a given text while preserving case."""
        def replace_word(word):
            # Use lower() for lookup regardless of original case
            lookup_key = word.lower() if self.lowercase else word
            if lookup_key in self.mapping:
                neutral = self.mapping[lookup_key]
                # Preserve case: uppercase, title-case, or lower-case
                if word.isupper():
                    return neutral.upper()
                elif word.istitle():
                    return neutral.capitalize()
                else:
                    return neutral
            return word

        words = text.split()
        return " ".join(replace_word(word) for word in words)
    
    
class AdvancedDegenderizer(BaseEstimator, TransformerMixin):
    def __init__(self, paths, ignore_case=True, match_possessives=True, match_punctuation=True):
        """
        paths: str or list of str
            Path(s) to the mapping file(s) in the format:
              neutral_word gendered_word gendered_word ...
        ignore_case: bool
            If True, regex matching is case-insensitive.
        match_possessives: bool
            If True, capture and preserve possessive suffixes (e.g. 's or ’s).
        match_punctuation: bool
            If True, capture and preserve trailing punctuation.
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        self.ignore_case = ignore_case
        self.match_possessives = match_possessives
        self.match_punctuation = match_punctuation
        self.mapping = {}
        self.patterns = {}

    def fit(self, X=None, y=None):
        """Loads the mapping from file(s) and compiles a regex pattern for each neutral word group."""
        self._load_mappings()
        self._compile_patterns()
        return self

    def transform(self, X):
        """
        For each input string in X (a list of strings), replace any occurrence of any gendered word
        (as defined by the mapping) with its corresponding neutral word, preserving word-level case,
        and (optionally) possessives and punctuation.
        """
        if not isinstance(X, list):
            raise ValueError("Input X must be a list of strings.")
        transformed_texts = []
        for text in X:
            new_text = text
            for neutral, pattern in self.patterns.items():
                new_text = pattern.sub(lambda m: self._replacement(m, neutral), new_text)
            transformed_texts.append(new_text)
        return transformed_texts

    def _load_mappings(self):
        """
        Loads mappings from file(s). Each line in a file should be in the format:
            neutral_word gendered_word gendered_word ...
        Returns a dictionary: { neutral_word: [gendered_word, ...], ... }.
        """
        mapping = {}
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    words = line.strip().split()
                    if len(words) < 2:
                        continue
                    neutral = words[0]
                    gendered = words[1:]
                    # Optionally, you can force lowercasing for consistency in matching.
                    # With ignore_case=True, regex will be case-insensitive, but we build our patterns
                    # using the gendered words as provided.
                    if neutral in mapping:
                        mapping[neutral].extend(gendered)
                    else:
                        mapping[neutral] = gendered
        # Remove duplicates
        for neutral in mapping:
            mapping[neutral] = list(set(mapping[neutral]))
        self.mapping = mapping

    def _compile_patterns(self):
        """
        For each neutral word and its list of gendered words, compile a regex pattern that
        matches any of those words, with optional groups for possessives and punctuation.
        """
        patterns = {}
        for neutral, gendered_list in self.mapping.items():
            # Sort by length descending so longer words match first.
            sorted_gendered = sorted(gendered_list, key=len, reverse=True)
            alternation = "|".join(re.escape(word) for word in sorted_gendered)
            # Build pattern based on options.
            if self.match_possessives or self.match_punctuation:
                # Group 1: the gendered word
                # Group 2: optional possessive suffix (['’]s) if match_possessives is True
                # Group 3: optional trailing punctuation if match_punctuation is True
                poss = r"((?:['’]s)?)" if self.match_possessives else ""
                punct = r"([^\w\s]*)" if self.match_punctuation else ""
                pattern_str = r"\b(" + alternation + r")" + poss + punct
            else:
                pattern_str = r"\b(" + alternation + r")\b"
            flags = re.IGNORECASE if self.ignore_case else 0
            patterns[neutral] = re.compile(pattern_str, flags)
            
        self.patterns = patterns
    
    def _replacement(self, match, neutral):
        """
        Given a regex match and the target neutral word, return the replacement string.
        Preserves the original word's case and reattaches any captured possessive and punctuation.
        """
        # Group 1: the matched gendered word
        orig_word = match.group(1)
        poss = match.group(2) if self.match_possessives and match.lastindex >= 2 and match.group(2) else ""
        punct = match.group(3) if self.match_punctuation and match.lastindex >= 3 and match.group(3) else ""
        # Preserve case: adjust neutral based on orig_word's case
        neutral_adj = self._preserve_case(orig_word, neutral)
        return neutral_adj + poss + punct

    def _preserve_case(self, original, neutral):
        """Adjust the neutral word's case to match that of the original word."""
        if original.isupper():
            return neutral.upper()
        elif original.istitle():
            return neutral.capitalize()
        else:
            return neutral

