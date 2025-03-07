import re

input_filename = "../data/words.txt"
output_filename = "../data/unique_words.txt"

unique_words = set()

with open(input_filename, mode="r", encoding="utf-8") as file:
    for line in file:
        words = re.findall(r"\b\w+\b", line.lower())
        unique_words.update(words)

with open(output_filename, mode="w", encoding="utf-8") as output_file:
    for word in sorted(unique_words):
        output_file.write(word + "\n")

