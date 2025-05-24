import csv
import string
import emoji
import re
from pyvi import ViTokenizer
import pandas as pd
from functools import lru_cache


# === Cached loading ===
# @lru_cache(maxsize=1)
def load_teencode_dict():
    with open(
        "../02.Dataset/vietnamese/teencode.csv", mode="r", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        return {row["Teencode"]: row["Meaning"] for row in reader}


# @lru_cache(maxsize=1)
def load_stopwords():
    with open("../02.Dataset/vietnamese/stopwords.txt", "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


# @lru_cache(maxsize=1)
def load_phrase_rules():
    rules = {}
    with open(
        "../02.Dataset/vietnamese/phrase_rules.csv", mode="r", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Phrase"] and row["Normalized"]:
                rules[row["Phrase"].strip()] = row["Normalized"].strip()
    return rules


# === Text Cleaning ===
def clean_icons(text):
    if pd.isna(text):
        return ""
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[:;][-~]?[)D(/\\|pP]", "", text)
    return text.replace("_x000D_", " ")


def lower(text):
    return text.lower().strip() if isinstance(text, str) else ""


def remove_links(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# === Teencode conversion ===
def convert_teencode_to_vietnamese(sentence):
    if pd.isna(sentence):
        return ""
    dictionary = load_teencode_dict()
    words = sentence.split()
    converted_words = []
    for word in words:
        core_word = word.strip(string.punctuation)
        if core_word in dictionary:
            new_word = word.replace(core_word, dictionary[core_word])
            converted_words.append(new_word)
        else:
            converted_words.append(word)
    return " ".join(converted_words)


# === Stopwords removal ===
def remove_vietnamese_stopwords(text):
    if pd.isna(text):
        return ""
    stopwords = load_stopwords()
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if i < len(words) - 1:
            two_word = f"{words[i]}_{words[i+1]}"
            if two_word in stopwords:
                i += 2
                continue
        if words[i] not in stopwords:
            result.append(words[i])
        i += 1
    return " ".join(result)


# === Tokenization ===
def word_tokenize(text):
    if pd.isna(text):
        return ""
    return ViTokenizer.tokenize(text)


def apply_phrase_rules(text):
    if pd.isna(text):
        return ""
    rules = load_phrase_rules()

    for phrase, normalized in sorted(
        rules.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = re.sub(rf"\b{re.escape(phrase)}\b", normalized, text)
    return text


# === Full Preprocessing Pipeline ===
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = clean_icons(text)
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_links(text)
    text = convert_teencode_to_vietnamese(text)
    text = apply_phrase_rules(text)
    text = word_tokenize(text)
    text = remove_vietnamese_stopwords(text)
    return text
