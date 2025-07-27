class TextProcessor:
    def __init__(
        self,
        teencode_path="teencode.csv",
        stopword_path="stopwords.txt",
        phrase_rules_path="phrase_rules.csv",
    ):
        self.teencode_dict = self.load_teencode_dict(teencode_path)
        self.stopwords = self.load_stopwords(stopword_path)
        self.phrase_rules = self.load_phrase_rules(phrase_rules_path)
        self.negation_prefixes = {
            "không",
            "chưa",
            "chẳng",
            "chớ",
            "chả",
            "đừng",
        }
        self.negative_indicators = {"bị", "nên", "được", "đáng", "thiếu", "mất"}

    def load_teencode_dict(self, path):
        import csv

        with open(path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return {row["Teencode"]: row["Meaning"] for row in reader}

    def load_stopwords(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())

    def load_phrase_rules(self, path):
        import csv

        rules = {}
        with open(path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Phrase"] and row["Normalized"]:
                    rules[row["Phrase"].strip()] = row["Normalized"].strip()
        return rules

    def clean_text(self, text):
        import re
        import string
        import emoji

        if not isinstance(text, str):
            return ""

        text = text.replace("_x000D_", " ")
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r"[:;][-~]?[)D(/\\|pP]", "", text)
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        text = re.sub(r"(\d+)\s+([^\d\s]+)", r"\1_\2", text)

        text = text.translate(
            str.maketrans({p: " " for p in string.punctuation if p != "_"})
        )

        return text

    def normalize_teencode(self, text):
        import string

        words = text.split()
        converted_words = []
        for word in words:
            core = word.strip(string.punctuation)
            if core in self.teencode_dict:
                new_word = word.replace(core, self.teencode_dict[core])
                new_word = new_word.replace(" ", "_")
                converted_words.append(new_word)
            else:
                converted_words.append(word)
        return " ".join(converted_words)

    def normalize_repeated_chars(self, text):
        import re

        return " ".join([re.sub(r"(.)\1{1,}$", r"\1", word) for word in text.split()])

    def apply_phrase_rules(self, text):
        import re

        for phrase, normalized in sorted(
            self.phrase_rules.items(), key=lambda x: len(x[0]), reverse=True
        ):
            text = re.sub(rf"\b{re.escape(phrase)}\b", normalized, text)
        return text

    def process_negations(self, text):
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            # không có + từ (ví dụ: không có bọc)
            if i + 2 < len(words) and words[i] == "không" and words[i + 1] == "có":
                result.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
                i += 3
            # phủ định + bị + từ (ví dụ: không bị rách)
            elif (
                i + 2 < len(words)
                and words[i] in self.negation_prefixes
                and words[i + 1] in self.negative_indicators
            ):
                result.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
                i += 3
            # phủ định + từ (ví dụ: không tốt)
            elif i + 1 < len(words) and words[i] in self.negation_prefixes:
                result.append(f"{words[i]}_{words[i+1]}")
                i += 2
            # bị/nên/... + từ (ví dụ: nên ủng_hộ)
            elif i + 1 < len(words) and words[i] in self.negative_indicators:
                result.append(f"{words[i]}_{words[i+1]}")
                i += 2
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)

    def process_special_word(self, text, special_word=""):
        new_text = ""
        text_lst = text.split()
        i = 0
        if special_word in text_lst:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                if word == special_word:
                    next_idx = i + 1
                    if next_idx <= len(text_lst) - 1:
                        word = word + "_" + text_lst[next_idx]
                    i = next_idx + 1
                else:
                    i = i + 1
                new_text = new_text + word + " "
        else:
            new_text = text
        return new_text.strip()

    def tokenize(self, text):
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)

    def remove_stopwords(self, text):
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and f"{words[i]}_{words[i+1]}" in self.stopwords:
                i += 2
                continue
            if words[i] not in self.stopwords:
                result.append(words[i])
            i += 1
        return " ".join(result)

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.normalize_teencode(text)
        text = self.normalize_repeated_chars(text)
        text = self.apply_phrase_rules(text)
        text = self.tokenize(text)
        text = self.process_negations(text)
        text = self.remove_stopwords(text)
        text = self.process_special_word(text, special_word="giao_hàng")
        text = self.process_special_word(text, special_word="giao")
        text = self.process_special_word(text, special_word="bao")
        text = self.process_special_word(text, special_word="bọc")
        text = self.process_special_word(text, special_word="bao_bọc")
        text = self.process_special_word(text, special_word="đóng_gói")
        return text
