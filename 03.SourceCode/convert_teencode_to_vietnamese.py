import string
import csv


# Convert comment to a full sentence
def convert_teencode_to_vietnamese(sentence, dictionary):
    words = sentence.split()
    converted_words = []
    for word in words:
        if word in dictionary:
            converted_words.append(dictionary[word])
            continue

        punctuation = ""
        temp_word = word
        while temp_word and temp_word[-1] in string.punctuation:
            punctuation = temp_word[-1] + punctuation
            temp_word = temp_word[:-1]
        if temp_word in dictionary:
            converted_words.append(dictionary[temp_word] + punctuation)
            continue

        leading_punctuation = ""
        temp_word = word
        while temp_word and temp_word[0] in string.punctuation:
            leading_punctuation += temp_word[0]
            temp_word = temp_word[1:]
        if temp_word in dictionary:
            converted_words.append(leading_punctuation + dictionary[temp_word])
            continue

        converted_words.append(word)

    return " ".join(converted_words)


dictionary = {}
with open(
    "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\teencode.csv",
    mode="r",
    encoding="utf-8",
) as file:
    reader = csv.DictReader(file)
    for row in reader:
        dictionary[row["Teencode"]] = row["Meaning"]

sentence = "hôm nay cta đi đâu?"

converted_sentence = convert_teencode_to_vietnamese(sentence, dictionary)

print("Câu sau khi chuyển đổi:", converted_sentence)
