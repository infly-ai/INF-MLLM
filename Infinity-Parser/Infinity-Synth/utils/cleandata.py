import re
import random


def remove_non_zh_en_characters(text: str) -> str:
    """
    Remove characters that are not Chinese, English, digits, or common punctuation.
    Keeps Chinese/English punctuation and whitespace.
    """
    # Remove newline first
    text = text.replace("\n", "")

    # Regex: keep Chinese, English, digits, Chinese punctuation, basic punctuation, whitespace
    pattern = re.compile(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\u3000-\u303f\uff00-\uffef.,!?;:()\[\]{}“”‘’\'\"\-\—\s]"
    )
    return re.sub(pattern, "", text)


def clean_dictionary_parts(parts: dict) -> dict:
    """
    Recursively clean dictionary keys and values by removing unwanted characters.
    """
    cleaned_parts = {}

    for key, value in parts.items():
        cleaned_key = remove_non_zh_en_characters(key)

        if isinstance(value, str):
            cleaned_value = remove_non_zh_en_characters(value)
        elif isinstance(value, dict):
            cleaned_value = clean_dictionary_parts(value)  # recursive
        else:
            cleaned_value = value  # leave non-string values unchanged

        cleaned_parts[cleaned_key] = cleaned_value

    return cleaned_parts


def split_text_into_paragraphs(text: str, min_length: int = 200, max_length: int = 400):
    """
    Randomly split text into paragraphs based on sentence boundaries.
    This split does NOT preserve semantic meaning; it is purely random.

    Args:
        text: input text
        min_length: minimum characters per paragraph
        max_length: maximum characters per paragraph

    Returns:
        list of paragraph strings
    """

    sentence_endings = re.compile(r"[。！？\.\!\?]+")
    paragraphs = []
    last_end = 0

    while last_end < len(text):

        target_length = random.randint(min_length, max_length)
        next_possible_end = last_end + target_length

        if next_possible_end >= len(text):
            paragraphs.append(text[last_end:].strip())
            break

        match = sentence_endings.search(text, next_possible_end)

        if match:
            end = match.end()
        else:
            end = next_possible_end  # fallback if no punctuation found

        paragraphs.append(text[last_end:end].strip())
        last_end = end

    return paragraphs
