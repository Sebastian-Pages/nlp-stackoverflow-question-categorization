# data science
import pandas as pd
import numpy as np

# nlp
import re
import string
import contractions

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

from tqdm import tqdm
from collections import Counter
import ast

tqdm.pandas(desc="Running Clean Operation")

# warnings
import warnings

warnings.filterwarnings("ignore")


def lower_case(text):
    return text.lower()


def remove_spaces_tabs(text):
    return " ".join(text.split())


# def remove_punct(text):
#     translator = str.maketrans("", "", string.punctuation)
#     return text.translate(translator)


def remove_punct(text):
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_non_ascii(text):
    return re.sub(r"[^\x20-\x7E]", "", text)
    # pattern = r"[^\x00-\x7F]"
    # return re.sub(pattern, "", text)


# def remove_single_char(text, keep=""):
#     pattern = r"\b(?![" + re.escape(keep) + r"]\b)[a-zA-Z]\b"
#     return re.sub(pattern, "", text)


def remove_single_char(text, keep=""):
    tokens = re.findall(r"\S+|\s+", text)
    keep_set = set(keep)
    result = "".join(
        [
            token
            for token in tokens
            if len(token.strip()) > 1 or token in keep_set or token.isspace()
        ]
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result


def remove_tags(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_stopwords(text, add_stopwords=None, keep_words=None):
    stop_words = set(stopwords.words("english"))
    if add_stopwords:
        stop_words.update(add_stopwords)
    if keep_words:
        stop_words -= set(keep_words)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


def remove_digits(text):
    return re.sub(r"\d", "", text)


def expand_contractions(text):
    return contractions.fix(text)


# def lemmatize_text(text, lemmatizer):
#     word_tokens = word_tokenize(text)
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
#     return " ".join(lemmatized_tokens)


def lemmatize_text(text, lemmatizer, min_length=1):
    word_tokens = word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) if len(word) > min_length else word
        for word in word_tokens
    ]
    return " ".join(lemmatized_tokens)


def stem_text(text, stemmer):
    word_tokens = word_tokenize(text)
    stemmed_tokens = " ".join([stemmer.stem(word) for word in word_tokens])
    return " ".join(stemmed_tokens)


def run_cleaning(
    text, pipeline, lemmatizer=None, stemmer=None, keep=None, extra_stopwords=None
):
    tokens = text
    for transform in pipeline:
        # if lemmatize or stem function pass in, perform transformation
        if transform.__name__ == "lemmatize_text":
            tokens = transform(tokens, lemmatizer)
        elif transform.__name__ == "stem_text":
            tokens = transform(tokens, stemmer)
        elif transform.__name__ == "remove_single_char":
            tokens = transform(tokens, keep=keep)
        elif transform.__name__ == "remove_stopwords":
            tokens = transform(tokens, add_stopwords=extra_stopwords)
        else:
            tokens = transform(tokens)
    return tokens


def check_for_errors(df, col_name):
    cleaned_text = " ".join(df[col_name].tolist())

    # Check for unusual tokens
    tokens = word_tokenize(cleaned_text)
    unusual_tokens = list(set([token for token in tokens if len(token) <= 1]))
    print(f"Unusual tokens (single characters): {unusual_tokens}")

    # Check for residual noise
    residual_noise = list(set(re.findall(r"[^a-zA-Z\s]", cleaned_text)))
    print(f"Residual noise: {residual_noise}")

    # Check for extra whitespace
    extra_spaces_present = bool(re.search(r"\s{2,}", cleaned_text))
    print(f"Extra spaces: {extra_spaces_present}")


# def filter_top_n_tags(df, column='Tags', top_n=1000):

#     tag_counts = Counter(tag for tags in df[column] for tag in tags)
#     top_tags = set([tag for tag, _ in tag_counts.most_common(top_n)])
#     df[column] = df[column].apply(lambda x: [tag for tag in x if tag in top_tags])
#     df = df[df[column].apply(lambda x: len(x) > 0)]

#     return df


# def filter_top_n_tags(df, column="Tags", top_n=1000):

#     tag_counts = Counter(tag for tags in df[column] for tag in tags)
#     top_tags = set([tag for tag, _ in tag_counts.most_common(top_n)])
#     filtered_tags = df[column].apply(lambda x: [tag for tag in x if tag in top_tags])
#     filtered_tags = filtered_tags[filtered_tags.apply(lambda x: len(x) > 0)]

#     return filtered_tags


def filter_top_n_tags(df, column="Tags", top_n=1000):

    tag_counts = Counter(tag for tags in df[column] for tag in tags)
    top_tags = set(tag for tag, _ in tag_counts.most_common(top_n))
    df[column] = df[column].apply(lambda x: [tag for tag in x if tag in top_tags])
    df = df[df[column].apply(lambda x: len(x) > 0)]

    return df


def count_tags(df, column="Tags"):

    df[column] = df[column].apply(ast.literal_eval)

    # Flatten the list of tags in the column
    tags_flat = [tag for tags in df[column] for tag in tags]

    # Count the total and unique tags
    unique_tags = len(set(tags_flat))

    # Print the results
    print(f"Number of unique tags: {unique_tags}")

    return unique_tags
