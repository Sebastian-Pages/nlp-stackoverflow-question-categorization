import uvicorn
import pickle
import os
import re
import string
import warnings
import contractions
from fastapi import FastAPI
from pydantic import BaseModel

# import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")


def lower_case(text):
    return text.lower()


def remove_spaces_tabs(text):
    return " ".join(text.split())


def remove_punct(text):
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_non_ascii(text):
    return re.sub(r"[^\x20-\x7E]", "", text)


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


def clean(text):

    # Pipeline definition
    pipeline = [
        lower_case,
        remove_tags,
        expand_contractions,
        remove_spaces_tabs,
        remove_url,
        remove_digits,
        remove_punct,
        remove_non_ascii,
        remove_stopwords,
        remove_single_char,
        lemmatize_text,
        remove_single_char,
    ]

    custom_stopwords = [
        "error",
        "gt",
        "lt",
        "quot",
        "using",
        "like",
        "would",
        "want",
        "work",
        "com",
        "way",
        "need",
    ]

    # Pipeline Setup
    lemmatizer = WordNetLemmatizer()

    clean_text = run_cleaning(
        text,
        pipeline=pipeline,
        lemmatizer=lemmatizer,
        keep="rc",
        extra_stopwords=custom_stopwords,
    )

    return clean_text


class Tagger:
    def __init__(self):
        pass
        self.base_dir = os.path.dirname(__file__)

        # Load the model during initialization
        self.model = self._load_model()
        self.mlb = self._load_mlb()
        self.vectorizer = self._load_vectorizer()

    def _load_model(self):
        # Load the pre-trained model using the absolute path
        model_path = os.path.join(
            self.base_dir, "data/logistic_regression_tfidf_model.pkl"
        )
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
        return model

    def _load_mlb(self):
        # Load the MultiLabelBinarizer using the absolute path
        mlb_path = os.path.join(self.base_dir, "data/mlb_model.pkl")
        with open(mlb_path, "rb") as mlb_file:
            mlb = pickle.load(mlb_file)
        print("MultiLabelBinarizer loaded successfully.")
        return mlb

    def _load_vectorizer(self):
        # Load the vectorizer using the absolute path
        vectorizer_path = os.path.join(self.base_dir, "data/tfidf_vectorizer.pkl")
        with open(vectorizer_path, "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        print("Vectorizer loaded successfully.")
        return vectorizer

    def embed_text(self, text: str):
        text_emb = self.vectorizer.transform(text)
        return text_emb

    def predict_tags(self, text: str):
        # Cleaning
        text = clean(text)

        # Embed the text
        embedded_text = self.embed_text([text])

        # Make prediction using the loaded model
        prediction = self.model.predict_proba(embedded_text)

        threshold = 0.175
        y_pred_custom = (prediction >= threshold).astype(int)
        predicted_tags = self.mlb.inverse_transform(y_pred_custom)

        # Convert prediction to a list (or format as needed)
        predicted_tags = predicted_tags

        return predicted_tags


app = FastAPI()

# Create a global Tagger object which loads the model when the API starts
tagger = Tagger()


# Define the data model for the incoming request
class TextInput(BaseModel):
    text: str


# Default route to check if the API is running
@app.get("/")
def read_root():
    return {"message": "API is running!"}


# Prediction endpoint
@app.post("/predict")
def predict_tags(data: TextInput):
    # Use the tagger object to get the predictions
    predicted_tags = tagger.predict_tags(data.text)
    return {"predicted_tags": predicted_tags}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
