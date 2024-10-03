import pickle
import os
import numpy as np


class Tagger:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)

        # Load the model during initialization
        self.model = self._load_model()
        self.mlb = self._load_mlb()
        self.vectorizer = self._load_vectorizer()

    def _load_model(self):
        # Load the pre-trained model using the absolute path
        model_path = os.path.join(self.base_dir, "data/logistic_regression_tfidf.pkl")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
        return model

    def _load_mlb(self):
        # Load the MultiLabelBinarizer using the absolute path
        mlb_path = os.path.join(self.base_dir, "data/mlb.pkl")
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
        # TODO

        # Embed the text
        embedded_text = self.embed_text(text)

        # Make prediction using the loaded model
        prediction = self.model.predict_proba(embedded_text)

        threshold = 0.05
        y_pred_custom = (prediction >= threshold).astype(int)
        predicted_tags = self.mlb.inverse_transform(y_pred_custom)

        # Convert prediction to a list (or format as needed)
        predicted_tags = prediction.tolist()

        return predicted_tags


# Main function to test the Tagger class
if __name__ == "__main__":
    # Initialize the Tagger
    tagger = Tagger()

    # Example input text
    text_input = "This is a sample text for testing the tag prediction."

    # Get predicted tags for the input text
    predicted_tags = tagger.predict_tags(text_input)

    # Output the predicted tags
    print(f"Predicted tags for the input text: {predicted_tags}")
