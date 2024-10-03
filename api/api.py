from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class Tagger:
    def __init__(self):
        print("ok")
        pass

    def predict_tags(self, text: str):

        return ["python", "java"]


app = FastAPI()

# Create a global Tagger object which loads the model when the API starts
tagger = Tagger()


# Define the data model for the incoming request
class TextInput(BaseModel):
    text: str


# Prediction endpoint
@app.post("/predict")
def predict_tags(data: TextInput):
    # Use the tagger object to get the predictions
    predicted_tags = tagger.predict_tags(data.text)
    return {"predicted_tags": predicted_tags}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
