import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import joblib
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def DropIrrelevant(Data,cols):
  return Data.drop(cols,axis=1)

def remove_numeric(text):
    return re.sub(r'\d+', '', text)

    return re.sub(r'\d+', '', text)
def removeStopWords(text):
  text = text.split()
  newText = ""
  for word in text:
    if word not in stop_words:
      newText += word+" "
  return newText.strip()


def applyFuncs(Data, textLabel):
    Data[textLabel] = Data[textLabel].apply(remove_numeric)
    Data[textLabel] = Data[textLabel].apply(lambda x: x.replace('|', ',').lower())
    Data[textLabel] = Data[textLabel].apply(lambda x: re.sub("[^a-z0-9\s]", "", x))
    Data[textLabel] = Data[textLabel].apply(lambda x: re.sub("\s+", " ", x))
    Data[textLabel] = Data[textLabel].apply(removeStopWords)
    return Data

def PreprocessPipeline(DataFrameObj, textCol):
    DataFrameObj = DataFrameObj[[textCol]]
    DataFrameObj = applyFuncs(DataFrameObj, textCol)
    return DataFrameObj


import tensorflow as tf
from tensorflow.keras.metrics import Metric, Precision, Recall

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assume y_pred is already passed through softmax if necessary
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.cond(tf.greater(p + r, 0),
                       lambda: 2 * ((p * r) / (p + r + tf.keras.backend.epsilon())),
                       lambda: 0.0)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Define the Feedforward Neural Network model architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(500,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(36, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), F1Score()])

# Load the model weights
model.load_weights("FNN_Job_Recommender_weights.h5")

# Load the saved model components
vectorizer = joblib.load('vectorizer.pkl')
one_hot_encoder = joblib.load('ONE_HOT_ENC_JT.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def createFeatures(text_data):
    features = vectorizer.transform(text_data)
    return features.toarray()

def predict_job_title(features):
    predictions = model.predict(features)
    job_indices = np.argmax(predictions, axis=1)
    job_titles = label_encoder.inverse_transform(job_indices)
    return job_titles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2 style="text-align:center">
        Click
        
        <a href="/docs">here for API Documentation</a>
    </h2>
    """

@app.post("/predict", tags=["Prediction"])
async def predict_job_skills(request: Request):
    body = await request.json()
    skills = body.get("Skills")
    if not skills:
        raise HTTPException(status_code=400, detail="Skills field is missing in the request body")
    else:
        data_frame = pd.DataFrame({"Skills": [skills]})
        preprocessed_data = PreprocessPipeline(data_frame, "Skills")
        features = createFeatures(preprocessed_data['Skills'])
        print("Shape of input features:", features.shape)
        predicted_job_titles = predict_job_title(features)
        return {"Predicted Job Title": predicted_job_titles[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
