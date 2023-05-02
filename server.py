from flask import Flask, request
from fastai.vision.all import load_learner
from fastai.vision.all import Image
from pathlib import Path
import io
import base64

PATH_TO_MODEL = Path('model.pkl')
TEMP_IMAGE_NAME = 'test.jpeg'

app = Flask(__name__)

@app.route("/")
def home():
    return '<h1>WELCOME TO SR(smart recycler) by chrino kabwe<h1>'

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    image_bytes = io.BytesIO(base64.b64decode(data['data']))
    image = Image.open(image_bytes)
    image.save(TEMP_IMAGE_NAME)
    learner = load_learner(PATH_TO_MODEL)
    pred, pred_idx, probs = learner.predict(Path(TEMP_IMAGE_NAME))
    return {
        "pred": pred,
        "pred_idx": pred_idx.item(),
        "probs": probs.tolist()
    }

app.run(host="0.0.0.0", port=8090)