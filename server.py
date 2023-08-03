from flask import Flask, request, render_template
from fastai.vision.all import load_learner
from fastai.vision.all import Image
from pathlib import Path
import io
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import ssl
from flask_cors import CORS

ssl._create_default_https_context = ssl._create_unverified_context

send_grid_api_key = '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'
admin_mails = ['glodymbutwile@gmail.com', 'audrynshidi@gmail.com', '18ck040@esisalama.org']

PATH_TO_MODEL = Path('model.pkl')
TEMP_IMAGE_NAME = 'test.jpeg'

app = Flask(__name__)
CORS(app)

recyclers = {}

def get_type_dechet(pred):
    if 'bouteille' in pred:
        return 'non-bio'
    return 'bio'

def send_mail(recycler):
    if recycler['is_full']:
        message = Mail(
            from_email='admin@m-capital.net',
            to_emails=admin_mails,
            subject='Sending with Twilio SendGrid is Fun',
            html_content='<strong>Recycler {} is full</strong>'.format(recycler['name']))
        try:
            sg = SendGridAPIClient(send_grid_api_key)
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print(e.message)
        

@app.route("/")
def home():
    return render_template('index.html', recyclers=recyclers)

@app.route("/api-token-auth/", methods=['POST'])
def apit_token():
    return {
        "token": "token"
    }


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    image_bytes = io.BytesIO(base64.b64decode(data['data']))
    image = Image.open(image_bytes)
    image.save(TEMP_IMAGE_NAME)
    learner = load_learner(PATH_TO_MODEL)
    pred, pred_idx, probs = learner.predict(Path(TEMP_IMAGE_NAME))
    return {
        "pred": get_type_dechet(pred),
        "pred_idx": pred_idx.item(),
        "probs": probs.tolist()
    }
    
@app.route("/recyclers", methods=['GET'])
def recyclers_list():
    return recyclers
    
@app.route("/recyclers/data", methods=['POST'])
def recycler_data():
    data = request.json
    if data != None:
        is_full = False
        if recyclers.get(data['name']) != None:
            is_full = recyclers[data['name']]['is_full']
        
        recyclers[data['name']] = data
        
        if data['is_full'] and not is_full:
            send_mail(data)
        return data
    return {
        "error": True
    }

app.run(host="0.0.0.0", port=8090)