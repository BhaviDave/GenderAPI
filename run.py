import os
# io, uuid
import requests
from flask import Flask, request, jsonify, render_template, send_file
import tensorflow as tf

from PIL import Image

import numpy as np
#import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'some-secret-for-gender-recognition'
app.model = None



# @app.route("/")
# def check():
#     return jsonify({'Message': "Server Running!"})


@app.route("/")
def upload_photo():
    return render_template('upload.html')


@app.route("/init")
def init_model():
    init_model()
    return "Init done"


@app.route("/file/<file_name>")
def display_file(file_name):
    print ("in get file")


    file_path = get_file_path(file_name)
    return send_file(file_path)


@app.route("/predict", methods=['POST'])
def precict_gender():
    result = []

    files = request.files.getlist("images")
    for file in files:
        classes = ['female', 'male']
        img_size = 224
        pil_img = Image.open(file)
        #numpy_image = np.array(pil_img)
        img_array_224 = pil_img.resize((img_size, img_size))
        img_array_224 = np.array(img_array_224)
        #img_array_224 = cv2.resize(numpy_image, (img_size, img_size))
        img_array_224_255 = img_array_224 / 255
        new_array = np.array(img_array_224_255).reshape(-1, img_size, img_size, 3)
        new_model = load_model()
        gender_prediction = new_model.predict(new_array)

        gender_prediction_pct = str(int(np.max(gender_prediction) * 100))

        gender_label_arg = np.argmax(gender_prediction)

        gender_text = classes[gender_label_arg]

        result.append({

                        'gender_text':gender_text,
                        'Confidence': str(gender_prediction_pct),

                    })

        print(result)
    return render_template('result.html', result=result)


def get_app_model():
    return init_model()


def init_model():
    if (app.model == None):
        print ("In init model")
        app.model = load_model()
    else:
        print ("Model initialized")

    return app.model


def load_model():
    new_model = tf.keras.models.load_model(r'resnet152v2-128-16-2.h5')
    return new_model

def get_file_path(file_name):
    return os.path.join("./uploads", file_name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)

