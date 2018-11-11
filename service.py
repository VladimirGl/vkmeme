# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#   python simple_request.py

# import the necessary packages

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

import base64
import numpy as np

import sklearn

import io
import pandas as pd

from glob import glob
import pickle

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

import requests

import base64
from glob import glob
import pickle
import pandas as pd

import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

graph = tf.get_default_graph()




def load_meme_meta(path):
    pd_data = pd.read_csv(path, sep='\|\|\|', header=None, engine='python')
    
    return pd_data

def load_all_pics(path):
    pics = glob(path + '*')
    
    results = dict()
    for i in pics:
        pic = base64.b64encode(open(i, "rb").read()).decode('utf-8')
        results[i.replace('|', '/')] = pic
        
    return results

activations_x = np.load('dump/all_activations.npy')
activations_y = np.load('dump/all_meme_ys.npy')

with open('dump/meme_map.pkl', 'rb') as f:
    inverse_map = pickle.load(f)
    
meme_meta = load_meme_meta('dump/megaresult.txt')
pics = load_all_pics('dump/images_main/')


def get_nearest_n(image, n):
    global graph
    with graph.as_default():
        activations = model.predict(image)
    
    dists = np.linalg.norm(activations_x - activations, axis=1)
    dists_topn = np.argsort(dists)[:n]
    
    ys = activations_y[dists_topn]
    ys = ['meme_pages/' + inverse_map[y].split('/')[-1][:-8] for y in ys]
    
    vals, counts = np.unique(ys, return_counts=True)
    counts_sorted = np.argsort(counts)[::-1]
    return vals[counts_sorted].tolist()[:3], counts[counts_sorted].tolist()[:3]


def get_neighbors_meta(nearest_n, counts):
    result = []
    
    data = meme_meta.loc[meme_meta[0].isin(nearest_n[:1])]
    result.append({'title': data[3].values[0],
                   'description': data[2].values[0],
                   'count': counts[0],
                   'link': data[5].values[0]})
    
    try:
        data = meme_meta.loc[meme_meta[0].isin(nearest_n[1:2])]
        result.append({'title': data[3].values[0],
                       'description': data[2].values[0],
                       'count': counts[1],
                       'link': data[5].values[0]})
    
        data = meme_meta.loc[meme_meta[0].isin(nearest_n[2:3])]
        result.append({'title': data[3].values[0],
                       'description': data[2].values[0],
                       'count': counts[2],
                       'link': data[5].values[0]})
    except:
        pass
    
    return result


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    resnet_model = ResNet50(weights='imagenet', include_top=False)

    tmp_model = Sequential()
    tmp_model.add(resnet_model)
    tmp_model.add(GlobalAveragePooling2D())

    model = tmp_model

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image

def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'='* (4 - missing_padding)
    return base64.decodestring(data)

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    print(flask.request)

    # ensure an image was properly uploaded to our endpoint

    print(flask.request.args)
    if flask.request.method == "POST":
        # read the image in PIL format
#        response = requests.get(flask.request.args["image"])
#        print(response.content)
        image_string = cStringIO.StringIO(decode_base64(flask.request.data))
        image = Image.open(image_string)

#            image = flask.request.files["image"].read()
#            image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        print(image)
        # classify the input image and then initialize the list
        # of predictions to return to the client

        nearest_n, counts = get_nearest_n(image, 9)
        result_json = get_neighbors_meta(nearest_n, counts)
        
        data["predictions"] = result_json

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
