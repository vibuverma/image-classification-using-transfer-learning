import numpy as np 
import sys
import os
import re
import glob


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask Util
from flask import Flask, redirect, render_template, url_for, request

#Define app name
app= Flask(__name__)
model_path= 'vgg19.h5'

### Loading Model
model= load_model(model_path)
model._make_predict_function()


### Preprocessing 
def model_predict(img_path, model):
    img= image.load_img(img_path, target_size=(224,224))
    x= image.img_to_array(img)
    x= np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    preds= model.predict(x)
    return preds

@app.route('/', method=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', method=['GET', 'POST'])
def upload():
    if request.method== 'POST':
        ## Get the file from host
        f= request.files['file']
        ## Saving the file to uploads
        basepath= os.path.dirname(__file__)
        file_path= os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        ## Making prediction
        pred= model_predict(file_path, model)
        
        ## Decoding predcitions
        pred_class= decode_predictions(pred, top=1)
        result= str(pred_class[0][0][1])
        return result
    return None




if __name__ == 'main':
    app.run(debug=True)