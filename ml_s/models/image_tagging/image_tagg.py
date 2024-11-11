import numpy as np
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
from time import time
import os
from models.customlogger import mldebugger, log, INFO
import json

model_mobilenet = None
model_vgg16 = None
dir_path = os.path.dirname(__file__)
 
def init_mobilenet():
    global model_mobilenet
    model_mobilenet = tf.keras.applications.mobilenet.MobileNet()
    output = mobile_net(filename = [os.path.join(dir_path,"test.png")])
    mldebugger.simplelog("init_mobilenet()", "init_mobilenet has been successfuly initiated, " +  "mobilenet_output: " + json.dumps(output), INFO)

def init_vgg16():
    global model_vgg16
    model_vgg16 = VGG16()
    output = vgg_16(filename = [os.path.join(dir_path,"test.png")])
    mldebugger.simplelog("init_vgg16()", "init_vgg16 has been successfuly initiated, " +  "vgg_16_output: " + json.dumps(output), INFO)
    

def mobile_net(buffer = "",filename =[]):
    global model_mobilenet
    '''
    @param: 
        filename: input image
    @returns: returns as json for tagged words for image with other parameters like                                                                          tags,algorithm,time,size,filename
        @type:json
            @keys:
                tags: returns tagged words from image file
                algorithm: type of algorithm is used
                time: time taken to tag the words for image from start to end.
                size: image size like (224,224)
                filename:apth of the input image file
    output:
    [{'filename': '../image_tagging/1.png', 'algorithm': 'mobilenet', 'tags': [{'word': 'screen', 'score': 0.3474924}, {'word': 'envelope', 'score': 0.31000337}, {'word': 'monitor', 'score': 0.076097146}, {'word': 'web_site', 'score': 0.06085838}, {'word': 'oscilloscope', 'score': 0.045444794}], 'time': 1.7286713123321533, 'size': (224, 224)}]
    @input:
        filename = ([filename ="../ocr/dataset/1.png"])
    '''
    json_response = []
    if filename:
        
        for file in filename:
            try:
                start_time = time()
                image = load_img(file,target_size = (224,224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                predictions = model_mobilenet.predict(image)
                results = imagenet_utils.decode_predictions(predictions)
                values1 = []
                for a_tuple in results:
                    for q in a_tuple:
                        j ={}
                        j["word"] = q[1]
                        j["score"] = float(q[2])
                        values1.append(j)
                end_time = time()-start_time
                json_ = {"filename":file,"algorithm": "mobilenet", "tags": values1,"time": end_time, "size": (224,224)}
                json_response.append(json_)
            except Exception as e:
                log("mobile_net()", "mobile_net failed Something wrong happened at filename, look out!", e)
                json_ = {"filename":file,"tags":"","algorithm" :"mobilenet", "time": 0,"size": (224,224),"error":True,"err_msg":"error_function: {},error: {}".format(e.__class__.__name__,str(e))}
                json_response.append(json_)
                
    elif buffer:
        
        try:
            start_time = time()
            buffer = Image.open(BytesIO(buffer))
            image = buffer.resize((224,224)).convert("RGB")
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            predictions = model_mobilenet.predict(image)
            results = imagenet_utils.decode_predictions(predictions)
            values1 = []
            for a_tuple in results:
                for q in a_tuple:
                    j ={}
                    j["word"] = q[1]
                    j["score"] = float(q[2])
                    values1.append(j)
            end_time = time()-start_time
            json_ = {"buffer": True,"algorithm": "mobilenet", "tags": values1,"time": end_time, "size": (224,224)}
            json_response.append(json_)
            
        except Exception as e:
                    
            log("mobile_net()", "mobile_net failed Something wrong happened at buffer, look out!", e)
            
            json_ = {"buffer":True,"tags":"","algorithm" :"mobilenet", "time": 0,"size": (224,224),"error":True,"err_msg":"error_function: {},error: {}".format(e.__class__.__name__,str(e))}
            json_response.append(json_)
        
    return json_response

#mobile_output = mobile_net(filename =["../image_tagging/1.png"])
#print(mobile_output) # 'time': 1.7037060260772705, 'size': (224, 224)
                     # 'time': 1.8472201824188232, with gpu memory limit

def vgg_16(buffer = "", filename =[]):
    global model_vgg16
    '''
        @param: 
        filename: input image
    @returns: returns as json for tagged words for image with other parameters like                                                                          tags,algorithm,time,size
        @type:json
            @keys:
                tags: returns tagged words from image file
                algorithm: type of algorithm is used
                time: time taken to tag the words for image from start to end.
                size: image size like (224,224)
                filename:apth of the input image file
    output:
    [{'filename': '../image_tagging/1.png', 'algorithm': 'vgg16', 'tags': [{'word': 'scoreboard', 'score': 0.99671644}, {'word': 'digital_clock', 'score': 0.0015653125}, {'word': 'screen', 'score': 0.00035412094}, {'word': 'monitor', 'score': 0.00035324937}, {'word': 'traffic_light', 'score': 0.00025730362}], 'time': 1.1110994815826416, 'size': (224, 224)}]
    @input:
        filename = ([image_path ="../ocr/dataset/1.png"])
    '''
    json_response = []
    
    if filename:
        
        for file in filename:
            try:
                start_time = time()
                image = load_img(file,target_size = (224,224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                predictions = model_vgg16.predict(image)
                results = decode_predictions(predictions)
                values1 = []
                for a_tuple in results:
                    for q in a_tuple:
                        j ={}
                        j["word"] = q[1]
                        j["score"] = float(q[2])
                        values1.append(j)
                end_time = time()-start_time
                json_ = {"filename":file,"algorithm": "vgg16", "tags": values1,"time": end_time, "size": (224,224)}
                json_response.append(json_)
            except Exception as e:
                log("vgg_16()", "vgg_16 failed Something wrong happened at filename, look out!", e)
                json_ = {"filename":file,"tags":"","algorithm" :"mobilenet", "time": 0,"size": (224,224),"error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
                json_response.append(json_)
            
    elif buffer:
        
        try:
            start_time = time()
            buffer = Image.open(BytesIO(buffer))
            image = buffer.resize((224,224)).convert("RGB")
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            predictions = model_vgg16.predict(image)
            results = decode_predictions(predictions)
            values1 = []
            for a_tuple in results:
                for q in a_tuple:
                    j ={}
                    j["word"] = q[1]
                    j["score"] = float(q[2])
                    values1.append(j)
            end_time = time()-start_time
            json_ = {"buffer":True,"algorithm": "vgg16", "tags": values1,"time": end_time, "size": (224,224)}
            json_response.append(json_)
            
        except Exception as e:
            log("vgg_16()", "vgg_16 failed Something wrong happened at buffer, look out!", e)
            
            json_ = {"buffer":True,"tags":"","algorithm" :"mobilenet", "time": 0,"size": (224,224),"error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
            json_response.append(json_)
        
    return json_response

#vgg16_output = vgg_16(filename =["../image_tagging/1.png"])
#print(vgg16_output) # 'time': 1.0657768249511719, 'size': (224, 224)
                    # 'time': 1.1189661026000977, with gpu memory limit