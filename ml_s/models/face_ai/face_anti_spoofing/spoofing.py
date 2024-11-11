import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from .Model import DeePixBiS
from .Loss import PixWiseBCELoss
from .Metrics import predict, test_accuracy, test_loss
from time import time
import os
from models.customlogger import mldebugger,INFO,log
import json

dir_path = os.path.dirname(__file__)
model_antispoofing = None

def init_face_anti_spoofing():
    global model_antispoofing
    model_antispoofing = DeePixBiS()
    model_antispoofing.load_state_dict(torch.load(os.path.join(dir_path,"DeePixBiS.pth")))
    model_antispoofing.eval()
    output = face_anti_spoofing(os.path.join(dir_path,"test1.jpg"))
    mldebugger.simplelog("init_face_anti_spoofing()", "init_face_anti_spoofing has been successfuly initiated, " +  "face_anti_spoofing_output: " + json.dumps(output), INFO)

def face_anti_spoofing(filename = "",buffer = ""):
    global model_antispoofing
    try:
        if model_antispoofing is None:
            return  { 
                "error":True, 
                "err_msg":"Anti Spoofing is disabled or not running, try again",
                "load": False,  # indicates model load failed
                "match": None,
            }
        
        start_time = time()
        tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])

        faceClassifier = cv.CascadeClassifier(os.path.join(dir_path,"classifier/haarface.xml"))

        if buffer:
            nparr_img = np.fromstring(buffer, np.uint8)
            image_input = cv.imdecode(nparr_img, cv.COLOR_RGBA2RGB)
        if filename:
            image_input = cv.imread(filename)

        grey = cv.cvtColor(image_input, cv.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

        for x, y, w, h in faces:
            faceRegion = image_input[y:y + h, x:x + w]
            faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
            # cv.imshow('Test', faceRegion)
            faceRegion = tfms(faceRegion)
            faceRegion = faceRegion.unsqueeze(0)
            mask, binary = model_antispoofing.forward(faceRegion)
            res = torch.mean(mask).item()
            end_time = time()-start_time
            # res = binary.item()
            # print(res)
            cv.rectangle(image_input, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if res < 0.7:
                cv.putText(image_input, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                json_response = {"buffer": True, "score": res , "time": end_time, "msg":"Face did not Match", "match": True}
                return json_response
            else:
                cv.putText(image_input, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                json_response = {"buffer": True, "score": res, "time": end_time, "msg": "Face Matched", "match": False }
                return json_response
            
    except Exception as e:
        log("face_anti_spoofing()", "face_anti_spoofing failed Something wrong happened, look out!", e)
        json_response = {"match": None, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
    
    return json_response

# init_face_anti_spoofing()
 