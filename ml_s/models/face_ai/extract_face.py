#import deepface
import traceback
import base64
import os
from deepface import DeepFace
from time import time
from .face_anti_spoofing import face_anti_spoofing  #newly added, sept 20th.
from models.customlogger import mldebugger,INFO,log
import json
import numpy as np

dir_path = os.path.dirname(__file__)


def init_face_recog():
    face_recog = face_model(filename = [os.path.join(dir_path,"test1.jpg"), os.path.join(dir_path,"test2.jpg")], model = "Facenet512", backend ="retinaface")
    face_vect = face_vectors(filename = os.path.join(dir_path,"test2.jpg"), model = "Facenet512", backend ="retinaface")
    face_distance = face_distance_matric(filename = [os.path.join(dir_path,"test1.jpg"), os.path.join(dir_path,"test2.jpg")], distance_metric = "cosine", model_ = "Facenet512", backend_ ="retinaface")
#     print(face_distance)
    mldebugger.simplelog("init_face_recog()", "init_face_recog has been successfuly initiated, " +  "face_recognition_output: " + json.dumps(face_recog), INFO)
#     mldebugger.simplelog("init_face_recog()", "init_face_recog has been successfuly initiated, " + "face_vector_output" + json.dumps(face_vect), INFO)
    mldebugger.simplelog("init_face_recog()", "init_face_recog has been successfuly initiated, " + "face_distance_output" + json.dumps(face_distance), INFO)


def face_model(filename = [], buffer = [], model = "Facenet512", backend ="retinaface"):
    
    '''
    @param: 
        filename: input images in array
        model: type of model is used
        backend: type of backend model is used
    @returns: returns as json for extracted text data with other parameters like                                                                          data,algorithm,time,model,backend,filename
        @type:json
            @keys:
                data: returns verified images true or false, distance between them and similarity metrics of image files
                algorithm: type of algorithm is used
                time: time taken to verify the face for the two image files from start to end.
                model: type of model is used
                backend: type of backend model is used
                filename: path names of input files
    output:
    {'data': {'verified': True, 'distance': 0.1529802435438289, 'max_threshold_to_verify': 0.3, 'model': 'Facenet512','similarity_metric': 'cosine'}, 'time': 16.35499119758606, 'algorithm': 'deepface', 'model': 'Facenet512', 'backend':'retinaface', 'filename': ['../face_ai/people/img1.jpg', '../face_ai/dataset/img11.jpg']}
    @input:
        filename = (["../face_recongnition/people/img1.jpg","../face_recongnition/dataset/img12.jpg"])
        model = "Facenet512"
        backend = 'retinaface'
    '''
    start_time = time()
    try:
        if filename:    
            df1 = DeepFace.verify(img1_path = filename[0], img2_path = filename[1],detector_backend = backend, model_name= model)
            spoofing_1 = face_anti_spoofing(filename = filename[0])
            spoofing_2 = face_anti_spoofing(filename = filename[1])
        
        if buffer:
            df1 = DeepFace.verify(img1_path = buffer[0], img2_path = buffer[1],detector_backend = backend, model_name= model)
            image_1 = buffer[0][22:]
            image_2 = buffer[1][22:]
            img_1 = base64.b64decode(image_1)
            img_2 = base64.b64decode(image_2)

            spoofing_1 = face_anti_spoofing(buffer = img_1)
            spoofing_2 = face_anti_spoofing(buffer = img_2)

        spoof = []
        if spoofing_1["match"] is not None and spoofing_2["match"] is not None:
            if spoofing_1['match'] == spoofing_2['match']:
                spoof_1 = {"spoofing": [spoofing_1,spoofing_2],"fake": False}
                spoof.append(spoof_1)
            else:
                spoof_2 = {"spoofing": [spoofing_1,spoofing_2],"fake": True}
                spoof.append(spoof_2)
    
        end_time = time()-start_time
        
        json_response = {"buffer": True,"data":df1, "time": end_time,"algorithm":"deepface","model":model,"backend":backend, "spoofing": spoof}
    
    except Exception as e:
  
        log("face_recog()", "face recog failed Something wrong happened, look out!", e)
        
        json_response = {"buffer": True,"data":"","algorithm" :"deepface", "time": 0,"model":model,"backend":backend,"spoofing": spoof, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
        
    return json_response

#face_verify1 = face_model(filename = ["../face_ai/people/img1.jpg","../face_ai/dataset/img11.jpg"], model = "Facenet512", backend = 'retinaface')
#face_verify2 = face_model(filename = ["./face_ai/people/img1.jpg","./face_ai/dataset/img12.jpg"], model = "ArcFace", #backend = 'retinaface')
#face_verify3 = face_model(filename = ["../face_recongnition/people/img1.jpg","../face_recongnition/dataset/img12.jpg"], model = "Facenet512", backend = 'ssd')


#print(face_verify1)# 'time': 16.427035570144653, 'algorithm': 'deepface', 'Model': 'Facenet512', 'Backend': 'retinaface'
#print(face_verify2)# 'time': 4.072130441665649, 'algorithm': 'deepface', 'Model': 'ArcFace', 'Backend': 'retinaface'
#print(face_verify3)# 'time': 0.555699348449707, 'algorithm': 'deepface', 'Model': 'Facenet512', 'Backend': 'ssd'


def face_vectors(filename = "",buffer = "", model = "Facenet512", backend ="retinaface"):
    '''
    @param: 
        filename: input image
        model: type of model is used
        backend: type of backend model is used
    @returns: returns as json for vector embeddings for image file with other parameters like                                                                          data,algorithm,time,model,backend,filename
        @type:json
            @keys:
                data: returns embeddings for image file
                algorithm: type of algorithm is used
                time: time taken to generate vector embeddings for the image file from start to end.
                model: type of model is used
                backend: type of backend model is used
                filename: path name of input file
    output:
    {'vectors': [-0.731559157371521, 0.5348088145256042, -0.9739866852760315, -0.5433751940727234, 0.8540410399436951, -0.6439685225486755, 1.8733546733856201, 1.654706597328186, 0.4379722774028778, -0.6392167210578918, 0.18695513904094696, 0.362013578414917, 0.6799625158309937, 0.598552942276001, -0.4336596727371216, -0.2631928026676178, -0.31832265853881836], 'count': 512, 'time': 0.5286829471588135, 'filename': '../face_ai/people/img1.jpg', 'algorithm': 'deepface', 'model': 'Facenet512', 'backend': 'retinaface', 'size': 10564}
    @input:
        filename = ("../face_recongnition/people/img1.jpg")
        model = "Facenet512"
        backend = 'retinaface'
    '''
    if filename:
        start_time = time()
        embedding1 = DeepFace.represent(img_path = filename, detector_backend = backend, model_name= model)
        spoofing = face_anti_spoofing(filename = filename)
        end_time = time()-start_time
    if buffer:
        start_time = time()
        embedding1 = DeepFace.represent(img_path = buffer, detector_backend = backend, model_name= model)
        image = buffer[22:]
        img = base64.b64decode(image)
        spoofing = face_anti_spoofing(buffer = img)
        end_time = time()-start_time
    try:
        spoof = []
        if spoofing["match"] is not None:
            if spoofing['match'] == False:
                spoof_1 = {"spoofing": spoofing,"fake": False}
                spoof.append(spoof_1)
            else:
                spoof_2 = {"spoofing": spoofing,"fake": True}
                spoof.append(spoof_2)
            
        json_response= {"vectors":embedding1, "count": len(embedding1),"time": end_time,"buffer": True,"algorithm":"deepface","model":model,"backend":backend,"size":len(str(embedding1)), "spoofing": spoof}
        
    except Exception as e:
#         traceback.print_exc()
        log("face_vectors()", "face_vectors failed Something wrong happened, look out!", e)
        json_response = {"vectors":"","algorithm" :"deepface", "time": 0,"model":model,"backend":backend,"buffer": True, "spoofing": spoof, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
    
    return json_response

#face_embeddings1 = face_vectors(filename = "../face_ai/people/img1.jpg", model = "Facenet512", backend = 'retinaface')
#face_embeddings2 = face_vectors(filename = "../face_recongnition/people/img1.jpg", model = "ArcFace", backend = 'retinaface')
#face_embeddings3 = face_vectors(filename = "../face_recongnition/people/img1.jpg", model = "Facenet512", backend = 'ssd')


#print(face_embeddings1)# 'time': 14.48285174369812, 'algorithm': 'deepface', 'Model': 'Facenet512', 'Backend': 'retinaface', 'size': 10571
#print(face_embeddings2) # 'time': 11.632098197937012, 'algorithm': 'deepface', 'Model': 'ArcFace', 'Backend': 'retinaface', 'size': 10970
#print(face_embeddings3) # 'time': 7.854784965515137, 'algorithm': 'deepface', 'Model': 'Facenet512', 'Backend': 'ssd', 'size': 10586


def face_distance_matric(filename = [],buffer = [], distance_metric = "", model_ = "Facenet512", backend_ ="retinaface",):
    '''
    @param: 
        filename: input image
        model: type of model is used
        backend: type of backend model is used
    @returns: returns as json for vector embeddings for image file with other parameters like                                                                          data,algorithm,time,model,backend,filename
        @type:json
            @keys:
                data: returns embeddings for image file
                algorithm: type of algorithm is used
                time: time taken to generate vector embeddings for the image file from start to end.
                model: type of model is used
                backend: type of backend model is used
                filename: path name of input file
    output:
    {'vectors': [-0.731559157371521, 0.5348088145256042, -0.9739866852760315, -0.5433751940727234, 0.8540410399436951, -0.6439685225486755, 1.8733546733856201, 1.654706597328186, 0.4379722774028778, -0.6392167210578918, 0.18695513904094696, 0.362013578414917, 0.6799625158309937, 0.598552942276001, -0.4336596727371216, -0.2631928026676178, -0.31832265853881836], 'count': 512, 'time': 0.5286829471588135, 'filename': '../face_ai/people/img1.jpg', 'algorithm': 'deepface', 'model': 'Facenet512', 'backend': 'retinaface', 'size': 10564}
    @input:
        filename = ("../face_recongnition/people/img1.jpg")
        model = "Facenet512"
        backend = 'retinaface'
        
    '''    
    def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(source_representation, test_representation):
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


    def l2_normalize(x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))


    if filename:
        start_time = time()
        embedding1 = face_vectors(filename = filename[0], model = model_ , backend = backend_)
        embedding2 = face_vectors(filename = filename[1], model = model_ , backend = backend_)
        end_time = time()-start_time
        
    if buffer:
        start_time = time()
        embedding1 = face_vectors(filename = buffer[0], model = model_ , backend = backend_)
        embedding2 = face_vectors(filename = buffer[1], model = model_ , backend = backend_)
        end_time = time()-start_time
    try:
        
        if distance_metric == 'cosine':
            distance = findCosineDistance(embedding1["vectors"], embedding2["vectors"])
        elif distance_metric == 'euclidean':
            distance = findEuclideanDistance(embedding1["vectors"], embedding2["vectors"])
        elif distance_metric == 'euclidean_l2':
            distance = findEuclideanDistance(l2_normalize(embedding1["vectors"]), l2_normalize(embedding2["vectors"]))
        else:
            raise ValueError("Invalid distance_metric passed - ", distance_metric)

        distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)

        json_response= {"distance": distance,"similarity_metric": distance_metric,"time": end_time,"buffer": True,"algorithm":"deepface","model":model_,"backend":backend_}
        
    except Exception as e:
        traceback.print_exc()
        log("face_distance_matric()", "face_distance_matric failed Something wrong happened, look out!", e)
        json_response = {"distance":"","similarity_metric": distance_metric,"algorithm" :"deepface", "time": 0,"model":model_,"backend":backend_,"buffer": True,"error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
    
    return json_response


