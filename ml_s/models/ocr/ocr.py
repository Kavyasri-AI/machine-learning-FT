
import easyocr
from io import BytesIO
from time import time
import keras_ocr
import traceback
import os
from models.customlogger import mldebugger, log, INFO
import json

'''
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
import tensorflow.keras.backend as K
import os
'''
#cpuCount = os.cpu_count()
#print("Number of CPUs in the system:", cpuCount)
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#K.backend._get_available_gpus()

'''
tf.config.list_physical_devices(
    device_type=None
)


config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 40} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(
    sess
)
print(device_lib.list_local_devices())
'''


#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
#sess = tf.Session(config=config)
#print(sess)
#keras.backend.set_session(sess)
#print("==")
#reader_easyocr = easyocr.Reader(['en'],gpu = True)
#pipeline_kerasocr = keras_ocr.pipeline.Pipeline()

reader_easyocr = None
pipeline_kerasocr = None

dir_path = os.path.dirname(__file__)


def init_easyocr():
    global reader_easyocr
    reader_easyocr = easyocr.Reader(['en'],gpu = True)
    output = text_extract_easyocr(filename= [os.path.join(dir_path,"test.png")])
    mldebugger.simplelog("init_easyocr()", "init_easyocr has been successfuly initiated, " +  "easyocr_output: " + json.dumps(output), INFO)
    
def init_kerasocr():
    global pipeline_kerasocr
    pipeline_kerasocr = keras_ocr.pipeline.Pipeline()
    output = text_extract_keras(filename= [os.path.join(dir_path,"test.png")])
    mldebugger.simplelog("init_kerasocr()", "init_kerasocr has been successfuly initiated, " +  "keras_output: " + json.dumps(output), INFO)

    
def text_extract_easyocr(buffer ="" , filename = [], batch = 4, url = ""):
    global reader_easyocr
    '''
    @param: 
        filename: input data in array
        batch:  number of training examples utilized in one iteration.
    @returns: returns as json for extracted text data with other parameters like                                                                          data,algorithm,time,batch,filename
        @type:json
            @keys:
                data: returns extracted data from image file
                batch:  number of training examples utilized in one iteration.
                algorithm: type of algorithm is used
                time: time taken to extract the text from image from start to end.
                filename: path of the given input files
    output:
    [{'filename': '../ocr/dataset/1.png', 'data': ['Workpuls 7e898658-b.', 'KAVYA', 'PDF', 'komplete Machine learning_with sub topics:', 'Kaagaz_202, Maria Anudeep SNCTIO; SANCTIO;', 'This PC', 'hjkjkg', 'Programmimg Language:', 'LPython:', 'Anudeep', 'Recycle Bin', 'e4bQ80ac , MMaria', 'Data types Numhers', 'Machine_Leamningmodels', 'Google Chrome', 'Lhat i5-miachine learnine?', 'Natural language processing:', 'Text preprocesssing level 1: Tokenization Lemmatizaion Gonnino', 'Mozilla Firefox', 'helper fun,', 'Active Presenter', 'ENG 12:11 PM d) 8J E 10/22/2021', "8 ' 2", 'Type here to search'], 'time': 1.081294059753418, 'batch': 40, 'algorithm': 'easyocr'}]
    @input:
        filename = (file_path ="../ocr/dataset/1.png")
        batch = 4
    '''
    
    json_response = []
    if filename:
        for file in filename:
            try:
                start_time = time()
                results = reader_easyocr.readtext(file, detail=0, paragraph=True,batch_size = batch)
                end_time = time()-start_time
                json_ = {"filename":file,"data":results, "time": end_time, "batch": batch, "algorithm":"easyocr"}
                json_response.append(json_)
                
            except Exception as e:
#                 traceback.print_exc()
                log("text_extract_easyocr()", "text_extract_easyocr failed Something wrong happened at filename, look out!", e)
                json_ = {"filename":file,"data":"","algorithm" :"easyocr", "time": 0,"batch": batch, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
                json_response.append(json_)
    elif buffer:
        try:
            start_time = time()
            results = reader_easyocr.readtext(buffer, detail=0, paragraph=True,batch_size = batch)
            end_time = time()-start_time
            json_ = {"buffer":True, "data":results, "time": end_time, "batch": batch, "algorithm":"easyocr"}
            json_response.append(json_)
                
        except Exception as e:
#             traceback.print_exc()
            log("text_extract_easyocr()", "text_extract_easyocr failed Something wrong happened at buffer, look out!", e)
            json_ = {"buffer":True, "data":"","algorithm" :"easyocr", "time": 0,"batch": batch, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
            json_response.append(json_)
        
    return json_response


'''
extracted_output_ocr1 = text_extract_easyocr(url = data) #'time': 1.65657639503479, 'batch': 4,
print(extracted_output_ocr1)
extracted_output_ocr2 = text_extract_easyocr([filename],1) # 'time': 1.5429942607879639, 'batch': 1
print(extracted_output_ocr2)
extracted_output_ocr3 = text_extract_easyocr([filename],8) # 'time': 1.3525683879852295, 'batch': 8,
print(extracted_output_ocr3)
extracted_output_ocr4 = text_extract_easyocr([filename],16) # 'time': 1.1685380935668945, 'batch': 16
print(extracted_output_ocr4)
extracted_output_ocr5 = text_extract_easyocr([filename],32) # 'time': 1.1204941272735596, 'batch': 32
print(extracted_output_ocr5)
extracted_output_ocr6 = text_extract_easyocr([filename],64) # 'time': 1.1436688899993896, 'batch': 64
print(extracted_output_ocr6)
extracted_output_ocr7 = text_extract_easyocr([filename],40) # 'time': 1.070023536682129, 'batch': 40
print(extracted_output_ocr7)'''


def text_extract_keras(buffer="", filename = [], batch = 4):
    global pipeline_kerasocr
    '''
    @param: 
        filename: input data in array
        batch:  number of training examples utilized in one iteration.
    @returns: returns as json for extracted text data with other parameters like                                                                          data,algorithm,time,batch,filename
        @type:json
            @keys:
                data: returns extracted data from image file
                batch:  number of training examples utilized in one iteration.
                algorithm: type of algorithm is used
                time: time taken to extract the text from image from start to end.
                filename: path of the given input files
       output: 
            [{'filename': '../ocr/dataset/1.png', 'data': ['katya', 'workpuls', 'tebgofss', 'bu', 'wi', 'pdf', 'w', 'icomplete', 'machine', 'learning', 'with', 'sub', 'topics', 'this', 'pc', 'hikjkg', 'kaagaz', '202', 'maria', 'anudeep', 'sanctiom', 'sanctiom', 'programmimg', 'language', 'wi', 'lpython', 'recycle', 'bin', 'eaboblac', 'sl', 'anudeep', 'sl', 'mlaria', 'data', 'types', 'numhers', 'machine', 'learning', 'models', 'google', 'what', 'ismachine', 'learmincy', 'chrome', 'natural', 'language', 'processing', 'text', 'preprocesssing', 'level', '1', 'tokenization', 'mozilla', 'lemmatizaion', 'firefox', 'stemmima', 'helper', 'funu', 'active', 'presenter', 'pm', '1211', 'eng', 'ai', 'e', 'type', 'here', 'to', 'search', 'o', '10222021', 'in'], 'time': 2.0458085536956787, 'batch': 64, 'algorithm': 'keras_ocr'}]
        @input:
        filename = [file_path ="../ocr/dataset/1.png"]
        batch = 4
        
    '''

    json_response = []
    if filename:
        
        for file in filename:
            try:
                start_time = time()
                images = [
                    keras_ocr.tools.read(file)]
                prediction_groups = pipeline_kerasocr.recognize(images)
                words=[]
                for prediction in prediction_groups[0]:
                    words.append(prediction[0])
                end_time = time()-start_time
                json_ = {"filename":file, "data":words, "time": end_time, "batch": batch, "algorithm":"keras_ocr"}
                json_response.append(json_)
                
            except Exception as e:
                log("text_extract_keras()", "text_extract_keras failed Something wrong happened at filename, look out!", e)
#                 traceback.print_exc()
                json_ = {"filename":file,"data":"","algorithm" :"keras_ocr", "time": 0,"batch": batch, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
                json_response.append(json_)
                
                
    elif buffer:
        try:
            start_time = time()
            buffer = BytesIO(buffer)
            images = [
                keras_ocr.tools.read(buffer)]
            prediction_groups = pipeline_kerasocr.recognize(images)
            words=[]
            for prediction in prediction_groups[0]:
                words.append(prediction[0])
            end_time = time()-start_time
            json_ = {"buffer":True, "data":words, "time": end_time, "batch": batch, "algorithm":"keras_ocr"}
            json_response.append(json_)
            
        except Exception as e:
#             traceback.print_exc()
            log("text_extract_keras()", "text_extract_keras failed Something wrong happened at buffer, look out!", e)
            json_ = {"buffer":True,"data":"","algorithm" :"keras_ocr", "time": 0,"batch": batch, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
            json_response.append(json_)
            
    return json_response
    
'''extracted_output_kerasocr1 = text_extract_keras([filename]) # 'time': 20.093761920928955, 'batch': 4,
print(extracted_output_kerasocr1)
extracted_output_kerasocr2 = text_extract_keras([filename],1) # 'time': 2.138873338699341, 'batch': 1
print(extracted_output_kerasocr2)
extracted_output_kerasocr3 = text_extract_keras([filename],8) # 'time': 2.342315673828125, 'batch': 8,
print(extracted_output_kerasocr3)
extracted_output_kerasocr4 = text_extract_keras([filename],16) # 'time': 2.1101062297821045, 'batch': 16
print(extracted_output_kerasocr4)
extracted_output_kerasocr5 = text_extract_keras([filename],32) # 'time': 2.0975639820098877, 'batch': 32
print(extracted_output_kerasocr5)
extracted_output_kerasocr6 = text_extract_keras([filename],64) # 'time': 2.1827118396759033, 'batch': 64
print(extracted_output_kerasocr6)'''



'''def pull_data(url,fname):
    a = requests.get(url)
    with open("./data/{}.zip".format(fname),"wb") as f:
        f.write(a.content)'''

#pull_data(url = "https://storage.googleapis.com/traqez-ml-datasets/images/PART2.zip",fname = "data2")
# init_easyocr()
