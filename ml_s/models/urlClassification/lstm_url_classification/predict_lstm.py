import os
import json
from time import time
import numpy as np
import traceback
from tensorflow.keras.models import load_model
from string import printable
from tensorflow.keras.preprocessing import sequence
from models.customlogger import mldebugger, INFO, log


model = None
def init_lstm_urlClassification():
    global model
    dir_path = os.path.dirname(__file__)
    model = load_model(os.path.join(dir_path,"models/simple_lstm.h5"))
    output = lstm_url_prediction(urls = ['titaniumcorporate.co.za','en.wikipedia.org/wiki/North_Dakota'])
    mldebugger.simplelog("init_urlClassification()", "init_urlClassification has been successfuly initiated, " +  "url_classifiaction_output: " + json.dumps(output), INFO)
    
    
def lstm_url_prediction(urls= []):
    global model
    '''
    @param: 
        urls: input urls
    @returns: returns as json for detected malicious urls for urls with other parameters like                                                                          tags,algorithm,time,size,filename
        @type:json
            @keys:
                url: returns tagged words from image file
                time: time taken to detect the malicious urls for given input urls from start to end.
                output: detected malicious urls by lstm model
    @output:            
    url_classifiaction_output: [{'url': 'titaniumcorporate.co.za', 'output': ['malicious'], 'time': 6.522950649261475},
    {'url': 'en.wikipedia.org/wiki/North_Dakota', 'output': ['safe'], 'time': 6.569924354553223}]
    
    @input:
        urls = ['titaniumcorporate.co.za','en.wikipedia.org/wiki/North_Dakota']
    '''
    json_response = []
    try:
        start_time = time()
        for url in urls:
            url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
            X = sequence.pad_sequences(url_int_tokens, maxlen=75)
            p = model.predict(X, batch_size=1)
            pred = []
            if p < 0.5:
                res = "safe"
                pred.append(res)
            else:
                res = "malicious"
                pred.append(res)
            end_time = time()-start_time
            json_ = {"url": url, "output":pred, "time": end_time}
            json_response.append(json_)
    except Exception as e:
        # traceback.print_exc()
        log("lstm_url_prediction()", "url_prediction failed Something wrong happened, look out!", e)
        json_ = {"url": url, "output":"", "time": 0,"error":True,"err_msg":"error_function: {},error: {}".format(e.__class__.__name__,str(e))}
        json_response.append(json_)

    return json_response

# init_lstm_urlClassification()
# lstm_url_prediction(urls = ['titaniumcorporate.co.za','en.wikipedia.org/wiki/North_Dakota'])