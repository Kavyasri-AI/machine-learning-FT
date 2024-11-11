import torch
import os
import numpy as np
import traceback
import json
from models.customlogger import mldebugger, INFO, log
from time import time

from transformers import BertTokenizer


phish_model = None
tokenizer = None
device = None
dir_path = os.path.dirname(__file__)
def init_bert_phish():
    global phish_model, tokenizer, device
    f = open(os.path.join(dir_path,"test.txt"), 'r')
    file = f.read()
    phish_model = torch.load(os.path.join(dir_path,'phishing_model'), map_location=torch.device('cuda'))
    tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = bert_phish_email([file])
    mldebugger.simplelog("init_bert_phish()", "init_bert_phish has been successfuly initiated, " +  "init_phish_output: " + json.dumps(output), INFO)

def bert_phish_email(emails= [], email= ""):
    '''
    @param: 
        data: subject of emails in list
    @returns: returns as json for detected phishing emails for emails with other parameters like                                                                          algorithm,time,email and output
        @type:json
            @keys:
                email: returns given email from emails list
                time: time taken to detect the phishing emails for given input emails from start to end.
                output: detected phishing emails 
    @output:            
    [{'email': 'Subject: Credit Unions Banks Update Account Info Verification Dear FCU client,As part of our security measures, we regularly screen activity in Federal Credit Unions (FCU) network We recently noticed the following issue on your account: A recent review of your transaction history determined that we require some additional information from you in order to provide you with secure service Case ID Number: 
    PP-065-617-349 For your protection, we have limited your access, until additional security measures can be completed We apologize for any inconvenience this may cause Please log and restore your access as soon as possible You must click the link below and fill in the form on the following page to complete the verification process Click here to update your accountPlease do not reply to this e-mail Mail sent to this address cannot be answered', 
    'output': 'Ham', 'algorithm': 'bert_email_phishing', 'time': 1.3875818252563477}]
    
    @input:
        data = [file]
    '''
    global phish_model, tokenizer, device
    try:
        if email:
            start_time = time()
            encoding = tokenizer.encode_plus(
                                email,
                                add_special_tokens = True,
                                max_length = 32,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt'
                            )

            # We need Token IDs and Attention Mask for inference on the new sentence
            test_ids = []
            test_attention_mask = []

            # Apply the tokenizer

            # Extract IDs and Attention Mask
            test_ids.append(encoding['input_ids'])
            test_attention_mask.append(encoding['attention_mask'])
            test_ids = torch.cat(test_ids, dim = 0)
            test_attention_mask = torch.cat(test_attention_mask, dim = 0)

            # Forward pass, calculate logit predictions
            with torch.no_grad():
                output = phish_model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

            prediction = 'Spam' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Ham'

            # print('Input Sentence: ', input_text)
            # print('Predicted Class: ', prediction)
#             email = input.replace("\n", "")
            end_time = time()-start_time
            return_json = {"email":email, "output":prediction, "algorithm": "bert_email_phishing", "time" : end_time}
            return return_json
        
        else :
            
            json_response = []
            start_time = time()
            for em in emails:
                '''
                Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
                - input_ids: list of token ids
                - token_type_ids: list of token type ids
                - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
                '''
                encoding = tokenizer.encode_plus(
                                    em,
                                    add_special_tokens = True,
                                    max_length = 32,
                                    pad_to_max_length = True,
                                    return_attention_mask = True,
                                    return_tensors = 'pt'
                                )

                # We need Token IDs and Attention Mask for inference on the new sentence
                test_ids = []
                test_attention_mask = []

                # Apply the tokenizer

                # Extract IDs and Attention Mask
                test_ids.append(encoding['input_ids'])
                test_attention_mask.append(encoding['attention_mask'])
                test_ids = torch.cat(test_ids, dim = 0)
                test_attention_mask = torch.cat(test_attention_mask, dim = 0)

                # Forward pass, calculate logit predictions
                with torch.no_grad():
                    output = phish_model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

                prediction = 'Spam' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Ham'

                # print('Input Sentence: ', input_text)
                # print('Predicted Class: ', prediction)
    #             email = input.replace("\n", "")
                end_time = time()-start_time
                json_ = {"email":em, "output":prediction, "algorithm": "bert_email_phishing", "time" : end_time}
                json_response.append(json_)
                
            return json_response
        
    except Exception as e:
#         traceback.print_exc()
        log("bert_phish_email()", "bert_phish_email failed Something wrong happened, look out!", e)
        json_ = {"output":"", "time": 0,"error":True,"err_msg":"error_function: {},error: {}".format(e.__class__.__name__,str(e))}
        json_response.append(json_)
        
    return json_response

# f = open(os.path.join(dir_path,"Email-test.json"), 'r')
# file = json.load(f)
# data = [i['Text'] for i in file if 'Text' in i]

# init_bert_phish()
# output = bert_phish_email(emails = data)
# print(output)


