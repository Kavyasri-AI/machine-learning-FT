import pytesseract
import os
import io
from time import time
from PIL import Image
import traceback
from models.customlogger import mldebugger, INFO, log
import json


dir_path = os.path.dirname(__file__)


reader_tesseract= None
tessdata_dir_config = None

def init_tesseract():
    """
        OCR options:
      --tessdata-dir PATH   Specify the location of tessdata path.
      --user-words PATH     Specify the location of user words file.
      --user-patterns PATH  Specify the location of user patterns file.
      -l LANG[+LANG]        Specify language(s) used for OCR.
      -c VAR=VALUE          Set value for config variables.
                            Multiple -c arguments are allowed.
      --psm NUM             Specify page segmentation mode.
      --oem NUM             Specify OCR Engine mode.
    NOTE: These options must occur before any configfile.
    Page segmentation modes:
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line,
           bypassing hacks that are Tesseract-specific.
    OCR Engine modes: (see https://github.com/tesseract-ocr/tesseract/wiki#linux)
      0    Legacy engine only.
      1    Neural nets LSTM engine only.
      2    Legacy + LSTM engines.
      3    Default, based on what is available.
  """
    global reader_tesseract
    global tessdata_dir_config
    pytesseract.pytesseract.tesseract_cmd = os.path.join(dir_path,"bin/tesseract/tesseract")
    tessdata_dir_config = r'--oem 1 --psm 6'
    #./tesseract ../../test.png output --oem 1 -l eng --tessdata-dir ./tessdata/
    reader_tesseract = pytesseract
    output = text_extract_tesseract(filename= [os.path.join(dir_path,"test.png")])
    mldebugger.simplelog("init_tesseract()", "init_tesseract has been successfuly initiated, " +  "tesseract_output: " + json.dumps(output), INFO)
    
def text_extract_tesseract(buffer ="" , filename = []):
    global reader_tesseract
    global tessdata_dir_config
    '''
    @param: 
        filename: input data in array
    @returns: returns as json for extracted text data with other parameters like                                                                          data,algorithm,time,batch,filename
        @type:json
            @keys:
                data: returns extracted data from image file
                algorithm: type of algorithm is used
                time: time taken to extract the text from image from start to end.
                filename: path of the given input files
    output:
    [{'filename': 'C:\\Users\\kavya\\test\\tess_eract\\test.png', 'data': ['kaye', 'ns', 'ga'
    , 'Workpuls', '7eBORFS8-b,', '3S8E', 'This PC', 'Recycle Bin', 'Google', 'Chrome', '©', 'Mozilla', 
    'Firefox', 'helper fun', 'Active', 'Presenter', 'hikikg,', 'Keagaz. 202', 'edb 080ac-', 'i]', 'Maria -', 
    'SANCTIO,', 'i]', 'Maria - SL', '=', 'Anudeep -', 'SANCTIO.', 'i]', 'Anudeep - SL', '&', '+', '{complete 
    Machine learning with', 'sub topi', '* Tokenizatio', '* Lemmati', '© Stammima', 'Ae', 'Qn', 'jon', '2', 
    ENG', 'IN', '12:11PM', '10/22/2021'], 'time': 0.7072386741638184, 'algorithm': 'tesseract'}]
    @input:
        filename = [os.path.join(dir_path,"test.png")]
    '''
    
    json_response = []
    if filename:
        for file in filename:
            try:
                start_time = time()
                res_ = reader_tesseract.image_to_string(file, config = tessdata_dir_config)
                res = res_.split("\n")
                # res_= ' '.join(res).split()
                # results= [x for x in res if x]
                results =list(filter(lambda item: item.strip(), res))
                end_time = time()-start_time
                json_ = {"filename":file,"data":results, "time": end_time, "algorithm":"tesseract"}
                json_response.append(json_)
                
            except Exception as e:
#                 traceback.print_exc()
                log("text_extract_tesseract()", "text_extract_tesseract failed Something wrong happened at filename, look out!", e)

                json_ = {"filename":file,"data":"","algorithm" :"tesseract", "time": 0,"error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
                json_response.append(json_)
    elif buffer:
        try:
            start_time = time()
            io_bytes = io.BytesIO(buffer)
            res_ = reader_tesseract.image_to_string(Image.open(io_bytes), config = tessdata_dir_config)
            res = res_.split("\n")
            results =list(filter(lambda item: item.strip(), res))
            end_time = time()-start_time
            json_ = {"buffer":True, "data":results, "time": end_time, "algorithm":"tesseract"}
            json_response.append(json_)
                
        except Exception as e:
            
#             traceback.print_exc()
            log("text_extract_tesseract()", "text_extract_tesseract failed Something wrong happened at buffer, look out!", e)

            json_ = {"buffer":True, "data":"","algorithm" :"tesseract", "time": 0, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
            json_response.append(json_)
        
    return json_response

# init_tesseract()