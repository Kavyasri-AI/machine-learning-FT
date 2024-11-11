from .guesslang import Guess
import time
import os 
from models.customlogger import mldebugger,log,INFO
import json

guess = None

dir_path = os.path.dirname(__file__)
filehandle = open(os.path.join(dir_path,"test.txt"))
data = filehandle.read()


def init_guess_lang():
    global guess
    guess = Guess()
    output = guess_language(snippet_string= data)
    mldebugger.simplelog("init_guess_lang()", "init_guess_lang has been successfuly initiated, " +  "guess_lang_output: " + json.dumps(output), INFO)

languages = ["C","C++", "Rust", "Go", "Python", "Kotlin", "Java", "Erlang", "Haskell", "PHP", "C#", "Swift"]


def guess_language(snippet_string="", snippet_array=[]):
    '''
    @param: 
        snippet_string: input data in string format.
        snippet_array:  snippet string data in array.
    @returns: returns as json for extracted text data with other parameters like                                                                          data,algorithm,time,batch,filename
        @type:json
            @keys:
                code: returns given snippet_string data
                language: Type of Language
                number of lines: number of line sin snippet string
                total_time: time taken to detect the languages from given snippet_string from start to end.

    output:
    
    [{'code': 'class COCO:\n    def __init__(self, split_data=None):\n        
    """\n        :param annotations (dict): annotation dictionary with train, test and val data\n        :param im   age_folder (str): location to the folder that hosts images.\n        :return:\n        """\n        self.dataset = {}  # this is basically split_data[\'train\'] having images and annotation lists\n        self.anns = {}  # a dictionary with key as id(caption) and value as annotation dict\n        self.imgToAnns = {} # a dictionary with key as image_id and value as list of annotation dicts\n        self.imgs = {}  # a dictionary with key as image_id and value as image dict\n\n        if split_data != None:\n            print(\'loading annotations into memory...\')\n            time_t = datetime.datetime.utcnow()\n            dataset = split_data[\'train\']  # dataset is dictionary with "images" and "annotations" keys\n            print(datetime.datetime.utcnow() - time_t)\n            self.dataset = dataset\n            self.createIndex()\n\n    def createIndex(s',
    'language': 'Python',
    'number of lines': 250,
    'total_time': 0.12293791770935059},
    {'code': ' Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics \n    with text pointers inside Git, while storing the file contents\n     on a remote server like GitHub.com or GitHub Enterprise. ',
    'language': 'some english text',
    'number of lines': 3,
    'total_time': 0.002619028091430664}]
    
    @input:
        guess_language(snippet_array = code_input)
    '''

    global guess
    try:
        if snippet_string:

            length = len(snippet_string.split('\n'))

            start_time = time.time()
            test_code = guess.language_name(snippet_string)
            total_time = time.time() - start_time

            if len(snippet_string.encode('utf-16-le')) >= 1024:
                snippet_string = snippet_string[0:1024]

            if test_code in languages:
                return_json = {"code":snippet_string, "buffer": True,  "language": test_code,"number of lines":length, "total_time": total_time} 

            else:
                return_json = {"code":snippet_string, "buffer": True, "language": "some english text", "number of lines":length, "total_time": total_time}

            return return_json


        else:

            json_array = []
            start0 = time.time()
            for snippet in snippet_array:

                length = len(snippet.split('\n'))

                start_time = time.time()
                test_code = guess.language_name(snippet)
                total_time = time.time() - start_time

                if len(snippet.encode('utf-16-le')) >= 1024:
                    snippet = snippet[0:1024]

                if test_code in languages:
                    return_json = {"code":snippet, "buffer": True, "language": test_code,"number of lines":length, "total_time": total_time} 

                else:
                    return_json = {"code":snippet, "buffer": True, "language": "some english text", "number of lines":length, "total_time": total_time}

                json_array.append(return_json)

            end = time.time()
            json_ = {
                "time": end-start0,
                "language": json_array,
                "algorithm": "tensorflow"
            }
            return json_
    except Exception as e:

        log("guess_language", "guess_language failed Something wrong happened at buffer, look out!", e)
            
        json_ = {"error":True,"err_msg":"error_function: {},error: {}".format(e.__class__.__name__,str(e))}
        
    return json_
    
    
# init_guess_lang()
# guess_language(snippet_array = code_input)
# guess_language(code_input2)