import queue
import threading
import click
import torch
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import whisper
import os
import json
import traceback
from time import time
from models.customlogger import mldebugger,INFO,log

dir_path = os.path.dirname(__file__)

class main:
    '''
    Main audio transcription class to record and transcribe audio.
    '''
    
    def __init__(self, model, english, verbose, energy, pause, dynamic_energy,audio_file_path):
        '''
        @param:
            model: Name of the model to use for transcription
            english: Boolean to determine if the English model should be used
            verbose: Boolean to determine verbosity of output
            energy: Energy level for mic to detect
            pause: Pause time before entry ends
            dynamic_energy: Flag to enable dynamic energy adjustment
            audio_file_path: Path to an audio file (if not using microphone)

        @output: Initializes the class with necessary configurations.
        '''
        if model != "large":
            model = model + ".en"
        name = model + ".pt"
        if os.path.isfile(name):
            self.audio_model = open(name, "rb").read()
        else:
            self.audio_model = whisper.load_model(model)
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.english = english
        self.verbose = verbose
        self.audio_file_path = audio_file_path

    def record_audio(self):
        '''
        Record audio from either a microphone or a file.
        
        @output: Pushes the recorded audio data to the audio queue.
        '''
        try:
            r = sr.Recognizer()
            r.energy_threshold = self.energy
            r.pause_threshold = self.pause
            r.dynamic_energy_threshold = self.dynamic_energy

            devices = sd.query_devices()
            if len(devices):
                print("No microphone found! Using audio file.")
                # audio_file_path = "male.wav"  # update with your audio file path
                audio_file = sr.AudioFile(self.audio_file_path)
                with audio_file as source:
                    audio = r.record(source)
                    torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                    audio_data = torch_audio
                    self.audio_queue.put_nowait(audio_data)
            else:
                with sr.Microphone(sample_rate=16000) as source:
                    print("Say something!")
                    while True:
                        audio = r.listen(source)
                        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                        audio_data = torch_audio
                        self.audio_queue.put_nowait(audio_data)
        except Exception as e:
            print("Error occurred during audio recording:", str(e))
    
    def transcribe_forever(self):
        '''
        Continuously transcribes audio data fetched from the audio queue.
        
        @output: Pushes the transcription results to the result queue.
        '''
        
        try:
            while True:
                audio_data = self.audio_queue.get()
                if self.english:
                    result = self.audio_model.transcribe(audio_data, language='english')
                else:
                    result = self.audio_model.transcribe(audio_data)
                if not self.verbose:
                    predicted_text = result["text"]
                    self.result_queue.put_nowait("You said: " + predicted_text)
                else:
                    self.result_queue.put_nowait(result)
        except Exception as e:
            print("Error occurred during transcription:", str(e))
            

    def run(self):
        
        '''
        Start threads for audio recording and transcription, and processes transcription results.
        
        @returns: A dictionary containing transcription details such as filename, output_text, and text_length.
        @type: dict
        '''
        
        record_thread = threading.Thread(target=self.record_audio)
        transcribe_thread = threading.Thread(target=self.transcribe_forever)
        
        # Set threads as daemonic
        record_thread.daemon = True
        transcribe_thread.daemon = True

        record_thread.start()
        transcribe_thread.start()

        while True:
            try:
                res = []
                result = self.result_queue.get()
                output_text = result["text"]
                # res.append(result)
                filename = self.audio_file_path
                res = {"filename":filename,"output_text":output_text,"text_length":len(output_text)}
            except Exception as e:
                print("Error occurred while processing result:", str(e))
                
            return res

# @click.command()
# @click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
# @click.option("--english", default=False, help="Whether to use English model", is_flag=True, type=bool)
# @click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
# @click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
# @click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
# @click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)


def init_speech_recog():
    '''
    Initialize the speech recognition with default parameters.

    @output: Logs the initiation results using the mldebugger.
    '''
    model = "base"
    english = True
    verbose = True
    energy = 300
    pause = 0.8
    dynamic_energy = True
    audio_file = os.path.join(dir_path,"male.wav")
    output = SpeechRecognizer(model=model,english=english, verbose=verbose, energy=energy, pause=pause, dynamic_energy=dynamic_energy,audio_file_path = audio_file)
    # print(output)
    mldebugger.simplelog("init_speech_recog()", "init_speech_recog has been successfully initiated, " +  "init_speech_recog output: " + json.dumps(output), INFO)


def SpeechRecognizer(model, english, verbose, energy, pause, dynamic_energy,audio_file_path):
    
    '''
    Transcribes speech using the specified model and parameters.
    
    @param:
        model (str): Name of the model to use for transcription.
        english (bool): True if the English model should be used; False otherwise.
        verbose (bool): True for verbose output; False for concise output.
        energy (int): Energy level for microphone sensitivity.
        pause (float): Duration of pause after which transcription is considered complete.
        dynamic_energy (bool): True to enable dynamic energy adjustment; False otherwise.
        audio_file_path (str): Path to an audio file for transcription (used if no microphone available).
        
    @returns:
        list:
            A list of dictionaries, where each dictionary contains:
                - model (str): Model name used for transcription.
                - data (dict): Transcription result.
                - time (float): Duration taken for transcription.
                - error (bool, optional): Indicates if an error occurred.
                - err_msg (str, optional): Detailed error message, if any.
    
    @output: 
        Provides a JSON representation of the transcription results and any relevant metadata.
        
        [{"model": "base", "data": {"filename": "/home/mlserver1/ml_services/models/speechrecognition/whisper/male.wav", "output_text": " that life suddenly decides to break it, be careful that you keep adequate coverage. But what for places to save money? Maybe it's taking longer to get things squared away than the bankers expected. Hiring the life for once company may win her concert at the climate and count. The boost is helpful, but inadequate. New self-assessment lines are heard late costs on the two-legged bones. What a discussion can't infer when a cuddle of this kind of song is in question. There's no dying or waxing or gasping. There's no delay, maybe personalize on black heart players whether high-quality flat surface can smooth out. There's just what kind of separate system uses a single self-contained unit to your childhood still holds. I think the counter-cassures will be bad, but both figures will die out in later years. Some beautiful chairs, cabinets, chest, dowhouses, etc.", "text_length": 867}, "time": 7.146722316741943}]

        
    @input:
    
    output = SpeechRecognizer(model=model,english=english, verbose=verbose, energy=energy, pause=pause, dynamic_energy=dynamic_energy,audio_file_path = audio_file)

    '''
    try:
        json_response=[]
        start_time = time()
        recognizer = main(model, english, verbose, energy, pause, dynamic_energy,audio_file_path)
        result = recognizer.run()
        end_time = time()-start_time
        json_ = {"model":model,"data":result,"time": end_time}
        json_response.append(json_)
    except Exception as e:
#         traceback.print_exc()
        log("SpeechRecognizer()", "SpeechRecognizer failed Something wrong happened, look out!", e)
        json_ = {"model":model,"data":" ","time": 0, "error":True,"err_msg":"error_function: {}, error: {}".format(e.__class__.__name__,str(e))}
        json_response.append(json_)

    return json_response 
# Example usage
# model = "base"
# english = True
# verbose = True
# energy = 300
# pause = 0.8
# dynamic_energy = True
# audio_file_path = os.path.join(dir_path,"male.wav")

# a =SpeechRecognizer(model,english, verbose, energy, pause, dynamic_energy,audio_file_path)

# print(a)


# [{'model': 'base', 'data': {'filename': 'male.wav', 'output_text': " that life suddenly decides
# to break it, be careful that you keep adequate coverage. But what for places to save money? 
# Maybe it's taking longer to get things squared away than the bankers expected. Hiring the 
# life for once company may win her concert at the climate and count. The boost is helpful, 
# but inadequate. New self-assessment lines are heard late costs on the two-legged bones. 
# What a discussion can't infer when a cuddle of this kind of song is in question. There's 
# no dying or waxing or gasping. There's no delay, maybe personalize on black heart players 
# whether high-quality flat surface can smooth out. There's just what kind of separate system 
# uses a single self-contained unit to your childhood still holds. I think the counter-cassures 
# will be bad, but both figures will die out in later years. Some beautiful chairs, cabinets, 
# chest, dowhouses, etc.", 'text_length': 867}, 'time': 4.97105073928833}]

