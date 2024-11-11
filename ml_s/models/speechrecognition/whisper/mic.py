# importing libraries
# import io
# from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import sounddevice as sd


@click.command()
# choosing model, default is base
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
# check for English model
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
#  verbose: bool Whether to display the text being decoded to the console. If True, displays all the details,
# If False, displays minimal details. If None, does not display anything
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
# Energy of a speech is another parameter for classifying the voiced/unvoiced parts 
# The voiced part of the speech has high energy because of its periodicity
# the unvoiced part of speech has low energy.
# The energy_threshold value is set to 300 by default. Under 'ideal' conditions (such as in a quiet room)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
# time delay for speech to end
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
def main(model, english,verbose, energy, pause,dynamic_energy):
    # there are no english models for large
    if model != "large":
        model = model + ".en"
    name = model + ".pt"
    # check if model is already present
    if os.path.isfile(name): 
        audio_model = open(name, "rb").read()
    else:
        audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    threading.Thread(target=record_audio,
                     args=(audio_queue, energy, pause, dynamic_energy)).start()
    threading.Thread(target=transcribe_forever,
                     args=(audio_queue, result_queue, audio_model, english, verbose)).start()

    while True:
        print(result_queue.get())


# def record_audio(audio_queue, energy, pause, dynamic_energy):
#     #load the speech recognizer and set the initial energy threshold and pause threshold
#     r = sr.Recognizer()
#     r.energy_threshold = energy
#     r.pause_threshold = pause
#     r.dynamic_energy_threshold = dynamic_energy

#     # speech detection
#     with sr.Microphone(sample_rate=16000) as source:
#         print("Say something!")
#         i = 0
#         while True:
#             #get audio to wav file
#             audio = r.listen(source)
#             torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
#             audio_data = torch_audio
#         # collecting data chunks and put them in queue
#             audio_queue.put_nowait(audio_data)
#             i += 1
            
            

def record_audio(audio_queue, energy, pause, dynamic_energy):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    devices = sd.query_devices()
    if len(devices):
        # speech detection
        print("No microphone found! Using audio file.")
        audio_file_path = "male.wav"  # update with your audio file path
        audio_file = sr.AudioFile(audio_file_path)
        with audio_file as source:
            audio = r.record(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)
    else:     
        with sr.Microphone(sample_rate=16000) as source:
            print("Say something!")
            i = 0
            while True:
                #get audio to wav file
                audio = r.listen(source)
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio
            # collecting data chunks and put them in queue
                audio_queue.put_nowait(audio_data)
                i += 1


def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose):
    # infinite loop
    while True:
        audio_data = audio_queue.get()
        # transcribe
        if english:
            result = audio_model.transcribe(audio_data,language='english')
        else:
            result = audio_model.transcribe(audio_data)
        # result
        if not verbose:
            predicted_text = result["text"]
            result_queue.put_nowait("You said: " + predicted_text)
        else:
            result_queue.put_nowait(result)


main()
