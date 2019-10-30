#!/usr/bin/env python

from __future__ import print_function

import os
from collections import deque
import math
import audioop

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.base import set_verbose_level

from classifier import pre_initialize, get_sentiment
from tensorflow.keras.models import load_model

import pyaudio
import wave

def save_speech(data, p, save_path, wave_output_filename):
    """ Saves mic data to temporary WAV file. Returns filename of saved
        file """
    filename = os.path.join(save_path, wave_output_filename)
    # writes data to WAV file
    _data = b''.join(data)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(_data)
    wf.close()
    return filename

def recognize_speech():
    # Define feature pipelines as Kaldi rspecifiers
    feats_rspec = (
        "ark:compute-mfcc-feats --config=conf/mfcc_hires.conf scp:data/test/wav.scp ark:- |"
    )
    ivectors_rspec = (
        "ark:compute-mfcc-feats --config=conf/mfcc_hires.conf scp:data/test/wav.scp ark:- |"
        "ivector-extract-online2 --config=conf/ivector_extractor.conf ark:data/test/spk2utt ark:- ark:- |"
    )

    # Decode wav files
    with SequentialMatrixReader(feats_rspec) as f, \
            SequentialMatrixReader(ivectors_rspec) as i, \
            open("out/test/decode.out", "a+") as o:
        for (key, feats), (_, ivectors) in zip(f, i):
            out = asr.decode((feats, ivectors))
            print(out["text"], file=o)
            # print("Detected text: ", out["text"])
            return out["text"]

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "utt1.wav"
SAVE_PATH = 'data/test'

THRESHOLD = 6000  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.

num_phrases = -1

set_verbose_level(0)

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
asr = NnetLatticeFasterRecognizer.from_files(
    "/media/ashen/52DAB5C7DAB5A81D/Ubuntu_test/zamia/exp/nnet3_chain/tdnn_f/final.mdl",
    "/media/ashen/52DAB5C7DAB5A81D/Ubuntu_test/zamia/exp/nnet3_chain/tdnn_f/graph/HCLG.fst",
    "/media/ashen/52DAB5C7DAB5A81D/Ubuntu_test/zamia/data/lang/words.txt",
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts)

p = pyaudio.PyAudio()

############################################
_, tokenizer = pre_initialize()
model = load_model('./lstm.h5')
############################################

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

audio2send = []
cur_data = ''  # current chunk  of audio data
rel = RATE/CHUNK
slid_win = deque(maxlen=int(SILENCE_LIMIT * rel) + 1)
#Prepend audio from 0.5 seconds before noise was detected
prev_audio = deque(maxlen=int(PREV_AUDIO * rel) + 1)
started = False
n = num_phrases
response = []

while num_phrases == -1 or n > 0:
    cur_data = stream.read(CHUNK)
    slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
    #print slid_win[-1]
    if sum([x > THRESHOLD for x in slid_win]) > 0:
        if(not started):
            #print "Starting record of phrase"
            started = True
        audio2send.append(cur_data)
    elif (started is True):
        #print "Finished"
        # The limit was reached, finish capture and deliver.
        filename = save_speech(list(prev_audio) + audio2send, p, SAVE_PATH, WAVE_OUTPUT_FILENAME)
        # Send file to Google and get response
        r = recognize_speech()
        if num_phrases == -1:
            print ("Detected speech: ", r)
            get_sentiment(r)
        else:
            response.append(r)
        # Remove temp file. Comment line to review.
        os.remove(filename)
        # Reset all
        started = False
        slid_win = deque(maxlen=int(SILENCE_LIMIT * rel) + 1)
        prev_audio = deque(maxlen=int(0.5 * rel) + 1)
        audio2send = []
        n -= 1
        #print "Listening ..."
    else:
        prev_audio.append(cur_data)


stream.stop_stream()
stream.close()
p.terminate()