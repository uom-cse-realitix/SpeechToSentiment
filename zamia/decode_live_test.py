#!/usr/bin/env python

from __future__ import print_function

import os

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

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

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "utt1.wav"
SAVE_PATH = 'data/test'

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
asr = NnetLatticeFasterRecognizer.from_files(
    "/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/exp/nnet3_chain/tdnn_f/final.mdl",
    "/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/exp/nnet3_chain/tdnn_f/graph/HCLG.fst",
    "/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/data/lang/words.txt",
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")


while True:
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* processing...")

    save_speech(frames, p, SAVE_PATH, WAVE_OUTPUT_FILENAME)
    # filename = os.path.join(SAVE_PATH, WAVE_OUTPUT_FILENAME)
    # wf = wave.open(filename, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()

    # Define feature pipelines as Kaldi rspecifiers
    feats_rspec = (
        "ark:compute-mfcc-feats --config=/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/conf/mfcc_hires.conf scp:data/test/wav.scp ark:- |"
    )
    ivectors_rspec = (
        "ark:compute-mfcc-feats --config=/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/conf/mfcc_hires.conf scp:data/test/wav.scp ark:- |"
        "ivector-extract-online2 --config=/home/ashen/Documents/GitHubRepos/pykaldi/examples/setups/zamia/conf/ivector_extractor.conf ark:data/test/spk2utt ark:- ark:- |"
    )

    # Decode wav files
    with SequentialMatrixReader(feats_rspec) as f, \
         SequentialMatrixReader(ivectors_rspec) as i, \
         open("out/test/decode.out", "w") as o:
        for (key, feats), (_, ivectors) in zip(f, i):
            out = asr.decode((feats, ivectors))
            # print(key, out["text"], file=o)
            print("Detected text: ", out["text"])

stream.stop_stream()
stream.close()
p.terminate()