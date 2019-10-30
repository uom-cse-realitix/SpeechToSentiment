# SpeechToSentiment

This repository integrates Pykaldi Speech Recognition Toolkit with a text classifier 
to obtain the sentiment of the user's command (locate, describe, no_op).

### Steps to setup:

1) Create an environment using Virtualenv or Conda
2) Clone the Pykaldi repository from https://github.com/pykaldi/pykaldi.git and build 
from source by following its `README` installation instructions
3) In the `zamia` directory, run `./models.sh` to download the zamia model
4) Set the source to the kaldi installation by running  `source path.sh`
5) Run `./decode_live_smart_test.py` to start sentiment analysis