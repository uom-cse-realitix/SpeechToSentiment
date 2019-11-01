from decode_live_smart_test import SpeechRecognizer
import os
# from online_asr.online_asr import

dir_path = os.path.dirname(os.path.realpath(__file__))
# filename = os.path.join(dir_path, "zamia")
sr = SpeechRecognizer(dir_path)
sr.main()