from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play


#C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/clips/common_voice_fr_17299484
sound_file = AudioSegment.from_wav("common_voice_fr_17300098.wav")

#play(sound_file)
audio_chunks = split_on_silence(sound_file, 

    # split on silences longer than 1000ms (1 sec)
    min_silence_len=75,

    # anything under -16 dBFS is considered silence
    silence_thresh=-40

    # keep 200 ms of leading/trailing silence
    #keep_silence=200
)
print("ok")
for i, chunk in enumerate(audio_chunks):
    
    out_file = ".//splitAudio//chunk{0}.wav".format(i)
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")