import os
import glob
import pydub
import librosa
#import customCsv as cCsv
import numpy as np
import matplotlib as plt


class editSongs  :
    #convert mp3 to wav
    def mp3towav(self,names,path):
        for name in names:
            pathName = path+name

            if(os.path.exists(pathName)):   
                sound = pydub.AudioSegment.from_mp3(pathName)
                newFile=pathName.replace(".mp3", ".wav")
                #peut etre a mettre: ,bitrate='16k', parameters=["-acodec","pcm_u16le","-ac","1","-ar","8000"]
                sound.export(newFile, format="wav",bitrate='16k')
                os.remove(pathName)

    #delete mp3 files
    def delete_unused_files(self,pathFile):
        os.chdir(pathFile)
        files=glob.glob('*.mp3')
        for file in files:
            os.remove(file)

    #get mfccs from wavs song
    def loaMffcsFromWav(self,names,path,len_chunk = 641):
        wavFiles=[]
        for file in names:
            file=path+file
            file = file.replace(".mp3", ".wav")
            y, sr = librosa.load(file, sr=16000)
            audios = self.split_list(y,len_chunk)
            song=[]
            for audio in audios:
                audio = np.array(audio)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                song.append(mfccs)

            wavFiles.append(song)
        return wavFiles
    #split list by chunk / divide the song all n ms
    def split_list(self,a_list,len_chunk):
        chunks = []
        for i in range(0, len(a_list), len_chunk):  
            chunks.append(a_list[i:i + len_chunk])
        chunks=chunks[:-1] 
        #print("chunk",chunks[:-1])
        return chunks

    def displayMffc(self,mfcc,text):
        print(text)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title(text)
        plt.tight_layout()
        plt.show()

#exemple of use
"""
def main():
    pathFile ="C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/clips/"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    pathCsv2 = "C:/Users/tompe/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    pathFileV2 ="C:/Users/tompe/Documents/deepLearning/Vocal_Assistant/data/clips/"
                
    #va récuperer toutes les informations du csv concernant le nom du song et son texte
    SongCsv = cCsv.customCsv(pathCsv2)
    SongCsv.readcsv()
    toolSong = editSongs()
    names,texts = SongCsv.getContent() 

    print("exemple: ",names[1])
    names = names[:40]
    #va checker si le fichier exister et le convertir en mp3
    #toolSong.mp3towav(names,pathFile)

    mfccs= toolSong.loaMffcsFromWav(names,pathFileV2)
    print("mfccs load")
    #va supprimer tout les fichier mp3 qu'il reste 
    #toolSong.delete_unused_files(pathFile)

if __name__ == "__main__":
    main()
"""