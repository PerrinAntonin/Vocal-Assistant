import glob
import pydub
import os
import customCsv

class editSongs:    
    def mp3towav(self,names,path):
        for name in names:
            pathName = path+name

            if(os.path.exists(pathName)):   
                sound = pydub.AudioSegment.from_mp3(pathName)
                newFile=pathName.replace(".mp3", ".wav")
                #peut etre a mettre: ,bitrate='16k', parameters=["-acodec","pcm_u16le","-ac","1","-ar","8000"]
                sound.export(newFile, format="wav",bitrate='16k')
                os.remove(pathName)

    def delete_unused_files(self,pathFile):
        os.chdir(pathFile)
        files=glob.glob('*.mp3')
        for file in files:
            os.remove(file)


def main():
    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    #va récuperer toutes les informations du csv concernant le nom du song et son texte
    SongCsv = customCsv.customCsv(pathCsv)
    SongCsv.readcsv()
    toolSong = editSongs.editSongs()
    names,texts = SongCsv.getContent() 

    print("exemple: ",names[1])

    #va checker si le fichier exister et le convertir en mp3
    toolSong.mp3towav(names,pathFile)
    #va supprimer tout les fichier mp3 qu'il reste 
    toolSong.delete_unused_files(pathFile)

if __name__ == "__main__":
    main()