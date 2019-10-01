import csv
import glob
import numpy as np
import os
import pydub
from os import path

def mp3towav(names,path):
    for name in names:
        pathName = path+name
        if(path.exist(pathName)):   
            sound = pydub.AudioSegment.from_mp3(pathName)
            newFile=pathName.replace(".mp3", ".wav")
            #peut etre a mettre: ,bitrate='16k', parameters=["-acodec","pcm_u16le","-ac","1","-ar","8000"]
            sound.export(newFile, format="wav",bitrate='16k')
            os.remove(pathName)

def delete_unused_files():
    files=glob.glob(path+'*.mp3')
    for file in files:
        os.remove(file)

def readcsv(pathCsv):
    with open(pathCsv , encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        names=[]
        texts=[]
        for line in tsvreader: 
            #buffer = line[1].replace("mp3", "wav")
            buffer = line[1]
            names.append(buffer)
            texts.append(line[2])
    names=names[1:]
    texts=texts[1:]
    print("nb de csv",len(names))
    return names,texts


if __name__ == "__main__":
    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    #va récuperer toutes les informations du csv concernant le nom du song et son texte 
    names,texts = readcsv(pathCsv)
    print("exemple: ",names[1])
    #va checker si le fichier exister et le convertir en mp3
    mp3towav(names,pathFile)
    #va supprimer tout les fichier mp3 qu'il reste 
    delete_unused_files()
