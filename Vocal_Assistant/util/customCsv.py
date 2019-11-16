import csv

class customCsv:
    def __init__(self,pathCsv):
        self._pathCsv = pathCsv
        self._names=[]
        self._texts=[]

    def readcsv(self):
        with open(self._pathCsv , encoding="utf8") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")

            for line in tsvreader: 
                #buffer = line[1].replace("mp3", "wav")
                buffer = line[1]
                self._names.append(buffer)
                self._texts.append(line[2])
        self._names=self._names[1:]
        self._texts=self._texts[1:]

    def getContent(self):
        print("nb of line found in csv: ",len(self._names))
        return self._names,self._texts

#exemple of use
"""
if __name__ == "__main__":
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    exampleCsv = customCsv(pathCsv)
    exampleCsv.readcsv()
    names,texts = exampleCsv.getContent()
    print("first exemple: ",names[0],"\n            ",texts[0])
    """