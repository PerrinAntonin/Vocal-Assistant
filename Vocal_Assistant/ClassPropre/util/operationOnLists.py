import coloredlogs, logging

class operationOnLists:
    def __init__(self,oldList,goalList):
        self.oldList = oldList
        self.goalList = goalList


    def divide_equitably(self):
        logger = logging.getLogger(__name__)
        coloredlogs.install(level='ERROR', logger=logger)

        oldListSize = len(self.oldList)
        if oldListSize<len(self.goalList):            
            logger.error('ERR: util.operationOnLists\nthe oldLIst is lower than goalList')
            return 0
        diff = oldListSize-len(self.goalList)
        #print("diff",diff)
        saut = oldListSize/diff
        #saut = round(saut,1)
        #print("saut",saut)
        #print ("ratio",oldListSize/diff)

        i=0
        nbIndexDeleted = 0
        
        while len(self.oldList) !=len(self.goalList):
            #print("new lenght", len(self.oldList))
            #print("Index to delete: ",int(round(i,0))-nbIndexDeleted)
            self.oldList.pop(int(round(i,0))-nbIndexDeleted)
            nbIndexDeleted +=1
            i +=saut
        return self.oldList,self.goalList
    
    def print(self):
        print(self.oldList,self.goalList)

#exemple of use
"""
def main():
    a = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]
    b = [3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]
    print("longueur de a: ",len(a))
    print("longueur de b: ",len(b))
    new,goal=operationOnLists(b,a).divide_equitably()
    print("longueur de goal: ",len(new))
    print("longueur de goal: ",len(goal))

if __name__ == "__main__":
    main()
    """