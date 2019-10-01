class operationOnLists:
    def __init__(self,oldList,goalList):
        self.oldList = oldList
        self.goalList = goalList


    def divide_equitably(self):
        oldListSize = len(self.oldList)
        if oldListSize<len(self.goalList):
            CRED = '\033[91m'
            CEND ='\033[0m'
            print(CRED+'ERR: util.operationOnLists\n        the oldLIst is lower than goalList'+CEND)
            return 0
        diff = oldListSize-len(self.goalList)
        print("diff",diff)
        saut = oldListSize/diff
        saut = 2.5
        print("saut",saut)
        print ("ratio",oldListSize/diff)

        i=0
        nbIndexDeleted = 0
        
        while len(self.oldList) !=len(self.goalList):
            
            print("new lenght", len(self.oldList))
            if i > int(i):
                print("Index to delete: ",int(i)+1-nbIndexDeleted)
                self.oldList.pop(int(i)+1-nbIndexDeleted)
                nbIndexDeleted +=1

            else:
                print("Index to delete: ",int(i)-nbIndexDeleted)
                self.oldList.pop(int(i)-nbIndexDeleted)
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
    new,goal=operationOnLists(a,b).divide_equitably()
    print("longueur de goal: ",len(new))
    print("longueur de goal: ",len(goal))

if __name__ == "__main__":
    main()
    """