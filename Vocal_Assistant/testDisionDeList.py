#len(a)=68
a = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]
b = [3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]

print("longueur de a: ",len(a))
print("longueur de b: ",len(b))

def divide_equitably(oldList,goalList):
    oldListSize = len(oldList)
    diff = oldListSize-len(goalList)
    print("diff",diff)
    saut = oldListSize/diff
    saut = 2.5
    print("saut",saut)
    print ("ratio",oldListSize/diff)

    i=0
    nbIndexDeleted = 0
    
    while len(oldList) !=len(goalList):
        
        print("new lenght", len(oldList))
        if i > int(i):
            print("Index to delete: ",int(i)+1-nbIndexDeleted)
            oldList.pop(int(i)+1-nbIndexDeleted)
            nbIndexDeleted +=1

        else:
            print("Index to delete: ",int(i)-nbIndexDeleted)
            oldList.pop(int(i)-nbIndexDeleted)
            nbIndexDeleted +=1
        i +=saut
    return oldList,goalList

goalList,oldList =divide_equitably(b,a)

print("longueur de goalList after: ",len(goalList))
print("longueur de oldList after: ",len(oldList))