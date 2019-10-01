#len(a)=68
a = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]
b = [3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]


print("longueur de a: ",len(a))
print("longueur de b: ",len(b))
diff = len(b)-len(a)
print("diff",diff)
saut = len(b)/diff
saut = 2.5
print("saut",saut)

bsize =len(b)
i=0
index_to_delete = []
indexDeleted = 0
exitwhile=False
while len(b) !=len(a):
    if i > int(i):
        if i<bsize:
            b.pop(int(i)+1-indexDeleted)
            indexDeleted +=1
        else:
            exitwhile = True
    else:
        if i<bsize:
            b.pop(int(i)-indexDeleted)
            indexDeleted +=1
        else:
            exitwhile = True

    if exitwhile == True:
        break
    i +=saut

print("longueur de a after: ",len(a))
print("longueur de b after: ",len(b))