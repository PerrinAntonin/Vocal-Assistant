#len(a)=68
a = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]
b = [3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,3,1,2,3,1,2,3,1,2,3]

bsize =len(b)
print("longueur de a: ",len(a))
print("longueur de b: ",len(b))
diff = len(b)-len(a)
print("diff",diff)
saut = len(b)/diff
saut = 2.5
print("saut",saut)
print ("ratio",len(b)/diff)

#for i in range(0, len(b), 3):
#    if i < len(b): 
#        b.pop(i)
i=0

exitwhile=False
while len(b) !=len(a):
    
    print("new lenght", len(b))
    if i > int(i):
        if i<bsize:
            print(int(i)+1)
            b.pop(int(i)+1)
        else:
            exitwhile = True

    else:
        if i<bsize:
            print(int(i))
            b.pop(int(i))

        else:
            exitwhile = True

    if exitwhile == True:
        break
    i +=saut

print("longueur de a after: ",len(a))
print("longueur de b after: ",len(b))
#while(len(a)!=len(b)):

