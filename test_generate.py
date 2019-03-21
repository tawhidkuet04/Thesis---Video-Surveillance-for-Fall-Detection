f = open("F:\Thesis\darkflow-master\darkflow-master\calcu\cc.txt", "x")
with open("F:\Thesis\darkflow-master\darkflow-master\calcu\pp.txt", encoding="utf-8") as file:
        x = [l.strip() for l in file]

for i in x :
    words = i.split()
    if words[0] == '0' :
        x = int(words[2]) - int(words[1]) + 1 
        for j in range(x) :
            f.write('0\n')
    else :
        x = int(words[2]) - int(words[1]) + 1 
        for j in range(x):
            f.write('1\n')
f.close()