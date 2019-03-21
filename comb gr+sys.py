with open("F:\Thesis\darkflow-master\darkflow-master\calcu\gr.txt", encoding="utf-8") as file:
        x = [l.strip() for l in file]
with open("F:\Thesis\darkflow-master\darkflow-master\calcu\sys.txt", encoding="utf-8") as file:
        y = [l.strip() for l in file]
f = open("F:/out2/gr+sys.txt", "x")

f.write("Frame   Grount truth   System\n")
cnt = 1 
for i,j in zip(x,y) :
    g = i.split()
    s = j.split()
    f.write("%d           %d             %d\n" % (cnt,int(g[0]),int(s[0])))
    cnt += 1 

f.close()