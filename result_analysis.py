with open("F:\Thesis\darkflow-master\darkflow-master\calcu\cc.txt", encoding="utf-8") as file:
        x = [l.strip() for l in file]
with open("F:\Thesis\darkflow-master\darkflow-master\calcu\ccc.txt", encoding="utf-8") as file:
        y = [l.strip() for l in file]
tp = 0 
tn = 0
fp = 0 
fn = 0 
for i,j in zip(x,y) :
    g = i.split()
    s = j.split()
    if g[0] == s[0] and g[0] == '0' :
        tn += 1
    if g[0] == s[0] and g[0] == '1' :
        tp += 1 
    if g[0] != s[0] and g[0] == '1' and s[0] == '0' :
        fn += 1
    if g[0] != s[0] and g[0] == '0' and s[0] == '1' :
        fp += 1 
accuracy = (tp+tn)/(tp+fp+tn+fn)
if tp+fp == 0 and fp == 0  :
    precison = 0    
else :
    precison = tp/(tp+fp)
if tp+fn == 0 and tp == 0 :
    sensivity = 0 
else :
    sensivity = tp/(tp+fn)
specificity = tn/(tn+fp)
if precison == 0 :
    f1 = 0 
else :
    f1 = (2*precison*sensivity)/(precison+sensivity)
fpr = fp/(fp+tn)
f = open("F:/out2/test/res.txt", "x")
f.write("tp  -- %d  tn -- %d fp -- %d fn -- %d \n" % (tp,tn,fp,fn))
f.write("accuracy -- %f\n" % (accuracy*100))
f.write("precison -- %f\n" % (precison*100))
f.write("sensivity -- %f\n" % (sensivity*100))
f.write("specificity -- %f\n" % (specificity*100))
f.write("f1 -- %f\n" % (f1*100))
f.write("fpr --%f\n" % (fpr*100))
#print("accuracy --",accuracy*100)
#print("precison --",precison*100)
#print("sensivity --",sensivity*100)
#print("f1 --",f1*100)
#print("fpr --",fpr*100)
f.close()