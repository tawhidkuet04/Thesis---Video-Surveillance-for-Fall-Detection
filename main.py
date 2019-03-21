import numpy as np
import cv2
import math
import time
from PIL import Image
import sys
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
f = open("F:\Thesis\darkflow-master\darkflow-master\calcu\ccc.txt", "w")
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('F:/out2/chute(3)/video (29).xlsx')
worksheet = workbook.add_worksheet()

row = 0 
col = 0
worksheet.write(row, col,   "Frame Number")
worksheet.write(row, col + 1, "Angle deviation")
worksheet.write(row, col + 2, "Axis Deviation")
worksheet.write(row, col + 3, "mhi_cof" ) 
worksheet.write(row, col + 4 , "w/h")
row += 1 


#############human detection initialize

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15,
    'gpu': 1.0
}

tfnet = TFNet(option)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
###############################
print(cv2.__version__)
xo = []
yo = []
yan = []
yaxi = []
ywid = []
prev = 0 
count = 0 
cap = cv2.VideoCapture('video (29).avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
arr = [] 
brr = []
crr = []
grid =[]
cnt = 0 
cnt2 = 0
cnt3 = 0
cnt4 =0 
cnt5 = 0
pk = 0 
fr = 5
normal = 0 
h = 0 
w = 0 
flag = 0 
MHI_DURATION = 10
DEFAULT_THRESHOLD = 127
timestamp = 0 
rectangle_analysis = 0 

rectangle_analysis = 0  
def getNeighbours(i, j, n, m) :
    arr = []
    if i-1 >= 0 and j-1 >= 0 :
        arr.append((i-1, j-1))
    if i-1 >= 0 :
        arr.append((i-1, j))
    if i-1 >= 0 and j+1 < m :
        arr.append((i-1, j+1))
    if j+1 < m :
        arr.append((i, j+1))
    if i+1 < n and j+1 < m :
        arr.append((i+1, j+1))
    if i+1 < n :
        arr.append((i+1, j))
    if i+1 < n and j-1 >= 0 :
        arr.append((i+1, j-1))
    if j-1 >= 0 :
        arr.append((i, j-1))
    return arr

ret,frame = cap.read()
(height,width)= (frame.shape[:2])
motion_history = np.zeros((height, width), np.float32)
##################################
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

######################################
while(cap.isOpened()):
    ret, frame = cap.read()
    if  ret == False :
        break
    
    xo.append(count)
##################### Background Subtraction + Morphological Operation ########################
    fgmask = fgbg.apply(frame)
    bg = fgmask 
    retval, threshold = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
    mask = threshold
    fggmask=threshold
 
    kernel = np.ones((15,15), np.uint8)
    blurred = cv2.blur(mask,(20,20))
    retval, threshold = cv2.threshold(blurred,30, 255, cv2.THRESH_BINARY)
    closing=cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel)
    retval, threshold = cv2.threshold(closing, 150, 255, cv2.THRESH_BINARY)
    mask = threshold 
    fgmask=  threshold 
    
    
##################### Background Subtraction + Morphological Operation ########################   




    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if len(contour_sizes) == 0:
        arr.clear()
        brr.clear()
        if len(yo) == 0 :
                yo.append(1.0)
        else :
                yo.append(yo[len(yo)-1])
        if len(yan) == 0 :
                yan.append(0)
        else :
                yan.append(yan[len(yan)-1])
        if len(ywid) == 0 :
                ywid.append(0)
        else :
                ywid.append(ywid[len(ywid)-1])      
        if len(yaxi) == 0 :
                yaxi.append(0)
        else :
                yaxi.append(yaxi[len(yaxi)-1])                  
        continue 
########################## motion history

    timestamp  += 1
    cv2.motempl.updateMotionHistory(mask, motion_history,timestamp , MHI_DURATION)
    mh = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
#######################################   
    

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    

################human detection#############################
    results = tfnet.return_predict(frame)
    mx = 0.0 
    tl=(0,0)
    br=(0,0)
    lab = ""
    for color, result in zip(colors, results):
        ttl = (result['topleft']['x'], result['topleft']['y'])
        bbr = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        a = "person"
        if label == a  :
            if result['confidence'] > mx :
                    mx = result ['confidence']
                    tl = ttl 
                    br = bbr 
                    lab= label
    w = 0
    h = 0 
    if mx >= .65 :
        w = abs(int(br[0])-int(tl[0]))
        h = abs(int(tl[1])-int(br[1]))
        
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2) 
    else :
        x,y,w,h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    if w/h >= 1.1:
        rectangle_analysis = rectangle_analysis + 1 
        cnt5 = 0 
    if h > w :
        cnt5 += 1
    if cnt5 > 5 :
        cnt5 = 0 
        rectangle_analysis = 0 

##############################################################

    if cv2.contourArea(biggest_contour) < 3000 :
        arr.clear()
        brr.clear()
        if len(yo) == 0 :
                yo.append(1.0)
        else :
                yo.append(yo[len(yo)-1])
        if len(yan) == 0 :
                yan.append(0)
        else :
                yan.append(yan[len(yan)-1])
        if len(ywid) == 0 :
                ywid.append(0)
        else :
                ywid.append(ywid[len(ywid)-1])      
        if len(yaxi) == 0 :
                yaxi.append(0)
        else :
                yaxi.append(yaxi[len(yaxi)-1])                 
        continue 

    _, contours, _ = cv2.findContours(mh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]


    biggest_contour_mh = max(contour_sizes, key=lambda x: x[0])[1]

    #print("%f" %(mh_pixel/(normal_pixel*50)))
    M = cv2.moments(biggest_contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
   
    MM = cv2.moments(biggest_contour_mh)
    # calculate x,y coordinate of center
    cXX = int(MM["m10"] / MM["m00"])
    cYY = int(MM["m01"] / MM["m00"])


    #print("%d %d" % (cX,cY))
######################### Calculate Co-Efficient ########################
    n = height
    m = width
    blob_val  = 0 
    mhi_val = 0 
    #fill state map
    stateMap = []
    for i in range(n):
         stateMap.append([False for j in range(m)])       
    queue = [(cY, cX)]
    while queue:
        e = queue.pop(0)
        i = e[0]
        j = e[1]
        if not stateMap[i][j]:
            stateMap[i][j] = True
            color = mask[i, j]
            if color > 10  : # check the color of the pixel against athreshold
                blob_val +=  mask[i, j] # fill light blue color
                neigh = getNeighbours(i, j, n, m)
                for ne in neigh:
                    queue.append(ne) # add neighbour pixels to the queue
    stateMap = []
    for i in range(n):
         stateMap.append([False for j in range(m)])       
    queue = [(cYY, cXX)]
    while queue:
        e = queue.pop(0)
        i = e[0]
        j = e[1]
        if not stateMap[i][j]:
            stateMap[i][j] = True
            color = mh[i, j]
            if color > 10  : # check the color of the pixel against a threshold
                mhi_val +=  mh[i, j] # fill light blue color
                neigh = getNeighbours(i, j, n, m)
                for ne in neigh:
                    queue.append(ne) # add neighbour pixels to the queue
                    
###################################################################                    
    if  blob_val == 0 :
        if len(yo) == 0 :
                yo.append(1.0)
        else :
                yo.append(yo[len(yo)-1])
        if len(yan) == 0 :
                yan.append(0)
        else :
                yan.append(yan[len(yan)-1])
        if len(ywid) == 0 :
                ywid.append(0)
        else :
                ywid.append(ywid[len(ywid)-1])      
        if len(yaxi) == 0 :
                yaxi.append(0)
        else :
                yaxi.append(yaxi[len(yaxi)-1])                   
        continue 
    #print("aaaa %f" % float(mhi_val/blob_val))
    mhi_cof = float(mhi_val/blob_val)
    mask = np.zeros(frame.shape, np.uint8)
    an = 0
    if M["mu20"]-M["mu02"] == 0 :
        if len(yo) == 0 :
                yo.append(1.0)
        else :
                yo.append(yo[len(yo)-1])
        if len(yan) == 0 :
                yan.append(0)
        else :
                yan.append(yan[len(yan)-1])
        if len(ywid) == 0 :
                ywid.append(0)
        else :
                ywid.append(ywid[len(ywid)-1])      
        if len(yaxi) == 0 :
                yaxi.append(0)
        else :
                yaxi.append(yaxi[len(yaxi)-1])                  
        continue
        
        



    an = (1.0/2.0)*math.atan((2*M["mu11"]/(M["mu20"]-M["mu02"])))
    an = math.degrees(an)   
    #cv2.drawContours(mask, [biggest_contour], 2, (0, 255, 0), 3)
    cv2.drawContours(mask, biggest_contour, -1,255, -1)
    ellipse = cv2.fitEllipse(biggest_contour)

    dev_angle = 0
    arr.append(an)
    # print("----%f" %(Major/Minor))
    i_min = (M["mu20"]+M["mu02"]-math.sqrt((M["mu20"]-M["mu02"])*(M["mu20"]-M["mu02"])+4 * M["mu11"]*M["mu11"]))/2
    i_max = (M["mu20"]+M["mu02"]+math.sqrt((M["mu20"]-M["mu02"])*(M["mu20"]-M["mu02"])+4 * M["mu11"]*M["mu11"]))/2
    const_val = math.sqrt(math.sqrt (4/math.pi))
    a = const_val * math.sqrt(math.sqrt(math.sqrt((i_max*i_max*i_max)/i_min)))
    b = const_val * math.sqrt(math.sqrt(math.sqrt((i_min*i_min*i_min)/i_max)))
    brr.append(a/b)
    ang_val = 0 
    axis_val = 0 
    #print("a %f %d" % (a/b,an))
    if len(arr) == fr :
        x_ = 0 
        tot = 0 
        for i in arr :
            x_ = x_ + i 
        x_ = x_ / fr
        for i in arr :
            tot = tot +  (i - x_) * (i-x_)
        dev_angle = tot/ fr
        ok = math.sqrt(dev_angle)
        ang_val = ok
       # print("an %d %f" % (count,ok) )  
        if ok > 15 :
            cnt = 1 
        else :
            cnt = 0 
    
    if len(brr) == fr :
        x_ = 0 
        tot = 0 
        for i in brr :
            x_ = x_ + i 
        x_ = x_ / fr
        for i in brr :
            tot = tot +  (i - x_) * (i-x_)
        dev_axis = tot/ fr
        ok = math.sqrt(dev_axis)
        axis_val = ok 
        #print("ratio %d %f" % (count,ok) ) 
        if ok > 0.3 :
            cnt2 = 1
        else :
            cnt2 = 0 
    if len(arr) == fr  :
        arr.pop(0)
    if len(brr) == fr  :
        brr.pop(0)
    #print(len(arr))
    cv2.ellipse(frame,ellipse,(255,0,0),2)
    cv2.ellipse(mask,ellipse,(255,0,0),2)
##################### Fall Detection Checking ###################
    if mhi_cof > 5 :
            mhi_cof = 0 
    if mhi_cof >= 1.3 :
        flag = 1 
    if flag == 1:
        if cnt == 1 or cnt2== 1 :
            cnt3 = 1 
    if cnt3== 1 :
        if mhi_cof< 1.1 :
            cnt4 += 1 
        else :
            pk += 1 
    if pk > 10 :
        flag = 0
        cnt =0 
        cnt2 = 0
        cnt4 = 0
        pk = 0 
    #print(cnt4)
    if mhi_cof == 0 :
        mhi_cof = 1.0
    yo.append(mhi_cof)
    yan.append(ang_val/10.0)
    yaxi.append(axis_val*5)
    ywid.append(w/h)
    count = count + 1 
    if cnt4 > 5 :
      #  print("------%d----%d------%d" %(cnt,cnt2,cnt3))
            if rectangle_analysis > 5 :
                cv2.putText(frame, 'Fall!!!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                f.write("1\n")
            else :
                f.write("0\n")
                cv2.putText(frame, 'Not Fall!!!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 0 , 255), 2, cv2.LINE_AA)
    else :
         f.write("0\n")
         cv2.putText(frame, 'Not Fall!!!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 0 , 255), 2, cv2.LINE_AA)
##################################################################################
    avg = 0 
    worksheet.write(row, col,   count)
    worksheet.write(row, col + 1, ang_val)
    worksheet.write(row, col + 2, axis_val)
    worksheet.write(row, col + 3, float(mhi_cof))
    worksheet.write(row, col + 4 , float(w/h) )
    row += 1 
   # cv2.imwrite("mhi/frame%d.jpg" % count ,frame ) 
   # cv2.imwrite("mhii/frame%d.jpg" % count, bg ) 
    #cv2.imwrite("mhiii/frame%d.jpg" % count, frami) 
    #cv2.imwrite("F:/out2/chute(3)/cam3/bin/frame%d.jpg" % count ,fgmask )
    #cv2.imwrite("F:/out2/chute(3)/cam3/mhi/frame%d.jpg" % count ,mh )
    out.write(frame)
    cv2.imshow("Frame", frame)
    #cv2.imshow("bl",threshold  )
    cv2.imshow("Mask", fgmask)
    #cv2.imshow("Masaaak", fggmask)
    #cv2.imshow("Maskold",  mh)
    #cv2.imshow("Ma", mask)

    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
workbook.close()
plt.figure(figsize=(50,10))
#plotting the points  
plt.plot(xo, yo, label = "C-Motion") 
plt.plot(xo, yan, label = "s?")   
plt.plot(xo, yaxi, label = "s?")  
plt.plot(xo, ywid, label = "Width/Height")  

# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
##plt.title('My first graph!') 
plt.legend()  
# function to show the plot 
plt.savefig('F:/out2/chute(3)/myfisg')
plt.show()
f.close()
print(yo)
cap.release()
cv2.destroyAllWindows()