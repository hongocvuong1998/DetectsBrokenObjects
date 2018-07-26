from Header import *

lower_blue=np.array([90,110,45])
upper_blue=np.array([130,255,255])


folder_R='D:\\Myfolder\Dataset\\TrainingDataSetScale\\DAMAGE\\'

folder_S='D:\\Myfolder\\Dataset\\TrainingDataSetScale\\DAMAGE_CROP\\'
k=0
#_______________________IF USING DISTANCE____________________________#

def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def Distance(Rect):      #Distance 2 point
    X1,Y1,X2,Y2=Rect
    dx=abs(X2-X1)
    dy=abs(Y2-Y1)
    return math.sqrt(math.pow(dx,2)+math.pow(dy,2))
def GetCenterPoint(Rect): # Return center point of Rect
    x1,y1,x2,y2=Rect
    x=((x2-x1)/2)+x1
    y=((y2-y1)/2)+y1
    return x,y
def ChooseBound(ListRect): # Choose a Bounding for object
    #maxDistance=0
    maxRect=ListRect[0]
    k=0
    while True:
        ListRect.sort(key=Distance,reverse=True) #Sort list rect by Distance
        #for i in range(len(ListRect)):
            # print(i+1,'     ',ListRect[i] ,'   ',Distance(ListRect[i]))  #ListRect[i]=Area[i],  ListRect[i][0]=Area[i].x1
        maxRect=ListRect[k] #Get maxRect in List
        maxDistance=Distance(ListRect[k]) #Get max 

        i=0
        while i < len(ListRect):
            if ListRect[i] != maxRect and ListRect[i][0]>=maxRect[0] and ListRect[i][1]>=maxRect[1] and ListRect[i][2]<=maxRect[2] and ListRect[i][3]<=maxRect[3]: 
                del ListRect[i] #del small Rect in large Rect 
                i=i-1
            i=i+1
        k=k+1
        maxRect=ListRect[0] #Get maxRect affter delete any element
        maxDistance=Distance(ListRect[0]) #Get Distance affter delete any element
        #print('Max Rect',maxRect)
        #print('Max Distance',maxDistance)
        if k==len(ListRect):
            break
    ListRect=remove_duplicates(ListRect)
    # for i in range(len(ListRect)): # Show list point after delete any element
    #     print(i+1,'     ',ListRect[i] ,'   ',Distance(ListRect[i]))
        
    i=0
    while i<len(ListRect): #Choose Rect
        if ListRect[i] != maxRect:
            x1,y1=GetCenterPoint(ListRect[i])
            x2,y2 = GetCenterPoint(maxRect)
            Rect=(x1,y1,x2,y2)
            if Distance(Rect)>10:
                del ListRect[i] #del Rect if it large distance
                i=i-1
        i=i+1
    
    
    if len(ListRect)==1:
        # If List have one element
        Point1=(ListRect[0][0],ListRect[0][1])
        Point2=(ListRect[0][2],ListRect[0][3])
    else:
        #print("Greater than one")
        # print(ListRect[1][0],ListRect[1][1],ListRect[1][2],ListRect[1][3])
        x1=[]
        y1=[]
        x2=[]
        y2=[]
        for i in range(len(ListRect)):
            x1.append(ListRect[i][0])
            y1.append(ListRect[i][1])
            x2.append(ListRect[i][2])
            y2.append(ListRect[i][3])
        Point1=(min(x1),min(y1))
        Point2=(max(x2),max(x2))
    
    #print(Point1,Point2)
    return Point1,Point2

            
#____________________________________________________________________#

def nothing(x):
  pass

def thresh_callback(val):
    threshold = val
    
    canny_output = cv2.Canny(not_mask, threshold, threshold * 2)
    
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    for i in range(len(contours)):
        contours_poly[i] = cv2.approxPolyDP(contours[i], 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
      
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8) 
    x_min=512  
    y_min=512
    x_max=0
    y_max=0  
    checkcase=True #If True => By area
                    #If False => By Distance

    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        x1=int(boundRect[i][0])
        y1=int(boundRect[i][1])
        x2=int(boundRect[i][0]+boundRect[i][2])
        y2=int(boundRect[i][1]+boundRect[i][3])

        if (x2-x1)*(y2-y1)<1000:
            #print("Smaller area")
            continue
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        if checkcase:
            if x1 <x_min:
                x_min=x1
            if y1<y_min : 
                y_min=y1 
            if x2>x_max :
                x_max=x2 
            if y2>y_max:
                y_max=y2 
        if not checkcase:
            Rect=(x1,y1,x2,y2)
            ListRect.append(Rect)
    if checkcase:
        Point1=(x_min,y_min)
        Point2=(x_max,y_max)

    if not checkcase:
        Point1,Point2 = ChooseBound(ListRect)
    
    #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    #cv2.rectangle(drawing, Point1, Point2, color, 2)
    #cv2.imshow('Contours', drawing)
    #cv2.waitKey(0)
    return Point1, Point2
    


def CropObject(img): 
    global not_mask
    #img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_LINEAR) #if large image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    not_mask = cv2.bitwise_not(mask)
    kernel = np.ones((5,5),np.uint8)
    not_mask = cv2.erode(not_mask,kernel,iterations = 1) #Eliminate interference
    #cv2.imshow('not_mask', not_mask)
    Point1,Point2 = thresh_callback(120)
    x1,y1=Point1
    x2,y2=Point2
    img_crop= img[y1:y2,x1:x2].copy() #If not using copy(), it will wrong
    cv2.rectangle(img, Point1, Point2, (0,0,255), 2)
    return img_crop, img

def SaveImageCrop(folderR,folderS): #folderR: folder read image____fordelS: folder save output image after crop 
    global k
    for root, dirs , files in os.walk(folderR):
        for file in files:
            if file.endswith(".jpg"):
                k=k+1
                string=os.path.join(root, file)
                print(string)
                namefile='.jpg'
                img=cv2.imread(string)
                img_crop,img=CropObject(img)
                # cv2.imshow('Img_Crop',img_crop)
                # cv2.imshow('img',img)
                savefile=folderS+str(k)+namefile
                cv2.imwrite(savefile, img)


def BoundingBoxVideo(Linkfile):
    cap = cv2.VideoCapture(Linkfile)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_crop,frame=CropObject(frame)
        cv2.imshow('video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

   

if __name__ == '__main__':
    SaveImageCrop(folder_R,folder_S)







