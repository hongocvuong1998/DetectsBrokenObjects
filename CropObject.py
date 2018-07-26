def thresh_callback(val):
    threshold = val
    
    canny_output = cv2.Canny(framegray, threshold, threshold * 2)
    
    
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
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        x1=int(boundRect[i][0])
        y1=int(boundRect[i][1])
        x2=int(boundRect[i][0]+boundRect[i][2])
        y2=int(boundRect[i][1]+boundRect[i][3])
        
        if x1 <x_min:
            x_min=x1
        if y1<y_min :
            y_min=y1
        if x2>x_max :
            x_max=x2
        if y2>y_max:
            y_max=y2
        
        #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    Point1=(x_min,y_min)
    Point2=(x_max,y_max)
    cv2.rectangle(drawing, Point1, Point2, color, 2)
    cv2.imshow('Contours', drawing)
    return Point1, Point2
    
font = cv2.FONT_HERSHEY_SIMPLEX
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



max_thresh = 255
thresh = 100 # initial threshold


source_window = "Source"


frame=cv2.imread("Namefile")  #Hình chưa crop

frame=crop_center(frame,972,1728)
frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
framegray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret,framegray = cv2.threshold(framegray,120,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
framegray = cv2.erode(framegray,kernel,iterations = 1)
cv2.imshow('mask',framegray)
Point1,Point2 = thresh_callback(150)

x1,y1=Point1
x2,y2=Point2

img= frame[y1:y2,x1:x2]

cv2.imshow('img', img)  # Hình đã crop


