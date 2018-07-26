import cv2
PathVideo = 'D:\\DataStoreData\\Video\\4.mp4'




cap = cv2.VideoCapture(PathVideo)
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('Output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 40, (512,512))


while(cap.isOpened()):
    ret, frame = cap.read()

    #framegray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Covert color image to gray image
    

    y = model(frame[None, ...])
    result= y.data.argmax(axis=1)[0]
    strout=''
    if result==0:
        strout='DAMAGE'
    else:
        strout='NORMAL'
   
    cv2.putText(frame,strout,(10,500), font, 1,(0,0,255),2) #Draw the text

    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()