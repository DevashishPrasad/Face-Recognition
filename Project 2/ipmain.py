import cv2
from recognize_faces import recognize
from encode_faces import encode
import os
import urllib.request
import numpy as np
import ssl
from collections import deque
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.43.1:8080/shot.jpg'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Path exists")
        else:
            print("Exception")

# cap = cv2.VideoCapture(0)
status='in'

while(True):
    # Capture frame-by-frame
    # ret, frame = cap.read()

    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    frame = cv2.resize(frame,(700,400))

    # Display the resulting frame
    cv2.imshow('frame',frame)

    key = cv2.waitKey(1)
    
    if  key == ord('p'):
        name = recognize(frame)[0]
        print(name)
        if len(name) != 0:
            file=open('inout.txt','a+')
            if os.path.getsize('inout.txt') is 0:
                file.writelines([name+str(i)+','+"in:"+str(datetime.now().time())+"\n"])
            else:
                file.seek(0)
                for record in file.readlines():            
                    if(record.split(',')[0] == name):
                        if(record.split(',')[1].split(':')[0] == 'in'):
                            status = 'out'
                        else:
                            status = 'in'
            file.writelines([name+','+status+":"+str(datetime.now().time())+"\n"])        
            file.close()                            
    elif key == ord('r'):
        name = input("Enter the name - ")
        path = 'dataset/' + str(name)
        mkdir_p(path)
        cv2.imwrite("dataset/" + str(name) + "/" + str(name) + ".jpg", frame)
        encode()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
