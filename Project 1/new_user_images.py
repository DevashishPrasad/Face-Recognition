import time
import cv2
import numpy as np
import ssl
import urllib
import os
import errno    


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Path exists")
        else:
            print("Exception")

def save_images(img,count,name):
    cv2.imwrite("dataset/" + str(name) + "/" + str(name) + str(count) + ".jpg", cv2.resize(img,(1280,720)))

def get_images():
    name = input("\n Please enter your name - ")
    path = 'dataset/' + str(name)
    mkdir_p(path)
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    url = 'http://192.168.43.1:8080/shot.jpg'
    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    
    # Initialize individual sampling face count
    count = 0
    controller = 0
    
    while(True):
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        img2 = img
        
        cv2.putText(img2,"Processing for user - " + name,(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        
        if(controller == 0):
            cv2.putText(img2,"Press A and Look in the camera",(40,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
            
            if cv2.waitKey(1) == ord('a'):
                for x in range(10):
                    time.sleep(0.2)
                    imgResp = urllib.request.urlopen(url)
                    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    save_images(img, count, name)
                    count+=1
                controller+=1
                print("\n [INFO] Task one completed successfully")
    
        if(controller == 1):
            cv2.putText(img2,"Press A and perform Right to Left",(40,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
            
            if cv2.waitKey(1) == ord('a'):
                for x in range(10):
                    time.sleep(0.2)
                    imgResp = urllib.request.urlopen(url)
                    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    save_images(img, count, name)
                    count+=1
                controller+=1
                print("\n [INFO] Task two completed successfully")
    
    
        if(controller == 2):
            cv2.putText(img2,"Press A and perform Up to Down",(40,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
            
            if cv2.waitKey(1) == ord('a'):
                for x in range(10):
                    time.sleep(0.2)
                    imgResp = urllib.request.urlopen(url)
                    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    save_images(img, count, name)
                    count+=1
                controller+=1
                print("\n [INFO] Task three completed successfully")
    
    
        if(controller == 3):
            cv2.putText(img2,"Press A and perform Rotation",(40,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
            
            if cv2.waitKey(1) == ord('a'):
                for x in range(20):
                    time.sleep(0.2)
                    imgResp = urllib.request.urlopen(url)
                    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    save_images(img, count, name)
                    count+=1
                controller+=1
                print("\n [INFO] Task four completed successfully")
    
        cv2.imshow("Frame", cv2.resize(img2,(1280,720)))
    
        if(controller >= 4):
            break
    
    print("\n [INFO] All tasks completed successfully")
    print("\n [INFO] Total " + str(count) + " images saved")
    print("\n [INFO] Exiting Program")
    cv2.destroyAllWindows()
    
