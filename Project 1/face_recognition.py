# import the necessary packages
import ssl
import urllib
from imutils.video import FPS
import numpy as np
import pickle
import time
import cv2
from face_descriptor import Face_Descriptor

fd = Face_Descriptor()
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("recognizer.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())

print("aaa")
# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.43.1:8080/shot.jpg'

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    # frame = vs.read()

    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    boxo = list(fd.get_detections(img))
    if not boxo is None:
        for box in boxo:
            x,y,w,h = box
    #        if isPositive(box) is False:
            pname = fd.predictFace(cv2.resize(img[y:h,x:w],(224,224)))
            
            # classification to recognize the face
            preds = recognizer.predict_proba(pname.reshape(1,-1))[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            if(proba < 0.875):
                name = "Unknown"
            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            
            
            cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,0),2)
            cv2.rectangle(img, (x,y), (w,h), (0,255,0), 2)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()