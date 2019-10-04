import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.models import Model,Sequential
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D,Dropout,Flatten,Activation


class Face_Descriptor:
    def __init__(self):
        print("[INFO] Loading the model...")
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        model.load_weights('vgg_face_weights.h5')
        self.vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        self.detector = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')

    def get_detections(self,img):
        (h, w) = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        box = None
        boxo = []
        # loop over the detections   
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
         
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
        
            if confidence > 0.7:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = box.astype('int')
                if(box is None or self.isPositive(box) is False):
                    continue
                boxo.append(box)
                
        return boxo
        
        
    
    def preprocess_image(self,img):
        img = np.array(np.expand_dims(img, axis=0),dtype = float)
        img = preprocess_input(img)
        return img
    
    def predictFace(self,img1):
        return self.vgg_face_descriptor.predict(self.preprocess_image(img1))[0,:]
        
    
    def isPositive(self,box):
        x,y,w,h = box
        if x<0 or x > 1280:
            return False
        if y<0 or y > 720:
            return False
        if w<0 or w > 1280:
            return False
        if h<0 or h > 720:
            return False
        return True
