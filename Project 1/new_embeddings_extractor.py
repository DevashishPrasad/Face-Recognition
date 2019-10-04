# import the necessary packages
from imutils import paths
import pickle
import cv2
import os
from face_descriptor import Face_Descriptor

fd = Face_Descriptor()

def extract_new_embeddings():
    
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("dataset"))
    
    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    
    knownEmbeddings = []
    knownNames = []
    labels = []
    
    if os.path.exists('face_features.pickle'):
        # load the face embeddings
        print("[INFO] loading face embeddings...")
        knownEmbedding = pickle.loads(open("face_features.pickle", "rb").read())
        knownEmbeddings = knownEmbedding["embeddings"]
        # encode the labels
        print("[INFO] encoding labels...")
        labels = knownEmbedding["names"]
        knownNames = list(labels)
    
    # initialize the total number of faces processed
    total = 0
    
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        if name in labels:
            print(name)
            continue
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        img = cv2.imread(imagePath)
        box = fd.get_detections(img)
        if(len(box) == 0):
            continue
        x,y,w,h = box[0]
        print(box)
        face = fd.predictFace(cv2.resize(img[y:h,x:w],(224,224)))
        # add the name of the person + corresponding face
        # embedding to their respective lists
        knownNames.append(name)
        knownEmbeddings.append(face)
        total += 1
    
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("face_features.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()