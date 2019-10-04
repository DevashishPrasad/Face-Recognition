# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import new_user_images as nwi
import new_embeddings_extractor as extractor

no = input("Enter the number of persons to register - ")
for i in range(int(no)):
    nwi.get_images()
    

extractor.extract_new_embeddings()

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("face_features.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 2622-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
clf = SVC(C=1.0, kernel="linear", probability=True)
clf.fit(data["embeddings"], labels)
print(clf.score(data["embeddings"], labels))

# write the actual face recognition model to disk
f = open("recognizer.pickle", "wb")
f.write(pickle.dumps(clf))
f.close()

# write the label encoder to disk
f = open("le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
