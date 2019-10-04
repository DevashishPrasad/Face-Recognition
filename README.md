# Face-Recognition
This repository contains codes for face recognition project (Automatic attendance system)
<br>
I implemented two approaches for face recognition
<br>
<br>
Project1 folder has implementation of approach 1<br>
Project2 folder has implementation of approach 2<br>

## Project 1
In this approach I first used a res_net_ssd to detect the face from the image. it was very fast at detecting the faces with descent accuracy. The output of this CNN (Detected face) is then given to another CNN VGG_Face_net. This generates a feature embedding of shape (1,2622). Finally these embeddings can be used to train any ML classifier (SVM here).

<br>
<img src="Arch.png" alter="No preview available" />
![No preview available](https://github.com/DevashishPrasad/Face-Recognition/Project 1/Arch.png "System architecture")
<br>
File | Description
--- | --- 
New_User_Images | takes images for a new user and stores into dataset folder
Face_descriptor | detects the face and generates feature embedding for any given image
New_Embeddings_Extractor | it extracts feature embeddings of newly registered user and saves these embeddings in a pickle file
Embeddings | it is the pickle file saved by New_Embeddings_Extractor
Classification_Model | it is the ML SVM model that will be trained on new embeddings and saved as Model_pickle
Model_pickle | trained ML model
Label_encoder.pickle | label encoder saved
face_recognition | putting everything in one place

This approach worked well but did not attain a high accuracy. As a result I implemented the 2nd approach.

## Project 2
In this approach I used dlib library of opencv for face recognition. This was very simple as well as gave a high accuracy results. For more informarion on the library visit [link](https://face-recognition.readthedocs.io/en/latest/readme.html). I also implemented the log file for storing in time and out time for attendace system.

File | Description
--- | --- 
recognize_faces.py 	| takes image as input and returns the label
encode_faces.py 	| creates face encodings and pickles it
main.py 			| runs the application using web camera
ipmain.py 			| runs the application using ip camera
dataset				| directory that stores images
inout.py			| useless

After running the script press P to get prediction and press R to register for 
new face, after pressing R type the name of the person to be registered on cmd 
line.

