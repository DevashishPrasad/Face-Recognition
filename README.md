# Face-Recognition
This repository contains codes for face recognition project (Automatic attendance system)
<br>
I implemented two approaches for face recognition in <b>PYTHON</b>
<br>
<br>
Project1 folder has implementation of approach 1<br>
Project2 folder has implementation of approach 2<br>

I will recommend to use the Project 2 

## Project 1
1) In this approach a Resnet_10 (10 conv layers) with SSD (Single Shot Detector) is used to detect the faces from the image. It is very fast at detecting the faces with descent accuracy. 

2) The output of this Resnet_10_SSD (Detected face) is then given to the next CNN VGG_Face_net (16 conv layers). This generates a feature embedding of shape (1,2622). 

3) Finally these embeddings are used to train any ML classifier (SVM here).

4) Each Identity (person) can have one or more embeddings. For registering the face, the embeddings are extracted for all the identities and then SVM classifier is trained over them. At the test time, the embedding of the subject (person) is calculated by CNN and SVM predicts the class (indentity of person). Confidence thresholds are used to determine unknow identities.  

Below is the project structure:
<br>
<img src="Project 1/Arch.png" alter="No preview available" />
<br>

File | Description
--- | --- 
New_User_Images  			|  takes images for a new user and stores into dataset folder
Face_descriptor  			|  detects the face and generates feature embedding for any given image
New_Embeddings_Extractor  	|  it extracts feature embeddings of newly registered user and saves these embeddings in a pickle file
Embeddings 					|  it is the pickle file saved by New_Embeddings_Extractor
Classification_Model  		|  it is the ML SVM model that will be trained on new embeddings and saved as Model_pickle
Model_pickle  				|  trained ML model
Label_encoder.pickle  		|  label encoder saved
face_recognition  			|  putting everything in one place

This approach worked well but did not attain a high accuracy. As a result the 2nd approach was implemented.

## Project 2
In this approach I used dlib library of opencv for face recognition. This was very simple as well as gave a high accuracy results. For more informarion on the library visit [link](https://face-recognition.readthedocs.io/en/latest/readme.html). I also implemented the log file for storing in time and out time for attendace system. The basic working and structure is same as the project 1. But the model backbones are optimized, as a result yeilded better results.

File | Description
--- | --- 
recognize_faces.py 	| takes image as input and returns the label
encode_faces.py 	| creates face encodings and pickles it
main.py 			| runs the application using web camera
ipmain.py 			| runs the application using ip camera
dataset				| directory that stores images

How to use:
1) Run the script (main.py or ipmain.py) 
2) Press R to Register a new Face, after that type the name of the person to be registered
3) Press P to get Prediction of the current face in the frame. 

