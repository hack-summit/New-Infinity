# New-Infinity
Smart cities
# MARK ME!: The most intelligent attendance system- designed for startups,schools and developing countries

It has 2 components:
          1)Android app- To record the attendances using NFC, QR and Face Recognition
          2)Web based dashboard- To view and manage attendances
How to use the code?
First, extract the embeddings from the face recognition data set using:
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
This then stores the embeddings in a python pickle file for future use.

Next, train the model using:
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickles

Then, use the below code to detect whether the given face matches with the database:
python recognize.py

NOTE:1)To use other faces for detection, change the path to the image file at ln:24, in recognize.py
     2)The model currently labels unknown images randomly, we are working on a fix to it.
     
NOTE:
    Execute "python flsk.py", and go to http://127.0.0.1:5000/AI to get into the Attendance dashboard
 
