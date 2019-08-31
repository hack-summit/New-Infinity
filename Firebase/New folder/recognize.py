
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to input image")
ap.add_argument("-d", "--detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer",
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le",
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
os.system("conda activate opencv-env")

print("[INFO] loading face detector...")
people=[]
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickles", "rb").read())


image = cv2.imread("images/a.jpg")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)


detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
	
	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:
	
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		if fW < 20 or fH < 20:
			continue

		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]
		people.append(name)
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.imshow("Image", image)
print(name)
cv2.waitKey(0)