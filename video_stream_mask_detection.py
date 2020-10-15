import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

def predict_mask(frame, faceNN, maskNN):
	(height, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# obtaining face detections after passing blob through the network
	faceNN.setInput(blob)
	detections = faceNN.forward()

	# list of faces, their locations and the predictions after applying the neural net
	face_list = []
	loc_list = []
	pred_list = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# Using confidence intervals to filter out weak detections
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(width - 1, endX), min(height - 1, endY))

			# extracting and preprocessing face ROI
			face_roi = frame[startY:endY, startX:endX]
			face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
			face_roi = cv2.resize(face_roi, (224, 224))
			face_roi = img_to_array(face_roi)
			face_roi = preprocess_input(face_roi)
			face_list.append(face_roi)
			loc_list.append((startX, startY, endX, endY))

	#prediction is made only if a face is detected
	if len(face_list) > 0:
		face_list = np.array(face_list, dtype="float32")
		pred_list = maskNN.predict(face_list, batch_size=32)
	return (loc_list, pred_list)

# constructing the argument parser
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
arg_parse.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
arg_parse.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(arg_parse.parse_args())

# load our serialized face detector model from disk
protoPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNN = cv2.dnn.readNet(protoPath, weightsPath)

# load the detector model
maskNN = load_model(args["model"])

# initializing the video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

#going through each frame in the video
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in a frame and determine if they're wearing a mask
	(locs, preds) = predict_mask(frame, faceNN, maskNN)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
