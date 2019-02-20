from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import shutil
import time

size_width=64

def rank5_accuracy(preds, labels):
	# initialize the rank-1 and rank-5 accuracies
	rank1 = 0
	rank5 = 0

	# loop over the predictions and ground-truth labels
	for (p, gt) in zip(preds, labels):
		# sort the probabilities by their index in descending
		# order so that the more confident guesses are at the
		# front of the list
		p = np.argsort(p)[::-1]

		# check if the ground-truth label is in the top-5
		# predictions
		if gt in p[:5]:
			rank5 += 1

		# check to see if the ground-truth is the #1 prediction
		if gt == p[0]:
			rank1 += 1

	# compute the final rank-1 and rank-5 accuracies
	rank1 /= float(len(preds))
	rank5 /= float(len(preds))

	# return a tuple of the rank-1 and rank-5 accuracies
	return (rank1, rank5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
args = vars(ap.parse_args())

tt=time.localtime()
timestr=time.strftime('%Y-%m-%d-%H-%M-%S', tt)
output_result=str(timestr)+".csv"

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
c=0
imagePaths = list(paths.list_images(args["input"]))
with open(output_result,'w') as txtWriter:
	txtWriter.write("id,label\n")
	for imagePath in imagePaths:
	# load the image and convert it to grayscale, then pad the image
	# to ensure digits caught only the border of the image are
	# retained
		image = cv2.imread(imagePath)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (size_width, size_width))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		(pred1, pred2) = model.predict(image)[0]

#print("[INFO] predicting...")
	#preds = model.predict_proba(image)[0]
	#(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])
# display the rank-1 and rank-5 accuracies
	#print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
	#print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

		label = "1" if pred1 > pred2 else "2"
		print(imagePath+":"+label+":"+str(pred1)+":"+str(pred2))
		print(str(pred1))
		c=c+1
		txtWriter.write(str(c)+","+label+"\n")
	#shutil.copy2(imagePath,args["input"]+"\\"+label)
txtWriter.close()

	# show the output image
