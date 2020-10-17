
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

# construct the argument parser
arg_Parse = argparse.ArgumentParser()
arg_Parse.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
arg_Parse.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(arg_Parse.parse_args())

# initialize LR, BS and epochs
Learn_Rate_Init = 1e-4
epochs = 25
batch_size = 30

# initialize list of images and labels
im_path = list(paths.list_images(args["dataset"]))
images = []
classes = []

# loop over the image paths
for path in im_path:
	# extract the class label
	label = path.split(os.path.sep)[-2]
	image = load_img(path, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	images.append(image)
	classes.append(label)

images = np.array(images, dtype="float32")
classes = np.array(classes)

#one-hot encoding
lb = LabelBinarizer()
classes = lb.fit_transform(classes)
classes = to_categorical(classes)

(X_train, X_test, Y_train, Y_test) = train_test_split(images, classes,
	test_size=0.20, stratify=classes, random_state=42)

augment = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load MobileNetV2 network
baseMod = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model
headMod = baseMod.output
headMod = AveragePooling2D(pool_size=(7, 7))(headMod)
headMod = Flatten(name="flatten")(headMod)
headMod = Dense(128, activation="relu")(headMod)
headMod = Dropout(0.5)(headMod)
headMod = Dense(2, activation="softmax")(headMod)

#place it on the Base of the model

model = Model(inputs=baseMod.input, outputs=headMod)

# Freeze layers in the base model so they will not be updated during the first training process
for layer in baseMod.layers:
	layer.trainable = False

# compile model
opt = Adam(lr=Learn_Rate_Init, decay=Learn_Rate_Init / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train NN Head
H = model.fit(
	augment.flow(X_train, Y_train, batch_size=batch_size),
	steps_per_epoch=len(X_train) // batch_size,
	validation_data=(X_test, Y_test),
	validation_steps=len(X_test) // batch_size,
	epochs=epochs)

# make predictions on the testing set
pred = model.predict(X_test, batch_size=batch_size)
pred = np.argmax(pred, axis=1)

# show a nicely formatted classification report
print(classification_report(Y_test.argmax(axis=1), pred,
	target_names=lb.classes_))

# serialize the model to disk
model.save(args["model"], save_format="h5")
