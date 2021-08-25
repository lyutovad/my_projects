import matplotlib.pyplot as plt
import numpy as np
import argparse
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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# param train
INIT_LR = 1e-4
EPOCHS = 10
#EPOCHS = 20
#EPOCHS = 40
BS = 32

# load images
print("loading images for train")
imagesPath = list(paths.list_images(args["dataset"]))
data_files = []
labels_files = []

for imagePath in imagesPath:
	# name_class
	label = imagePath.split(os.path.sep)[-2]

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data_files.append(image)
	labels_files.append(label)

data_files = np.array(data_files, dtype="float32")
labels_files = np.array(labels_files)
#///
lb = LabelBinarizer()
labels_files = lb.fit_transform(labels_files)
labels_files = to_categorical(labels_files)

(trainX, testX, trainY, testY) = train_test_split(data_files, labels_files,
	test_size=0.20, stratify=labels_files, random_state=42)

aug_generate = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 модель
baseModelLearning = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


headModelLearning = baseModelLearning.output
headModelLearning = AveragePooling2D(pool_size=(7, 7))(headModelLearning)
headModelLearning = Flatten(name="flatten")(headModelLearning)
headModelLearning = Dense(128, activation="relu")(headModelLearning)
headModelLearning = Dropout(0.5)(headModelLearning)
headModelLearning = Dense(2, activation="softmax")(headModelLearning)

modelLearning = Model(inputs=baseModelLearning.input, outputs=headModelLearning)

# fix train, not learning first layers
for layer in baseModelLearning.layers:
	layer.trainable = False

print("compiling_train model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
modelLearning.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train
print("training head")
H = modelLearning.fit(
	aug_generate.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("save model")
modelLearning.save(args["model"], save_format="h5")
#model.save(args["model"])



# view train
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
