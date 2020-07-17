# adapted from https://github.com/gracelynxs/malaria-detection-model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import resnet
import config
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# define the # of epochs ( 50 is good)
#initial learning rate ( 0.1 is good) and batch size
num_epochs = 1
init_lr= 1e-1
bs = 32

# create a function called polynomial decay which helps us decay our
# learning rate after each epoch

def poly_decay(epoch):
	# initialize the maximum # of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 1.0  # turns our polynomial decay into a linear decay

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

# determine the # of image paths in training/validation/testing directories
totalTrain = len(list(paths.list_images(config.train_path)))
totalVal = len(list(paths.list_images(config.val_path)))
totalTest = len(list(paths.list_images(config.test_path)))


# initialize the training data augmentation object
# randomly shifts, translats, and flips each training sample
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)


# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.train_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=bs)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.val_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.test_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)


# initialize our ResNet model and compile it
model = resnet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // bs,
	validation_data=valGen,
	validation_steps=totalVal // bs,
	epochs=num_epochs,
	callbacks=callbacks)


# reset testing generator and use the trained model to make predictions
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // bs) + 1)

# finds the index of the label with largest predicted probability of each
# testing image

predIdxs = np.argmax(predIdxs, axis=1)

# displays a classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
