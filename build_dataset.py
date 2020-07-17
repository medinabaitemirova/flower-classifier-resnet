# adapted from https://github.com/gracelynxs/malaria-detection-model
#setup
import config
from imutils import paths
import random
import shutil
import os

#shuffling all the images and splitting into training/testing/validation
imagePaths = list(paths.list_images(config.orig_input_dataset))
random.seed(30)
random.shuffle(imagePaths)
i = int(len(imagePaths) * config.train_split)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]
i = int(len(trainPaths) * config.val_split)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# defining training/testing/validation datasets
datasets = [
	("training", trainPaths, config.train_path),
	("validation", valPaths, config.val_path),
	("testing", testPaths, config.test_path)
    ]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print(" 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image and its class label
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2]

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print(" 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct the path to the destination image and then copy
		# the image itself
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)
