#adapted from https://github.com/gracelynxs/malaria-detection-model
#setup
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

def residual_module(data, K, stride, chanDim, red=False,
	reg=0.0001, bnEps=2e-5, bnMom=0.9):
	#the shortcut branch initialized as the input (identity) data
	shortcut = data

	#ResNet module block 1 (1x1 CONVs)
	bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(data)
	act1 = Activation("relu")(bn1)
	conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
		kernel_regularizer=l2(reg))(act1)

	#ResNet module block 2 (3x3 CONVs)
	bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(conv1)
	act2 = Activation("relu")(bn2)
	conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
		padding="same", use_bias=False,
		kernel_regularizer=l2(reg))(act2)

	#ResNet module block 3 (1x1 CONVs)
	bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(conv2)
	act3 = Activation("relu")(bn3)
	conv3 = Conv2D(K, (1, 1), use_bias=False,
		kernel_regularizer=l2(reg))(act3)

	# if we are to reduce the spatial size, apply a CONV layer to
	# the shortcut
	if red:
		shortcut = Conv2D(K, (1, 1), strides=stride,
			use_bias=False, kernel_regularizer=l2(reg))(act1)

	# add together the shortcut and the final CONV
	x = add([conv3, shortcut])

	# return the addition as the output of the ResNet module
	return x

def build(width, height, depth, classes, stages, filters,
	reg=0.0001, bnEps=2e-5, bnMom=0.9):
	# initialize the input shape to be "channels last" and the
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	# set the input and apply BN
	inputs = Input(shape=inputShape)
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(inputs)

	# apply CONV => BN => ACT => POOL to reduce spatial size
	x = Conv2D(filters[0], (5, 5), use_bias=False,
		padding="same", kernel_regularizer=l2(reg))(x)
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(x)
	x = Activation("relu")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# loop over the number of stages
	for i in range(0, len(stages)):
		# initialize the stride, then apply a residual module
		# used to reduce the spatial size of the input volume
		stride = (1, 1) if i == 0 else (2, 2)
		x = residual_module(x, filters[i + 1], stride,
			chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

		# loop over the number of layers in the stage
		for j in range(0, stages[i] - 1):
			# apply a ResNet module
			x = residual_module(x, filters[i + 1],
				(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

	# apply BN => ACT => POOL
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(x)
	x = Activation("relu")(x)
	x = AveragePooling2D((8, 8))(x)

	# softmax classifier
	x = Flatten()(x)
	x = Dense(classes, kernel_regularizer=l2(reg))(x)
	x = Activation("softmax")(x)

	# create the model
	model = Model(inputs, x, name="resnet")

	# return the constructed network architecture
	return model
