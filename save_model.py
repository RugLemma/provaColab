# USAGE
# py save_model.py --model saved_model.model
# py save_model.py --model name_model.model
# name_model.model must be the same that you have to insert in the command when you execute the load_model.py script


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import numpy as np


# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.optimizers import Adam
from pyimagesearch.resnet import ResNet
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from pyimagesearch import config,SGDres,LRF,EarlystopModi,EarlyStopLrValure
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt


import argparse

#per creare il modello salvato
ap = argparse.ArgumentParser()
'''
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path dataset of input images")
'''
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-j", "--jesimo", type=str)
args = vars(ap.parse_args())
print(args)
# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 200
#NUM_EPOCHS = 100
#NUM_EPOCHS = 50
#NUM_EPOCHS = 20
#NUM_EPOCHS = 15
#NUM_EPOCHS = 5
#NUM_EPOCHS = 11
#NUM_EPOCHS = 1
INIT_LR = 1e-1
BS = 32
#BS = 64


def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the training training data augmentation object
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
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	#target_size=(128, 128),  da errore perchè l'input che vuole la rete è 64x64
	color_mode="rgb",
	#shuffle=True,
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


# initialize our ResNet model and compile it
'''
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
'''
input_tensor = Input(shape=(224, 224, 3))
#model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True, classes = 2)
model = InceptionV3(input_tensor=input_tensor,weights=None ,include_top=True, classes = 2)
opt = SGD(lr=INIT_LR, momentum=0.9)
#opt = SGD(lr=INIT_LR, momentum=0.9, nesterov= True )
#opt = Adam(lr=0.001,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)
#opt = Adam(lr=1e-3,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)
'''
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
'''
#model.compile(loss="binary_crossentropy", metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
'''
for lay in model.layers:
	print(lay.name)
	print(lay.get_weights())
'''
# define our set of callbacks and fit the model
#callbacks = [LearningRateScheduler(poly_decay)]
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
#es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=30)
#hi = History()
#lrf = LRF.LRFinder(min_lr=1e-5, max_lr=INIT_LR, steps_per_epoch=totalTrain // BS, epochs=3)
#sgdr = SGDres.SGDRScheduler(min_lr=0.001, max_lr=INIT_LR,steps_per_epoch=totalTrain // BS, lr_decay=0.8, cycle_length=5, mult_factor=1.5 )
esm = EarlystopModi.EarlyStoppingM(monitor='val_acc', mode= 'max', patience=40,  min_lr=1e-5 , max_lr=1e-2,steps_per_epoch=totalTrain // BS, lr_decay=0.8, cycle_length=20, mult_factor=1.5)
#esmlr = EarlyStopLrValure.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=40)

H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=[esm]
	#callbacks=[lrf]
	#callbacks=[sgdr]
	#callbacks=[es,sgdr]
	#callbacks=callbacks
)
'''
new_lr = lrf.clr()
print(new_lr)
'''

'''
lrf.plot_loss()
lrf.plot_lr()
'''

# reset the testing generator and then use our trained model to
# make predictions on the data

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)


valore= np.argmax(predIdxs, axis=1)
print(valore)
# show a nicely formatted classification report
print(classification_report(testGen.classes, valore,
	target_names=testGen.class_indices.keys()))

# plot the training loss and accuracy
if(esm.stopped_epoch!=0):
	N = esm.stopped_epoch+1
else:
	N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
plt.legend(loc="best")
plt.savefig(args["plot"])


print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])