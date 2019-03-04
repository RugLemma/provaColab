# USAGE
# py load_model.py -m saved_model.model
# py load_model.py -m name_model.model
# name_model.model must be the same of the model saved in the folder after the execution of save_model.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import time
import datetime
import json

import numpy as np

#import os
'''
import random as rn
import tensorflow as tf
'''

'''
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2017)

np.random.seed(7)
rn.seed(7)


# Limit operation to 1 thread for deterministic results.
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
'''

#from keras import backend as K
'''
tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''

# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from pyimagesearch import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import os.path
from prettytable import PrettyTable

'''
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
with tf.Session() as sess:
    print (sess.run(c))
'''

import argparse
'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
'''
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# define the total number of epochs to train for along with the
# initial learning rate and batch size
#NUM_EPOCHS = 50
#NUM_EPOCHS = 20
#NUM_EPOCHS = 2
NUM_EPOCHS = 1
INIT_LR = 1e-1
BS = 32
#BS = 64

'''
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
'''

# testing directories

totalTest = len(list(paths.list_images(config.TEST_PATH)))


# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


def save_resu():
	timestamp = time.time()
	str_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y_%m_%d %H_%M_%S')
	dir_name = "Report/" + str_time
	os.mkdir(dir_name)
	#data['Precision'] = precision
	data['Accuracy'] = accuracy
	data['Recall'] = recall
	data['Specificity'] = specificity
	json_data = json.dumps(data)
	path_file = dir_name + "/metrics.json"
	path_f = dir_name + "/info.txt" #In this file I want to write the fp, fn and other info
	file = open(path_file, "w")
	f_files = open(path_f, "w")
	info = str(x) +"\n False Positive("+str(fp)+"):\n"+str(fp_name)+"\n False negative("+str(fn)+"):\n"+str(fn_name)
	f_files.write(info)
	file.write(json_data)
	file.close()


# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)
'''
print(len(predIdxs))
print(predIdxs)
'''
valore = np.argmax(predIdxs, axis=1)
#print(valore)
tn = fn = tp = fp = 0
tn_name = fp_name = tp_name = fn_name = ""

for i in range(0,len(predIdxs)):

	iesima = np.argmax(predIdxs[i])
	image_name = testGen.filenames[i]
	tipo=os.path.dirname(image_name)
	#print(tipo)
	if (tipo=="Negativi"):
		if iesima==0:
			tn=tn+1
			#tn_name=tn_name + image_name + "\n"
		if iesima==1: 
			fp=fp+1
			fp_name=fp_name + image_name + "\n" + str(iesima) + "\n" + str(predIdxs[i]) +"\n" 
	if tipo=="Positivi":
		if iesima==1:
			tp=tp+1
			#tp_namee=tp_name + image_name + "\n"
		if iesima==0:
			fn=fn+1
			fn_name=fn_name + image_name + "\n" + str(iesima) + "\n" + str(predIdxs[i]) +"\n"
	'''
	print(image_name)
	print(iesima)
	print(predIdxs[i])
	'''

# show a nicely formatted classification report
print(classification_report(testGen.classes, valore,
	target_names=testGen.class_indices.keys()))

print(tp)
print(tn)
print(fn)
print(fp)

x = PrettyTable()

x.field_names = ["Confusion Matrix", "Positivi", "Negativi"]

x.add_row(["Positivi", tp, fp])
x.add_row(["Negativi", fn, tn])
print('False Positive')
print(fp_name)
print('False Nagative')
print(fn_name)

print(x)

accuracy = (tp+tn)/(tp+tn+fn+fp)
print("Accuracy: " + str(accuracy))
precision = tp / (tp+fp)
print("Precion: " + str(precision))
recall = tp / (tp + fn)
print("Recall: " + str(recall))
specificity = (tn)/(tn + fp)
print("Specificity: " + str(specificity))
data = {}


if os.path.exists("Report"):
	save_resu()

else:
	os.mkdir("Report")
	save_resu()
# plot the training loss and accuracy
'''
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
plt.legend(loc="lower left")
plt.savefig(args["plot"])
'''

