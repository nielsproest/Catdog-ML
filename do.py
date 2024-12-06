import os, random, time
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

#Train generator
train_datagen = ImageDataGenerator(
	rescale=1./255.,
	rotation_range=40,
	shear_range=0.2,
	zoom_range=0.2
)
train_folder = os.path.join("catdog_data", "train")
train_generator = train_datagen.flow_from_directory(
	train_folder,
	target_size=(200, 200),
	shuffle=True
)

#Validation generator
valid_datagen = ImageDataGenerator()
validation_folder = os.path.join("catdog_data", "validation")
valid_generator = train_datagen.flow_from_directory(
	validation_folder,
	target_size=(200, 200),
	shuffle=True
)

#Test generator
test_datagen = ImageDataGenerator()
test_folder = os.path.join("catdog_data", "test")
test_generator = train_datagen.flow_from_directory(
	test_folder,
	target_size=(200, 200)
)

from tensorflow import device
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.regularizers import L2

#default is 1E-3
learning_rate = 1E-5 #1E-3 -> 1E-6 -> 1E-7

#Unused
def prepare_vgg16():
	#VGG16
	model = Sequential()
	model.add(Conv2D(64,kernel_size=(3,3),activation="relu",input_shape=(224, 224, 3)))
	model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))

	model.add(MaxPooling2D((2, 2), strides=2))
	model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))

	model.add(MaxPooling2D((2, 2), strides=2))
	model.add(Conv2D(256,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(256,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(256,kernel_size=(3,3),activation="relu"))

	model.add(MaxPooling2D((2, 2), strides=2))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))

	model.add(MaxPooling2D((2, 2), strides=2))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))
	model.add(Conv2D(512,kernel_size=(3,3),activation="relu"))

	model.add(MaxPooling2D((2, 2), strides=2))
	model.add(Flatten())
	model.add(Dense(4096, activation="relu"))
	model.add(Dropout(rate=0.2))
	model.add(Dense(4096, activation="relu"))
	model.add(Dropout(rate=0.2))
	model.add(Dense(2, activation="softmax"))
	opt = Adam(learning_rate=learning_rate) 
	model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
	return model

def prepare_shorty():
	model = Sequential()
	model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2), strides=2))

	model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=2))

	model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=2))

	model.add(Flatten())
	model.add(Dense(512, activation="relu")) #, kernel_regularizer=L2(l2=0.01)
	model.add(Dropout(rate=0.5))
	model.add(Dense(2, activation="softmax"))
	opt = Adam(learning_rate=learning_rate)
	model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
	return model

def test():
	"""model = load_model("result.hdf5")
	predict_x=model.predict(test_generator)
	classes_x=predict_x.argmax(axis=-1)
	for i,file in enumerate(test_generator.filenames):
		pred = ["cats","dogs"][classes_x[i]]
		if (file.startswith(pred)):
			print("Correct predict: ", file, predict_x[i])
		else:
			print("Wrong predict: ", file, predict_x[i])"""

def finalize():
	#Retrain on validation and test data until same loss and acc
	"""firstRun = True
	old_loss = history.history["val_loss"]
	while (history.history["val_loss"] > old_loss or firstRun):
		firstRun = False
		history = train(1,data=test_datagen)"""

def train():
	model = prepare_shorty()

	from keras import callbacks
	earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
											mode ="min", patience = 5, 
											restore_best_weights = True)

	def train(epochs,data=train_generator):
		return model.fit(
			train_generator,
			validation_data=valid_generator,
			epochs=epochs,
			batch_size=128, #default is 32
			shuffle=True,
			callbacks =[earlystopping]
		)
	history=train(100)

	model.save("result.hdf5")

	score = model.evaluate(test_generator)
	loss,acc = score[0],score[1]
	print("Test loss:", loss)
	print("Test accuracy:", acc)

	# summarize history for accuracy
	plt.plot(history.history["accuracy"])
	plt.plot(history.history["val_accuracy"])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig("Model Accuracy {0:.4f}.png".format(acc))
	plt.clf()
	plt.cla()
	# summarize history for loss
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("model loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig("Model Loss {0:.4f}.png".format(loss))
	plt.clf()
	plt.cla()

train()