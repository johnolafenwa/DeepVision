#import needed classes
import keras
from keras.datasets import mnist
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image

#load the mnist dataset
(train_x, train_y) , (test_x, test_y) = mnist.load_data()

#Define the model
model = Sequential()
model.add(Dense(units=128,activation="relu",input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))


#Specify the training components
model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])

#Load the pretrained model
model.load_weights("mnistmodel.h5")

#Normalize the test dataset
test_x = test_x.astype('float32') / 255

#Load an image from your system
img = image.load_img(path="testimage.png",grayscale=True,target_size=(28,28))
img = image.img_to_array(img)
img = img.reshape((28,28))

#Create a flattened copy of the image
test_img = img.reshape((1,784))

#Predict the class
img_class = model.predict_classes(test_img)
classname = img_class[0]
print("Class: ",classname)

#Display the original non-flattened copy of the image
plt.title("Prediction Result: %s"%(classname))
plt.imshow(img)
plt.show()




