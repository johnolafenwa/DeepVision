#import needed classes
import keras
from keras.datasets import cifar10
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
from keras.models import Model,Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import os

#load the mnist dataset
(train_x, train_y) , (test_x, test_y) = cifar10.load_data()

#normalize the data
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

#print the shapes of the data arrays
print("Train Images: ",train_x.shape)
print("Train Labels: ",train_y.shape)
print("Test Images: ",test_x.shape)
print("Test Labels: ",test_y.shape)



#Encode the labels to vectors
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)

#define a common unit
def Unit(x,filters):
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    return out

#Define the model

def residual_block(x,channels):

    residual = x

    net = Unit(x, channels)
    net = Unit(net,channels)

    output = keras.layers.add([net, residual])

    return output

def MiniModel(input_shape):
    images = Input(input_shape)


    net = Unit(images,64)
    net = Unit(net,64)
    net = Unit(net,64)
    net = MaxPooling2D(pool_size=(2,2))(net)

    net = Unit(net,128)
    net = Unit(net,128)
    net = Unit(net,128)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Unit(net,256)
    net = Unit(net,256)
    net = Unit(net,256)

    net = Dropout(0.25)(net)
    net = AveragePooling2D(pool_size=(8,8))(net)
    net = Flatten()(net)
    net = Dense(units=10,activation="softmax")(net)

    model = Model(inputs=images,outputs=net)

    return model

input_shape = (32,32,3)
model = MiniModel(input_shape)

#Print a Summary of the model

model.summary()

#Define the Learning rate schedule function

def lr_schedule(epoch):

    lr = 0.001

    if epoch > 15:
        lr = lr / 100
    elif epoch > 10:
        lr = lr / 10
    elif epoch > 5:
        lr = lr / 5

    print("Learning Rate: ",lr)

    return lr

#Pass teh scheduler function to the Learning Rate Scheduler class
lr_scheduler = LearningRateScheduler(lr_schedule)

#Directory in which to create models
save_direc = os.path.join(os.getcwd(), 'cifar10savedmodels')

#Name of model files
model_name = 'cifar10model.{epoch:03d}.h5'

#Create Directory if it doesn't exist
if not os.path.isdir(save_direc):
    os.makedirs(save_direc)

#Join the directory with the model file
modelpath = os.path.join(save_direc, model_name)

checkpoint = ModelCheckpoint(filepath=modelpath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             period=1)

#Specify the training components
model.compile(optimizer=Adam(lr_schedule(0)),loss="categorical_crossentropy",metrics=["accuracy"])

#Fit the model
model.fit(train_x,train_y,batch_size=128,epochs=20,shuffle=True,validation_split=0.1,verbose=1,callbacks=[checkpoint,lr_scheduler])

#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=128)

print("Accuracy: ",accuracy[1])



