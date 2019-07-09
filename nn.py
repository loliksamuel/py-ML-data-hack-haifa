'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets   import mnist
from keras.models     import Sequential
from keras.layers     import Dense, Dropout
from keras.optimizers import RMSprop
# import matplotlib.pyplot as plt

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
# print("creating deep learning to classify images to digit(0-9). MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8")

num_classes = 10 # there are 10 classes (10 digits from 0 to 9)
batch_size  = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs      = 20 # 20 iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
#iterations  = 60000/128

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print("photo of  x_train[8] is :")
#plt.imshow(x_train[8], cmap=plt.cm.binary)
#print("label of x_train[8] is ", y_train[8])

# These MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8.
print("x : number of dimensions: ",x_train.ndim) # 3
print("X : dimension size: "      ,x_train.shape) # (60000, 28, 28)
print("x : data type: "           ,x_train.dtype) # uint8

print("y : number of dimensions: ",y_train.ndim) # 1
print("y : dimension size: "      ,y_train.shape) # (60000) train samples
print("y : data type: "           ,y_train.dtype) # uint8

print ("x_test : "  ,  x_test.shape)
print ("y_test : "  ,  y_test.shape)
# converting all 60000 images from matrix (28X28) to a vector (of size 784 features or neurons)
x_train = x_train.reshape(60000, 784)
x_test  =  x_test.reshape(10000, 784)
print ("raw train data", x_train)
print ("raw test  data", x_test )
print ("\nraw train data[0]", x_train[0])
print ("raw test  data[0]", x_test [0] )

# normalizing 0-255 (int)   to    0-1 (float)
x_train = x_train.astype ('float32')
x_test  = x_test .astype ('float32')
x_train /= 255
x_test  /= 255
print ("\nnormalized train data", x_train)
print ("normalized test  data", x_test )
print ("\nnormalized train data[0]", x_train[0])
print ("normalized test  data[0]", x_test [0] )
print("x : number of dimensions for train: ",x_train.shape) #(60000, 784)
print("x : number of dimensions for test: " , x_test.shape) #(10000, 784)
print("x : data type: "                     , x_test.dtype) # float32
print  (x_train.shape[0], 'train samples')
print  ( x_test.shape[0],  'test samples')

# convert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test , num_classes)

# configure the model. the input/hidden/output layers,  the activation function, The main data structure in Keras is the Sequential class, then we add any activation function...https://en.wikipedia.org/wiki/Activation_function
# model.add(Dense(10, activation=’sigmoid’, input_shape=(784,)))
# model.add(Dense(10, activation=’softmax’))
model = Sequential()
model.add(Dense  (512        , activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense  (512        , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense  (num_classes, activation='softmax'))# 10 neurons for 0-9 digits

# Prints a string summary of the  neural network.
model.summary()

#configure  model
model.compile(loss       = 'categorical_crossentropy'
              ,optimizer = RMSprop()  # or sgd=stocastic gradient descent
              ,metrics   = ['accuracy']) # Accuracy = (TP + TN) / (TP + FP + FN + TN)

# train  model
history = model.fit( x_train
                    ,y_train
                    ,batch_size=batch_size
                    ,epochs    =epochs
                    ,verbose   =1
                    ,validation_data=(x_test, y_test))

# evaluate the model with unseen data
score = model.evaluate(  x_test
                       , y_test
                       , verbose=0)

print('Test loss:'    , score[0])
print('Test accuracy:', score[1])# 0.9018   the higher the better

# predict unseen data
# x_test = [0,0,0,0,1,0,0,0,0,0]
# predictions = model.predict(x_test)
# print(predictions[11])


# plot