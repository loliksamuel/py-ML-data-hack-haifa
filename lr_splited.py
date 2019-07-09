import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = datasets.load_diabetes()

x = data.data
y = data.target


# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y)
print("x : number of dimensions: ",x_train.ndim) # 3
print("X : dimension size: "      ,x_train.shape) # (60000, 28, 28)
print("x : data type: "           ,x_train.dtype) # uint8

print("y : number of dimensions: ",y_train.ndim) # 1
print("y : dimension size: "      ,y_train.shape) # (60000) train samples
print("y : data type: "           ,y_train.dtype) # uint8

print ("x_test : "  ,  x_test.shape)
print ("y_test : "  ,  y_test.shape)

#print ("raw train data", x_train)
#print ("raw test  data", x_test )
print ("\nraw train data[0]", x_train[0] )
print (  "raw test  data[0]", x_test [0] )
print ("\nraw train y[0]"   , y_train[0] )
print (  "raw test  y[0]"   , y_test [0] )


# reset x with 1 column instead of 10
x_train = x_train[:,0]
x_train = np.expand_dims(x_train,1)
print ("reset x with 1 column as feature (instead of 10), x.shape=", x_train.shape, " ex.:", x_train[0])

# reset y with 1 column as label(as was before)
y_train = np.expand_dims(y_train,1)
print ("reset y with 1 column as label   (instead of  0), y.shape=", y_train.shape, " ex.:", y_train[0])


# reset x with 1 column instead of 10
x_test = x_test[:,0]
x_test = np.expand_dims(x_test,1)
print ("reset x with 1 column as feature (instead of 10), x.shape=", x_train.shape, " ex.:", x_train[0])

# reset y with 1 column as label(as was before)
y_test = np.expand_dims(y_test,1)
print ("reset y with 1 column as label   (instead of  0), y.shape=", y_train.shape, " ex.:", y_train[0])

# create model
model = linear_model.LinearRegression()


# Prints a string summary of the  neural network.
#model.summary()

#train model on train data
model.fit(x_train, y_train)

print("model summary")
print(dir(model))
print(dir())
print(type(model))
print(dir(model)) # diagnose the model arguments


# predict on test data (unseen data)
predictions = model.predict(x_test)
print("predictions: ",predictions[0] , " vs actual ", y_test[0])

# plot
plt.plot   (x_test,predictions)#prepare lr line
plt.scatter(x_test,y_test)   #prepare data points
plt.show   ()                  #show graph of line + points




#
#
# print("hi")
# for i in range(2,20):
#     print (i)
# fruit=["banana", "apple", "orange"]
# for f in fruit:
#     print (f)
#
# def foo(a,b):
#     print("sum up")
#     return a+b