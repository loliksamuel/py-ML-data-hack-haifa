
'''
jupiter-http://localhost:8888/
open-source web application that allows us to create and share codes and documents.
https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks
jupyter notebook
jupyter notebook notebook.ipynb
jupyter notebook --help

pip-https://pip.pypa.io
installer for python libraries
pip install ___
pip list
pip show ___

numpy
import numpy as np
a=[1,2,3]
arr = np.array(a) # numpy arr is faster(written in c)
np.transpose(arr) # numpy support matrix transpose
arr.shape # show the # dimentions
np.expand.dims(arr.axis) #Expand the shape of an array.

scikit-learn - scalers, normalizers, models
import sklearn as sk
sk.datasets # return np array
sk.linear_model

matplotlib-https://matplotlib.org/
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot   (x,y)
plt.bar    (x,y)
plt.show   ()
plt.close  ()
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = datasets.load_diabetes()
# print (data)
x = data.data
y = data.target
print(x.shape)# (442, 10)  442 samples, 10 features
print(y.shape)# (442, 1 )  442 labels ,  1 label


y = np.expand_dims(y,1)
print (y.shape)
x = x[:,0]
x = np.expand_dims(x,1)
print(x.shape)
#print(dir(model))
#print(dir())
#print(type(model))
#print(dir(model))
model = linear_model.LinearRegression()
model.fit(x,y)
predictions = model.predict(x)
print("predictions: ",predictions[0] , " vs actual ", y[0])

plt.plot   (x,predictions)#line
plt.scatter(x,y)#points
plt.show()#points+line


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