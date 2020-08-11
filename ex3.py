from scipy.io import loadmat
import matplotlib.pyplot as plt
mnist_raw=loadmat('ex3data1.mat')

mnist = {'data':mnist_raw['X'],'target':mnist_raw["y"]}

x,y=mnist["data"],mnist["target"]

number=x[15]
number_image=number.reshape(20,20)

print(y[15])

plt.imshow(number_image,cmap=plt.cm.binary,interpolation='nearest')
plt.show()