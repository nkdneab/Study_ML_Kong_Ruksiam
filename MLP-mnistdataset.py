from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

#download
mnist_raw = loadmat("mnist-original.mat")
mnist = {"data":mnist_raw["data"].T,
"target":mnist_raw["label"][0]}

x,y=mnist["data"],mnist["target"]

#shuffle data
shuffle=np.random.permutation(70000)
x,y=x[shuffle],y[shuffle]

#training and test set
#จำนวน 70,000 รูป แบ่งเป็น 60,000 (Training Set) + 10,000 (Test Set)
x_train,x_test,y_train,y_test = x[:60000],x[60000:],y[:60000],y[60000:]

fig,ax = plt.subplots(10,10,
                      figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},
                      gridspec_kw=dict(hspace=0.1,wspace=0.1)
                      )
#display image data before training
#for i , axi in enumerate(ax.flat):
#    axi.imshow(x_train[i].reshape(28,28),cmap='binary',interpolation='nearest')
#    axi.text(0.05,0.05,str(int(y_train[i])),transform=axi.transAxes,color='black')
#plt.show()

#train model
model = MLPClassifier()
model.fit(x_train,y_train)

#prediction
pred=model.predict(x_test)

#display image data after training & prediction
for i,axi in enumerate(ax.flat):
    #display test image data
    axi.imshow(x_test[i].reshape(28,28),cmap='binary',interpolation='nearest')
    #display text true number image data
    axi.text(0.05,0.05,str(int(y_test[i])),transform=axi.transAxes,color='black')
    #display text predict number image data
    axi.text(0.75,0.75,str(int(pred[i])),transform=axi.transAxes,color='green' if pred[i] == y_test[i] else 'red')
plt.show()