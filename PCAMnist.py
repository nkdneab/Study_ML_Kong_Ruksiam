from scipy.io import loadmat
import matplotlib.pyplot as plt
mnist_raw=loadmat('mnist-original.mat')
from sklearn.model_selection import  train_test_split
from sklearn.decomposition import PCA

mnist = {'data':mnist_raw['data'].T,'target':mnist_raw["label"][0]}

x_train,x_test,y_train,y_test=train_test_split(mnist["data"],mnist["target"], random_state=0)

pca=PCA(.95) #ลดให้เหลือ 95%
data = pca.fit_transform(x_train) #ลดขนาด
result = pca.inverse_transform(data) #ทำให้กลับมาเหมือนเดิม
#print(str(pca.n_components_))
new_complement=str(pca.n_components_)

#SHOW image
plt.figure(figsize=(8,4))

#image feature 784 components
plt.subplot(1,2,1)
plt.imshow(mnist['data'][0].reshape(28,28),cmap=plt.cm.gray,interpolation='nearest')
plt.xlabel('784 components')
plt.title('Original')

#image feature 154 components 95%
plt.subplot(1,2,2)
plt.imshow(result[0].reshape(28,28),cmap=plt.cm.gray,interpolation='nearest')
plt.xlabel(new_complement+' components')
plt.title('PAC')
plt.show()





