from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sb

x,y = make_blobs(n_samples=300,centers=4,cluster_std=0.5,random_state=0) #ค่าปคือ

#train
model = KMeans(n_clusters=4) #บอกโมเดลว่าให้เเยกมาเป็นกี่ประเภท
model.fit(x)

#new point
x_test,y_test = make_blobs(n_samples=10,centers=4,cluster_std=0.5,random_state=0)

#predict
y_pred = model.predict(x)
print(y_pred)
y_pred_new = model.predict(x_test)
centers = model.cluster_centers_
#y_pred = [ 3 1 0 0 0 2 1 ... 3 ]
#SHOW
plt.scatter(x[:,0],x[:,1],c=y_pred)#c=คือเเบ่งสีตามกลุ่มของ y_pred
plt.scatter(x_test[:,0],x_test[:,1],c=y_pred_new,s=120)
plt.scatter(centers[0,0],centers[0,1],c='blue',label='Centroid 1')
plt.scatter(centers[1,0],centers[1,1],c='green',label='Centroid 2')
plt.scatter(centers[2,0],centers[2,1],c='red',label='Centroid 3')
plt.scatter(centers[3,0],centers[3,1],c='black',label='Centroid 4')
plt.legend(frameon=True)
plt.show()


#ตรวจวัดความถูก ด้วย confusion_matrix
mat = confusion_matrix(y,y_pred)
sb.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=[0,1,2,3],yticklabels=[0,1,2,3])
plt.xlabel('True Data')
plt.ylabel("Predict Data")
plt.show()