from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sb


#download and display image
faces = fetch_lfw_people(min_faces_per_person=60)
#print(faces.target_names)
#print(faces.images.shape)


#เเสดงผลรูปเป็นตารางพร้อมบอกชื่อ
#fig,ax =plt.subplots(3,5)
#for i ,axi in enumerate(ax.flat): #flat คือการบีบอาเรให้เป็นเส้นตรงเเล้ว เพื่อทำให้มันเเบบเอามันรันใน enumerate ได้ เพื่อเเยกเป็นตำเเหน่งๆใน axi
#    axi.imshow(faces.images[i],cmap='bone') #ค่าสี คือ cmap เป็นสีกระดูก
#    axi.set(xticks=[],yticks=[]) #[]คือให้ xtricks ytricks ไม่มีอะไรเลยไม่งั้นจะมีขอบเป็นพิกัด
#    axi.set_ylabel(faces.target_names[faces.target[i]].split()[-1],color='black') #รูปที่ i เป็นของคนที่ตัวเลข[faces.target[i]] ตัวเลขนั้นมันเท่ากับ ชื่อนี้เอาเเค่ชื่นต้น
#plt.show()

#reduce & create model
pca = PCA(n_components=150,svd_solver='randomized',whiten=True)

svc = SVC(kernel='rbf',class_weight='balanced')
#กำหนดkernel,class_weight
#ถ้าจะเอา PCA ไปใช้งานร่วมกับ SVC เราต้องสร้างท่อเชื่อม
#create model
model = make_pipeline(pca,svc)

#train ,test
x_train,x_test,y_train,y_test = train_test_split(faces.data,faces.target,random_state=40)

#
param = {"svc__C":[1,5,10,50],"svc__gamma":[0.0001,0.0005,0.001,0.005]}

#train data to model
grid = GridSearchCV(model,param)
grid.fit(x_train,y_train)
#print(grid.best_params_) {'svc__C': 5, 'svc__gamma': 0.001}
#print(grid.best_estimator_)

#โมเดลใหม่ที่trainด้วยค่าที่ดีที่สุด
model2 = grid.best_estimator_

#prdict
pred = model2.predict(x_test)

#เเสดงผลเป็นเเผนภาพ
#fig,ax =plt.subplots(4,6)
#for i ,axi in enumerate(ax.flat): #flat คือการบีบอาเรให้เป็นเส้นตรงเเล้ว เพื่อทำให้มันเเบบเอามันรันใน enumerate ได้ เพื่อเเยกเป็นตำเเหน่งๆใน axi
#    axi.imshow(x_test[i].reshape(62,47),cmap='bone') #62,47 dataบอกมาต้องรู้
#    axi.set(xticks=[],yticks=[]) #[]คือให้ xtricks ytricks ไม่มีอะไรเลยไม่งั้นจะมีขอบเป็นพิกัด
#    axi.set_ylabel(faces.target_names[pred[i]].split()[-1],color='green' if pred[i]==y_test[i] else 'red') #รูปที่ i เป็นของคนที่ตัวเลข[faces.target[i]] ตัวเลขนั้นมันเท่ากับ ชื่อนี้เอาเเค่ชื่นต้น
#plt.show()

#เเสดงผลเเบบaccuracy score
print('Accuracy = ',accuracy_score(y_test,pred))

#เเสดงผลเเบบ comfustion matrix
mat = confusion_matrix(y_test,pred)

sb.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('True Data')
plt.ylabel("Predict Data")
plt.show()


