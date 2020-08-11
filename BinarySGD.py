from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def displayImage(x_wannaknow):
    plt.imshow(x_wannaknow.reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
    plt.show()

def displayPredict(model, y_anser, x_wannaknow):
    print("Actually = ", y_anser)
    print("Prediction  = ", model.predict([x_wannaknow])[0])

mnist_raw = loadmat("mnist-original.mat")
mnist = {"data":mnist_raw["data"].T,
"target":mnist_raw["label"][0]}

x,y=mnist["data"],mnist["target"]

#training and test set
#จำนวน 70,000 รูป แบ่งเป็น 60,000 (Training Set) + 10,000 (Test Set)
x_train,x_test,y_train,y_test = x[:60000],x[60000:],y[:60000],y[60000:]


#เช็คว่าใช่เลข 0มั้ย
#ข้อมูลค่าที่ 5000 -> model -> ใช่ 0 มั้ย -> True False
predict_number=5000
#y_train = [0,0,0,.....,9,...,9]
y_train_0 = (y_train==0)
y_test_0 = (y_test==0)
#y_train_0 = [True,True,True,....False,False,...,False]

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train,y_train_0)

#displayPredict(sgd_clf,y_test_0[predict_number],x_test[predict_number])
#displayImage(x_test[predict_number])


#การวัดประสิทะิภาพด้วย cross validation score test
#score = cross_val_score(sgd_clf,x_train,y_train_0,cv=3,scoring="accuracy")
#print(score)

#การเช็คว่าผิดตรงไหนด้วย cross validation predict test
def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 0"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

y_train_pred = cross_val_predict(sgd_clf,x_train,y_train_0,cv=3)
cm=confusion_matrix(y_train_0,y_train_pred)

#เเสดงผลข้อมูลจาก Confusion Matrix
plt.figure()
displayConfusionMatrix(cm)



#การหาค่าAccuracy, Precision, Recall และ F1 Score จะอาศัยข้อมูลจาก Confusion Matrix
#https://medium.com/@kongruksiamza/%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B-machine-learning-ep-4-%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%88%E0%B8%B3%E0%B9%81%E0%B8%99%E0%B8%81%E0%B9%81%E0%B8%9A%E0%B8%9A%E0%B9%84%E0%B8%9A%E0%B8%A3%E0%B8%B2%E0%B8%A3%E0%B8%B5%E0%B9%88-binary-classifier-6ebc8e1a5e61
y_test_pred = sgd_clf.predict(x_test)

classes = ['Other Number','Number 5']
print(classification_report(y_test_0,y_test_pred,target_names=classes))
print("Accuracy Score = ",accuracy_score(y_test_0,y_test_pred)*100)