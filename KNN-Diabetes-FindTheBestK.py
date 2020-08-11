from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("diabetes.csv")

x = df.drop('Outcome',axis=1)#axis=0 คือเเถว

#เอาคอลลัมออกให้พร้อมใช้ #ค่าต่างๆ
x=x.values
#ผลเฉลย
y=df['Outcome']

#training and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

#find the best model
k_neighbors = np.arange(1,9)
train_score=np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

for i,k in enumerate(k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    #วัดประสิทธิภาพ
    train_score[i]=knn.score(x_train,y_train)
    test_score[i]=knn.score(x_test,y_test)
    print(test_score[i]*100)
plt.title('Compare k Value in Model')
plt.plot(k_neighbors,test_score,label='Test score')
plt.plot(k_neighbors,train_score,label='Train score')
plt.legend()
plt.xlabel("K Number")
plt.ylabel("Score")
plt.show()