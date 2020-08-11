from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


irisdataset = datasets.load_iris()
x_train,x_test,y_train,y_test = train_test_split(irisdataset["data"],irisdataset["target"],test_size=0.4,random_state=0)

#model
knn = KNeighborsClassifier(n_neighbors=60)

#training
knn.fit(x_train,y_train)

#prediction
y_pred = knn.predict(x_test)

#accuracy
print("Accuracy Score = ",accuracy_score(y_test,y_pred)*100)

