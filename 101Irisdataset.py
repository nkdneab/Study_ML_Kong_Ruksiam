from sklearn import datasets
from sklearn.model_selection import  train_test_split


irisdataset = datasets.load_iris()


#75% 25%
x_train,x_test,y_train,y_test = train_test_split(irisdataset["data"],irisdataset["target"],test_size=0.2,random_state=0)
print(y_train)