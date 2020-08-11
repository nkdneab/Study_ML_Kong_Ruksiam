import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  accuracy_score

dataset = pd.read_csv('adult.csv')


def cleandata(dataset):#ทำให้data ที่ไม่ใช่ตัวเลขกลายเป็นตัวเลขให้หมด เพศชายหญิง = ๅ 0
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

dataset = cleandata(dataset)

#training and test set
training_set,test_set = train_test_split(dataset,test_size=0.2)

def split_feature_class(dataset,feature):
    features = dataset.drop(feature,axis=1) #ข้อมูลทั้งหมดที่ไว้ใช้เรียนไม่มีเเฉลย
    labels = dataset[feature].copy() #เฉพาะ เฉลย
    return features,labels

#trainset
training_features,treaining_labels=split_feature_class(training_set,'income')

#testset
test_features,test_labels=split_feature_class(test_set,'income')

#model
model=GaussianNB()
model.fit(training_features,treaining_labels)

#predict
pred = model.predict(test_features)

print("Accuracy = ",accuracy_score(test_labels,pred))