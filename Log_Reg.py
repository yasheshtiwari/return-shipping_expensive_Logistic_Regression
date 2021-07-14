# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:58:06 2021

@author: yashesh
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#get data for traning
df = pd.read_csv("TrainingData_V1.csv")
df_test = pd.read_csv("TestingData_For_Candidate.csv")

#Check nullable value
#df.info()

#missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/79948)*100})
#print(df.item_color.unique())
#Check the duplicate records
#print(df.duplicated().sum())

#Fill delivery_date missing value with above date. 
df.delivery_date = df.delivery_date.fillna(method='ffill')
df_test.delivery_date = df_test.delivery_date.fillna(method='ffill')
#missing_data1 = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/79948)*100})

#to check item_size is unsize is return or not
df_unsize = df[df.item_size == 'unsized']
rtn = df_unsize["return"].value_counts()
df_unsize["return"].value_counts().plot(kind='bar')


#print(df_test.item_size.unique())

#assigning numerical values and storing in another column
labelencoder = LabelEncoder()

df["item_color"] = labelencoder.fit_transform(df["item_color"])
df_test["item_color"] = labelencoder.fit_transform(df_test["item_color"])
#Correct the item_size columns
df["item_size"] = labelencoder.fit_transform(df["item_size"].astype(str))
df_test["item_size"] = labelencoder.fit_transform(df_test["item_size"].astype(str))

#print(df.item_size.unique())


#Fill user dob missing value with above date. 
df.user_dob = df.user_dob.fillna(method='ffill')
df_test.user_dob = df_test.user_dob.fillna(method='ffill')
#Convert date of birth in age format

def age(born):
    today = date.today()
    born = datetime.strptime(str(born), '%d-%m-%Y').date()
    return today.year - born.year - ((today.month,today.day) < (born.month, born.day))

def ageTest(born):
    today = date.today()
    born = datetime.strptime(str(born), '%m/%d/%Y').date()
    return today.year - born.year - ((today.month,today.day) < (born.month, born.day))

df["user_dob"] = df["user_dob"].apply(age)
df_test["user_dob"] = df_test["user_dob"].apply(ageTest)

df[['order_date','delivery_date']] = df[['order_date','delivery_date']].apply(pd.to_datetime) #if conversion required
df['del_diff'] = (df['delivery_date'] - df['order_date']).dt.days

df_test[['order_date','delivery_date']] = df_test[['order_date','delivery_date']].apply(pd.to_datetime) #if conversion required
df_test['del_diff'] = (df_test['delivery_date'] - df_test['order_date']).dt.days


#Get one hot encoding of column user_title
one_hot = pd.get_dummies(df["user_title"])
df = df.drop("user_title", axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df_test["user_title"])
df_test = df_test.drop("user_title", axis = 1)
df_test = df_test.join(one_hot)

#df_1 = pd.get_dummies(df["user_title"])

#drop unused columns
df = df.drop(columns = ["brand_id",'item_id','order_item_id','order_date','delivery_date','user_id','user_state','user_reg_date'])
df = df[['item_size',"item_color","del_diff","item_price","user_dob","Company","Family","Mr","Mrs","not reported","return"]]
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1:].values

df_test = df_test.drop(columns = ["brand_id",'item_id','order_item_id','order_date','delivery_date','user_id','user_state','user_reg_date'])
df_test = df_test[['item_size',"item_color","del_diff","item_price","user_dob","Company","Family","Mr","Mrs","not reported"]]
X_test = df_test.iloc[:,:].values

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X)

X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(X_test, y_pred)


