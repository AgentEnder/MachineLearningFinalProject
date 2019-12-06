import pandas as pd
import preprocessing
from sklearn.model_selection import train_test_split
from feature_selection import select_features

dataset = pd.read_csv('data/student-mat.csv', delimiter=";")
dataset = preprocessing.preprocess(dataset)
x, y = preprocessing.split_attributes(dataset, 3)
#x = select_features(x, 16)
#y=preprocessing.bucketize_y(y,2)
#y = preprocessing.statsBuckets(y["G3"])

cols = list(x.columns)
cols.extend(y.columns)

new_data = pd.DataFrame(data=x.join(y), columns=cols)
#new_data.to_csv("temp.csv")

#splitting train and test

x_train,x_test,y_train,y_test = train_test_split(x,y.values[:,2],test_size=.2,random_state=0)

import random_forests
cm = random_forests.classify(x_train,x_test,y_train,y_test)

#Neural Network
import ann
#history = ann.build_and_train_net(x_train,y_train,x_test,y_test)
cm2 = ann.test_classifier(x_test,y_test)