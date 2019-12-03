import pandas as pd
import preprocessing
from feature_selection import select_features

dataset = pd.read_csv('data/student-mat.csv', delimiter=";")
dataset = preprocessing.preprocess(dataset)
x, y = preprocessing.split_attributes(dataset, 3)
x = select_features(x, 5)
y = preprocessing.bucketize_y(y, 5)

cols = list(x.columns)
cols.extend(y.columns)

new_data = pd.DataFrame(data=x.join(y), columns=cols)

#Neural Network
import ann
ann.build_and_train_net(x,y)
ann.test_classifier(x,y)