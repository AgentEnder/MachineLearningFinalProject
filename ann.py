import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

import numpy as np


def build_and_train_net(data,classes):
    
    
    classifier=Sequential()
    classifier.add(Dense(activation="relu",units=1,input_dim=3))
    classifier.add(Dense(units=1,activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    x=data.values
    y=classes.values[:,2]
    y=np.divide(y,)
    classifier.fit(x,y,batch_size=10,nb_epoch=100)
    classifier.save("Model.h5")

    pass
def test_classifier(data,classes):
    classifier = load_model("Model.h5")
    predictions=classifier.predict(data.values)
    print(predictions)
def run_classifier(data):
    pass