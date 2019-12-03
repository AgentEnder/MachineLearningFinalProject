import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def build_and_train_net(data,classes):
    
    
    classifier=Sequential()
    classifier.add(Dense(activation="relu",units=20,input_dim=data.shape[1]))
    classifier.add(Dense(units=15,activation="sigmoid"))
    classifier.add(Dense(units=10,activation="sigmoid"))
    classifier.add(Dense(units=data.shape[1],activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    x=data.values
    y=classes.values[:,2]
    #y=np.true_divide(y,4)
    enc = OneHotEncoder()
    y=np.reshape(y,[len(y),1])
    y=enc.fit_transform(y).toarray()
    print(y[0])
    classifier.fit(x,y,batch_size=10,nb_epoch=500)
    classifier.save("Model.h5")

    pass
def test_classifier(data,classes):
    from sklearn.metrics import confusion_matrix
    classifier = load_model("Model.h5")
    predictions=classifier.predict(data.values)
    #temp_predictions=np.zeros(predictions.shape)
    #temp_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    
    predictions=predictions.argmax(1)
    y=classes.values[:,2]
        
    cm = confusion_matrix(predictions,y)
    print(cm)
def run_classifier(data):
    pass