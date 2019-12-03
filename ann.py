import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def build_and_train_net(data,classes):
    x=data.values
    y=classes
    #y=np.true_divide(y,4)
    enc = OneHotEncoder()
    y=np.reshape(y,[len(y),1])
    y=enc.fit_transform(y).toarray()
    
    classifier=Sequential()
    classifier.add(Dense(activation="relu",units=32,input_dim=data.shape[1]))
    classifier.add(Dense(units=16,activation="relu"))
    classifier.add(Dense(units=16,activation="relu"))
    classifier.add(Dense(units=16,activation="relu"))
    classifier.add(Dense(units=y.shape[1],activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    
    print(y[0])
    classifier.fit(x,y,batch_size=10,nb_epoch=300)
    classifier.save("Model.h5")

    pass
def test_classifier(data,classes):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    classifier = load_model("Model.h5")
    predictions=classifier.predict(data.values)
    #temp_predictions=np.zeros(predictions.shape)
    #temp_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    
    predictions=predictions.argmax(1)
    y=classes
        
    cm = confusion_matrix(predictions,y)
    print(cm)
    acc = accuracy_score(predictions,y)
    print(acc)
def run_classifier(data):
    pass