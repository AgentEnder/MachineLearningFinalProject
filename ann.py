import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def build_and_train_net(data,classes,testDat,testClass):
    x=data.values
    y=classes
    #y=np.true_divide(y,4)
    enc = OneHotEncoder()
    y=np.reshape(y,[len(y),1])
    y=enc.fit_transform(y).toarray()
    testClass=np.reshape(testClass,[len(testClass),1])
    testClass=enc.fit_transform(testClass).toarray()
    classifier=Sequential()
    classifier.add(Dense(activation="relu",units=16,input_dim=data.shape[1]))
    classifier.add(Dropout(rate=.15))
    classifier.add(Dense(units=12,activation="relu"))
    classifier.add(Dense(units=6,activation="sigmoid"))
    classifier.add(Dense(units=y.shape[1],activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    
    print(y)
    #history = classifier.fit(x,y,batch_size=10,nb_epoch=10,validation_data=(testDat.values,testClass))
    history = classifier.fit(x,y,batch_size=10,nb_epoch=80)
    
    valCheck=False
    if valCheck==True:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(history.history["val_loss"])),history.history["val_loss"])
        plt.show()
        curMax=10
        curMaxI=0
        for i in range(len(history.history["val_loss"])):
            if history.history["val_loss"][i] < curMax:
                curMax=history.history["val_loss"][i]
                curMaxI=i
        print(str(curMax)+" " +str(curMaxI))
    classifier.save("Model.h5")
    return history

    pass
def test_classifier(data,classes):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    
    classifier = load_model("Model.h5")
    
    from tensorflow.keras.utils import plot_model
    plot_model(classifier, to_file='model.png',expand_nested=True)

    
    predictions=classifier.predict(data.values)
    #temp_predictions=np.zeros(predictions.shape)
    #temp_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    
    predictions=predictions.argmax(1)
    y=classes
        
    cm = confusion_matrix(y,predictions)
    print(cm)
    acc = accuracy_score(y,predictions)
    print(acc)
    return cm
def run_classifier(data):
    pass