import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_and_train_net(data,classes):
    
    
    classifier=Sequential()
    classifier.add(Dense(activation="relu",input_dim=3))
    classifier.add(Dense(activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    classifier.fit(x)
    classifier.save("Model.h5")

    pass

def run_classifier(data):
    pass