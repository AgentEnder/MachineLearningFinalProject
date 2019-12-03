from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
def classify(dataset):
    trainingDataset = dataset
    attributesMat = trainingDataset.iloc[:, 1:-3].values
    classMat = [int(x) for x in trainingDataset.iloc[:, -1].values]
    X_train, X_test, Y_train, Y_test = train_test_split(attributesMat, classMat, test_size=0.4, random_state=0)
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, Y_train)
    test_pred = model.predict(X_test)
    confusionMat = confusion_matrix(Y_test, test_pred)
    print(f"Accuracy: {accuracy_score(Y_test, test_pred)}")
    pass