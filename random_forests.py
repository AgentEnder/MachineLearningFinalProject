from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
def classify(x_train,x_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    confusionMat = confusion_matrix(y_test, test_pred)
    print(f"Accuracy: {accuracy_score(y_test, test_pred)}")
    pass