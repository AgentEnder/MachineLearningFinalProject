from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
def classify(x_train,x_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy', random_state = 0)
    model.fit(x_train, y_train)
    temp=model.estimators_[5]
    from sklearn.tree import export_graphviz
    export_graphviz(temp, out_file='forest.dot',  feature_names = x_train.columns, class_names = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"],rounded = True, proportion = False, precision = 2, filled = True)
    from subprocess import call
    call(['dot', '-Tpng', 'forest.dot', '-o', 'forest.png', '-Gdpi=600'])

    
    test_pred = model.predict(x_test)
    
    cm = confusion_matrix(y_test, test_pred)
    print(f"Accuracy: {accuracy_score(y_test, test_pred)}")
    return cm