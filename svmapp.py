# Load libraries

import eel
eel.init('web')
@eel.expose
def svm_function(loc,r,d,t):
    if loc=='Santacruz':
        import pandas
        from sklearn import model_selection
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
        import numpy as np
        # Load dataset
        names=['Rainfall','Drainage','Tide','CLASS']
        dataset = pandas.read_csv("E:/Review 6 March/Updated Data 1.1.csv", names=names)
        # Split-out validation dataset
        dataset.head()
        dataset.describe()
        dataset.dropna(inplace=True)
        dataset.describe()

        array = dataset.values
        X = array[:,0:3]
        Y = array[:,3]
        validation_size = 0.30
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        # Test options and evaluation metric
        seed = 7
        scoring = 'accuracy'
        # Spot Check Algorithms
        models = []
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
        # Using SVM Classifier
        # Make predictions on validation dataset
        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(X_train, Y_train)
        #predictions = svm.predict(X_validation)
        xtest=np.array([r,d,t]).reshape(-1,3)
        ynew = svm.predict(xtest)
        return ynew.tolist()
    else:
        import pandas
        from sklearn import model_selection
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
        import numpy as np
        # Load dataset
        names=['Rainfall','Drainage','Tide','CLASS']
        dataset = pandas.read_csv("E:/Review 6 March/Updated Data 1.1.csv", names=names)
        # Split-out validation dataset
        array = dataset.values
        X = array[:,0:3]
        Y = array[:,3]
        validation_size = 0.30
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        # Test options and evaluation metric
        seed = 7
        scoring = 'accuracy'
        # Spot Check Algorithms
        models = []
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
        # Using SVM Classifier
        # Make predictions on validation dataset
        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(X_train, Y_train)
        #predictions = svm.predict(X_validation)
        xtest=np.array([r,d,t]).reshape(-1,3)
        ynew = svm.predict(xtest)
        return ynew.tolist()

my_options = {
    'mode': "chrome", #or "chrome-app",
    'host': 'localhost',
    'port': 1024,
    'chromeFlags': [ "--browser-startup-dialog"]
}
eel.start('index_svm.html',block=False,options=my_options)
while True:
    eel.sleep(10)

    
