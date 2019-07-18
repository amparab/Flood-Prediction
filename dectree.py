# Load libraries

import eel
eel.init('web')
@eel.expose
def dectree_function(loc,r,d,t):
    if loc=='Santacruz':      
        import numpy as np
        import pandas as pd
        names = ['Rainfall','Drainage','Tide','class']
        dataframe = pd.read_csv("C:/Users/SAHIL/Desktop/FloodPredict/Updated SANTACRUZ 1.1.csv", names=names)
        array = dataframe.values
        X = array[:,[0,1,2]]
        Y = array[:,3]
        from sklearn.model_selection import train_test_split  
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) 
        #model fitting
        from sklearn.tree import DecisionTreeClassifier  
        classifier = DecisionTreeClassifier(random_state=0)  
        classifier.fit(X_train, y_train)
        xtest=np.array([int(r),int(d),float(t)]).reshape(-1,3)
        ynew = classifier.predict(xtest)
        return ynew.tolist()
    else:
        import numpy as np
        import pandas as pd
        names = ['Rainfall','Drainage','Tide','class']
        dataframe = pd.read_csv("C:/Users/SAHIL/Desktop/FloodPredict/Updated Colaba 1.1.csv", names=names)
        array = dataframe.values
        X = array[:,[0,1,2]]
        Y = array[:,3]
        from sklearn.model_selection import train_test_split  
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) 
        #model fitting
        from sklearn.tree import DecisionTreeClassifier  
        classifier = DecisionTreeClassifier(random_state=0)  
        classifier.fit(X_train, y_train)
        xtest=np.array([int(r),int(d),float(t)]).reshape(-1,3)
        ynew = classifier.predict(xtest)
        return ynew.tolist()
 
my_options = {
    'mode': "chrome", #or "chrome-app",
    'host': 'localhost',
    'port': 1024,
    'chromeFlags': [ "--browser-startup-dialog"]
}
eel.start('index_dectree.html',block=False,options=my_options)
while True:
    eel.sleep(10)

    
