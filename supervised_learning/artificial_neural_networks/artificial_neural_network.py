# Artificial Neural Network

#part 1: data preprocessing
#Classification 

#importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt

#importing datasets
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#to prevent dummy variable trap
X = X[:,1:]

#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# making ANN

# importing keras and packages
import keras
#sequential model to initialize and dense to build layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#initialising the ANN
classifier = Sequential()

# adding input and hidden layers 
classifier.add(Dense(output_dim = 6, #specifying number of nodes in hidden, usually average of number of input and output nodes
                     init = 'uniform', #initializing the weights
                     activation = 'relu', #chosing rectifier as activation fxn in hidden layer
                     input_dim = 11)) #number of input nodes = no of independent variable
# adding dropout
classifier.add(Dropout(p = 0.1))

# addding second hidden layer
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu'))
# adding dropout
classifier.add(Dropout(p = 0.1))

#adding outpur layer
classifier.add(Dense(output_dim = 1,
                     init = 'uniform',
                     activation = 'sigmoid'))

#compiling the ann - stocastic gradiant decscent
classifier.compile(optimizer = 'adam', #algorithm to adjust weights using adam (type of stocastic gradiant descent)
                   loss = 'binary_crossentropy', #for binary output
                   metrics =['accuracy'])

# fitting the ann to training set
classifier.fit(X_train,Y_train,batch_size = 10, nb_epoch = 100)

# making predictions and evaluating the model
#predicting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

# evaluating and improving and tuning the ann
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
# improving the ann

# dropout regularization to reduce overfitting if needed
# tuning the ann 
# parameter tuning (hyper parameters)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25 , 32] , 
              'epochs' : [100,500],
              'optimizer' : ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


