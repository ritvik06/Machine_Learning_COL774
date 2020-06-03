import pandas as pd
import numpy as np
import math
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.model_selection import KFold

X = np.asarray(pd.read_csv("./MNIST/train.csv",encoding='latin-1'))
Test = np.asarray(pd.read_csv("./MNIST/test.csv",encoding='latin-1'))

#==============================TRAINING DATASET================================    

X_train = []
Y_train = []

for i in range(len(X)):
    if (X[i][-1]==7 or X[i][-1]==8):
        X_train.append(X[i][:-1])
        if(X[i][-1]==7):
            Y_train.append(-1)
        else:
            Y_train.append(1)

X_train = np.asarray(X_train)
X_train = X_train/255
Y_train = np.asarray(Y_train)

#==============================TEST DATASET================================    

X_test = []
Y_test = []

for i in range(len(Test)):
    if (Test[i][-1]==7 or Test[i][-1]==8):
        X_test.append(Test[i][:-1])
        if(Test[i][-1]==7):
            Y_test.append(-1)
        else:
            Y_test.append(1)

X_test = np.asarray(X_test)
X_test = X_test/255.
Y_test = np.asarray(Y_test)

def accuracy(pred,truth):
    correct = 0
    for i in range(len(pred)):
        if(pred[i]==truth[i]):
            correct+=1
    return correct

#     print('C: ',c,' Correct:',correct,'/ 900')
#     print('Accuracy:',float(correct*100/900),'%')
        

#=============================CROSS VALIDATION ACCURACY=================================
kfold = KFold(5, True, 1)
C_acc = {}
list_C = [10**-5,10**-3,1,5,10]

for c in list_C:
    C_acc[c] = 0
for train, val in kfold.split(X_train):
    for c in list_C:
        clf = SVC(C = c,kernel = 'rbf',gamma=0.05)
        clf.fit(X_train[train],Y_train[train])

        y_pred = clf.predict(X_train[val])

        C_acc[c]+=accuracy(y_pred,Y_train[val])

for c in list_C:
    print('C: ', c, ' Accuracy:',float(C_acc[c]*100/4500),'%')

#=============================TEST DATA ACCURACY=================================

for c in list_C:
    clf = SVC(C = c,kernel = 'rbf',gamma=0.05)
    clf.fit(X_train,Y_train)
    
    y_pred = clf.predict(X_test)
    res = accuracy(y_pred,Y_test)
    
    print('C: ',c,' Correct:',res,'/',len(X_test))
    print('Accuracy:',float(res*100/len(X_test)),'%')  
