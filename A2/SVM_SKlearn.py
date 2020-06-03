import pandas as pd
import numpy as np
import math
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

X = np.asarray(pd.read_csv("./MNIST/train.csv",encoding='latin-1'))
Val = np.asarray(pd.read_csv("./MNIST/val.csv",encoding='latin-1'))
Test = np.asarray(pd.read_csv("./MNIST/test.csv",encoding='latin-1'))

#Extract classes 7 and 8 for binary classification

#=================================TRAIN============================================#

X_train = []
X_neg = []
X_pos = []
Y_train = []

for i in range(len(X)):
    if (X[i][-1]==7 or X[i][-1]==8):
        X_train.append(X[i][:-1])
        if(X[i][-1]==7):
            Y_train.append(-1)
            X_neg.append(X[i][:-1])
            X_neg = X_neg
        else:
            Y_train.append(1)
            X_pos.append(X[i][:-1])

X_neg = np.asarray(X_neg)
X_neg = X_neg/255
X_pos = np.asarray(X_pos)
X_pos = X_pos/255
X_train = np.asarray(X_train)
X_train = X_train/255
Y_train = np.asarray(Y_train)

m,n = X_train.shape
Y_train = Y_train.reshape(-1,1) * 1.

print(np.shape(X_train))
# print(np.shape(Y_train))
# print(np.shape(X_neg))
# print(np.shape(X_pos))

#=================================VALIDATION============================================#

X_val = []
Y_val = []

for i in range(len(Val)):
    if (Val[i][-1]==7 or Val[i][-1]==8):
        X_val.append(Val[i][:-1])
        if(Val[i][-1]==7):
            Y_val.append(-1)
        else:
            Y_val.append(1)

X_val = np.asarray(X_val)
X_val = X_val/255.
Y_val = np.asarray(Y_val)

m_val,n_val = X_val.shape
Y_val = Y_val.reshape(-1,1) * 1.

# print(np.shape(X_val))
# print(np.shape(Y_val))

#=================================TEST============================================#

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

m_test,n_test = X_test.shape
Y_test = Y_test.reshape(-1,1) * 1.

# print(np.shape(X_test))
# print(np.shape(Y_test))


clf = SVC(C = 1,kernel = 'linear')
# clf = SVC(C = 1,gamma=0.05, kernel = 'rbf')
clf.fit(X_train,Y_train.ravel())
print(clf.n_support_[0] + clf.n_support_[1]) # Number of support vectors

correct = 0
y_pred = clf.predict(X_val)
for i in range(len(y_pred)):
    if(y_pred[i]==Y_val[i][0]):
        correct+=1
        
print('Correct:',correct,'/',len(X_val))
print('Accuracy:',float(correct*100/len(X_val)),'%')

correct = 0
y_pred = clf.predict(X_test)
for i in range(len(y_pred)):
    if(y_pred[i]==Y_test[i][0]):
        correct+=1
        
print('Correct:',correct,'/',len(X_test))
print('Accuracy:',float(correct*100/len(X_test)),'%')
