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

#==============================TRAINING DATASET================================    

X_train = []
Y_train = []

for i in range(len(X)):
    X_train.append(X[i][:-1])
    Y_train.append(X[i][-1])

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

# print(Y_train)

#==============================VALIDATION DATASET================================    


X_val = []
Y_val = []

for i in range(len(Val)):
    X_val.append(Val[i][:-1])
    Y_val.append(Val[i][-1])

X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)

#==============================TEST DATASET================================    

X_test = []
Y_test = []

for i in range(len(Test)):
    X_test.append(Test[i][:-1])
    Y_test.append(Test[i][-1])

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

def extract_Xdata(num1 : int,num2 : int):
    X_num = []
    
    for i in range(len(X)):
        if (int(X[i][-1])==num1 or int(X[i][-1])==num2):
            X_num.append(X[i][:-1])

    return X_num

def extract_Ydata(num1 : int,num2 : int):
    Y_num = []
    
    for i in range(len(X)):
        if (X[i][-1]==num1 or X[i][-1]==num2):
            if(X[i][-1]==num1):
                Y_num.append(-1)
            elif(X[i][-1]==num2):
                Y_num.append(1)
        
    return Y_num

# print(np.shape(extract_Xdata(7,8)))

# clf = SVC(C = 1,kernel = 'linear')
# # clf = SVC(C = 1,gamma=0.05, kernel = 'rbf')
# clf.fit(X_dash,Y_dash)
dict_Xdata = {}
dict_Ydata = {}
dict_par = {}
Y_res = []

for i in range(9):
    for j in range(i+1,10,1):
        dict_Xdata[(i,j)] = extract_Xdata(i,j)
        dict_Ydata[(i,j)] = extract_Ydata(i,j)
        
        
print('====================DONE 1================================================#')


for i in range(1):
    for j in range(i+1,10,1):
        clf = SVC(C = 1,kernel = 'linear')
        clf.fit(dict_Xdata[(i,j)],dict_Ydata[(i,j)])
        print(clf)
        dict_par[(i,j)] = clf
        
print('====================DONE 2================================================#')


for i in len(X_val):
    votes = np.zeros(10)
    for j in range(9):
        for k in range(j+1,10):
            res = dict_par[(j,k)].predict(np.expand_dims(X_val[i],axis=0))[0]
    
            if(res==-1):
                votes[j]+=1
            elif(res==1):
                votes[k]+=1
    
    Y_res.append(np.max(votes))
    
print('====================DONE 3================================================#')

for i in range(len(Y_res)):
    if(Y_res[i]==Y_val[i][0]):
        correct+=1
        
print('Correct:',correct,'/',len(X_val))
print('Accuracy:',float(correct*100/len(X_val)),'%')

for i in range(len(Y_res)):
    if(Y_res[i]==Y_test[i][0]):
        correct+=1
        
print('Correct:',correct,'/',len(X_val))
print('Accuracy:',float(correct*100/len(X_val)),'%')

