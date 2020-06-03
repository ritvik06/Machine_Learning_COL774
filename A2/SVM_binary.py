import pandas as pd
import numpy as np
import math
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.spatial.distance import cdist

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

def gaussian(X1,X2,gamma):
    return (np.exp(gamma*cdist(np.expand_dims(X1,axis=0),np.expand_dims(X2,axis=0),'sqeuclidean')))

# gaussian([1,2,3,4,5],[1,2,3,4,6],-0.05)
# print(math.exp(-0.05))
# gaussian(X_dash[0],X_dash[1],-0.05)

#=================================PART A============================================#


#Initializing values and computing H. Note the 1. to force to float type
C = 1.
gamma = -0.05
X_dash = Y_train * X_train

H = np.ones((m,m)) * 1.

mode = 1

#Converting into cvxopt format - as previously
if mode==1:
    H = np.dot(X_dash , X_dash.T) * 1.
elif mode==2:
#     for i in range(m):
#         for j in range(m):
#             H[i][j] = gaussian(X_dash[i],X_dash[j],-0.05)

    dist = cdist(X_train,X_train,'sqeuclidean')
    K = np.exp(gamma*dist)
    H = (np.dot(Y_train,Y_train.T))*K

P = cvxopt_matrix(H)

q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(Y_train.reshape(1,-1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])


#==================Computing and printing parameters===============================#
if mode==1:    
    w = ((Y_train * alphas).T @ X_train).reshape(-1,1)
    # b = Y_train - np.dot(X_train, w)
    # print(b)
    # print(alphas)
    max = np.dot(w.T,X_neg[0])

    for i in range(len(X_neg)):
        if(np.dot(w.T,X_neg[i])>max):
            max = np.dot(w.T,X_neg[i])

    min = np.dot(w.T,X_pos[0])

    for i in range(len(X_pos)):
        if(np.dot(w.T,X_pos[i])<min):
            min = np.dot(w.T,X_pos[i])

    b = -1*((max[0] + min[0])/2)

S = []
for i in range(len(alphas)):
    if alphas[i][0]>1e-4:
        S.append(alphas[i])
print(len(S))
# print(S)

mode = 1
if mode==2:
    w_dash = (Y_train * alphas)
    w_neg = (alphas * -1)
    w_pos = alphas
    min,max = (0,0)
    for i in range(len(X_neg)):
        output_gauss = 0
        for j in range(m):
            output_gauss += w_neg[j][0]*gaussian(X_train[j],X_neg[i],-0.05)
#       print(output_gauss)
        if (min>output_gauss):
            min = output_gauss
        
    print("=====================DONE====================")
    
    for i in range(len(X_pos)):
        output_gauss = 0
        for j in range(m):
            output_gauss += w_pos[j][0]*gaussian(X_train[j],X_pos[i],-0.05)
#       print(output_gauss)
        if (max<output_gauss):
            max = output_gauss
            
    
    b = -1*((max + min)/2)
    print(b)

#===============Testing Val Accuracy================================================#
correct = 0
# print(b)
mode=1
for i in range(len(X_val)):
    output_gauss = 0

    if mode==1:
        output =  (np.dot(w.T,X_val[i])) + b
    #     print(output)
        if(output[0]>=0):
            ans = 1
        else:
            ans = -1

        if(int(Y_val[i][0])==ans):
            correct+=1
    #     print('Mine: ',ans,' Expected; ',int(Y_val[i][0]))
    
    elif mode==2:
        for j in range(m):
            output_gauss += w_dash[j][0]*gaussian(X_train[j],X_val[i],-0.05)
#             print(output_gauss)
        output_gauss += b[0][0]
        if(output_gauss>=0):
            ans = 1
        else:
            ans = -1
            
        if(int(Y_val[i][0])==ans):
            correct+=1

print('Correct:',correct,'/',len(X_val))
print('Accuracy:',float(correct*100/len(X_val)),'%')

#===============Testing Test Accuracy================================================#
correct = 0

for i in range(len(X_test)):
    output_gauss = 0
    if mode==1:
        output =  (np.dot(w.T,X_test[i])) + b
#         print(output)
        if(output[0]>=0):
            ans = 1
        else:
            ans = -1

        if(Y_test[i]==ans):
            correct+=1
        
    elif mode==2:
        for j in range(m):
            output_gauss += w_dash[j][0]*gaussian(X_train[j],X_test[i],-0.05)
#             print(output_gauss)
        output_gauss += b
        if(output_gauss>=0):
            ans = 1
        else:
            ans = -1
            
        if(int(Y_test[i][0])==ans):
            correct+=1

print('Correct:',correct,'/',len(X_test))
print('Accuracy:',float(correct*100/len(X_test)),'%')

