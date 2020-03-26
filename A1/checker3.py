import numpy as np
import csv
import math

#Preparing dataset for processing
X_train = np.loadtxt('./q3/logisticX.csv',delimiter=',')
Y_train = np.loadtxt('./q3/logisticY.csv')
Y_train = Y_train.reshape(len(Y_train),1)
theta = np.zeros((3,1))

#Keep copies before normalisation step

X_copy = X_train
Y_copy = Y_train

#Normalise values

M = len(X_train)

X_train = (X_train - np.mean(X_train)) / (np.std(X_train))
X_train = np.hstack((np.reshape(np.ones(M), (M, 1)), X_train))

# print(theta)

def sigmoid(array):
    
    for i in range(len(array)):
#        print("entered")
#         print(array[i][0])
        array[i][0] = 1/(1+np.exp(-array[i]))
    return array

def cost_func(X,Y,Theta):
    mul = np.dot(X,Theta)
    activ = sigmoid(mul)

    cost = -1*np.sum(Y*np.log(activ) + (1-Y)*np.log(1-activ))
#     print(cost)
    return cost

# cost_func(X_train,Y_train,theta)

def derivative(X,Y,Theta,count):
    #derivative for one sample, have to sum it over m samples
    
    if count<(M+1):
        mul = np.dot(X,Theta)
#         print("Mul ",mul)
        activ = sigmoid(mul)
#         print("Activ ",activ)
        dif = Y-activ
        return np.dot(X.T,dif)
    else:
        return -1

def hessian(X,Y,Theta):
    mul = np.dot(X,Theta)
    activ = sigmoid(mul)
    hes = np.dot(np.dot(X.T,np.diag((activ*(1-activ)).reshape(-1,))),X)
    return hes

count = 0

while(count<10):
    mul = np.dot(X_train,theta)
#     print(theta)
#     print(X_train[53])

    activ = sigmoid(mul)
    
#     print(activ)

    #Y-sigmoid(Theta(T).X)
    dif = Y_train-activ
    
    hes = hessian(X_train,Y_train,theta)
    print(hes)

#     print(derivative(X_train,Y_train,theta,count))
#     print(np.linalg.pinv(hes))
    
#     print(np.dot(np.linalg.pinv(hes),derivative(X_train,Y_train,theta,count)))
    theta = theta - np.dot(np.linalg.pinv(hes),derivative(X_train,Y_train,theta,count))
        
    #Theta now updated, calculate params again
    
#     print(cost_func(X_train, Y_train, theta))
    count+=1

# print("Final Cost ", cost_func(X_train, Y_train, theta))
    
