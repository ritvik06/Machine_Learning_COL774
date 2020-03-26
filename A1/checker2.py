import numpy as np

x = np.loadtxt('./q3/logisticX.csv',delimiter=',')
y = np.loadtxt('./q3/logisticY.csv').reshape(-1,1)

x = (x - np.mean(x))/np.std(x)                  # Normalize
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)

############ Part (a) ############

theta = np.zeros((x.shape[1],1))
sigmoid = lambda z: 1/(1 + np.exp(z))
pred = sigmoid(np.dot(x,theta))
diff = y - pred
cost = -1*np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))
iterations = 0

while(cost > 0.0001 and iterations < 10):       # Stopping Criteria
    hessian = np.dot(np.dot(x.T,np.diag((pred*(1-pred)).reshape(-1,))),x)
#    print(hessian)
    print(pred)

#    print(np.dot(x.T,diff))
    theta = theta - np.dot(np.linalg.pinv(hessian),np.dot(x.T,diff))
#    print(theta)
    pred = sigmoid(np.dot(x,theta))
    diff = y - pred
    cost = -1*np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))
    iterations += 1
    #print(iterations," - ",cost)

print("Final Cost - ",cost)
print("Final Parameters - {0},{1},{2}".format(theta[0],theta[1],theta[2]))

# theta[0] = -0.2130308
# theta[1] = -2.65801937
# theta[2] = 2.66106075

