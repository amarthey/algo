# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:47:02 2017

@author: antoinemarthey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import special
from collections import Counter


#loads the datasets
y_train = np.genfromtxt('/Users/antoinemarthey/Desktop/y_train.csv',delimiter=",", dtype="int")
y_test = np.genfromtxt('/Users/antoinemarthey/Desktop/y_test.csv',delimiter=",", dtype="int")
x_train = np.genfromtxt('/Users/antoinemarthey/Desktop/X_train.csv', delimiter=",", dtype="int")
x_test = np.genfromtxt('/Users/antoinemarthey/Desktop/X_test.csv', delimiter=",", dtype="int")

## Question 1
class BayesClassifier(object):
    def __init__(self):
        self.param_bernoullis = np.zeros((2,54))
        self.param_pareto = np.zeros((2,3))
    def train(self, x_train, y_train):
        self.param_prior = np.divide(float(sum(y_train)),float(len(y_train)))
        # Bernoulli/Pareto parameters when y = 1
        self.param_bernoullis[0,:] = np.mean(x_train[y_train == 0, 0:54], axis = 0)
        self.param_pareto[0,0] = (len(y_train) - np.sum(y_train))/sum(np.log(x_train[y_train == 0,54]))
        self.param_pareto[0,1] = (len(y_train) - np.sum(y_train))/sum(np.log(x_train[y_train == 0,55]))
        self.param_pareto[0,2] = (len(y_train) - np.sum(y_train))/sum(np.log(x_train[y_train == 0,56]))
        # Bernoulli/Pareto parameters when y = 0
        self.param_bernoullis[1,:] = np.mean(x_train[y_train == 1, 0:54], axis = 0)
        self.param_pareto[1,0] = np.sum(y_train)/sum(np.log(x_train[y_train == 1,54]))
        self.param_pareto[1,1] = np.sum(y_train)/sum(np.log(x_train[y_train == 1,55]))
        self.param_pareto[1,2] = np.sum(y_train)/sum(np.log(x_train[y_train == 1,56]))
    def predict(self, x_test):
        self.posterior = np.zeros((2, len(x_test)))
        pred = np.zeros(len(x_test), dtype="int")
        for i in range(len(x_test)):
            self.posterior[0, i] = ((1 - self.param_prior) * np.prod(1 - np.abs((self.param_bernoullis[0,:] - x_test[i, 0:54]))) * 
            self.param_pareto[0,0]*(x_test[i,54] ** -(self.param_pareto[0,0] + 1)) *
            self.param_pareto[0,1]*(x_test[i,55] ** -(self.param_pareto[0,1] + 1)) *
            self.param_pareto[0,2]*(x_test[i,56] ** -(self.param_pareto[0,2] + 1)))
            self.posterior[1, i] = (self.param_prior * np.prod(1 - np.abs((self.param_bernoullis[1,:] - x_test[i, 0:54])))* 
            self.param_pareto[1,0]*(x_test[i,54] ** -(self.param_pareto[1,0]+1)) *
            self.param_pareto[1,1]*(x_test[i,55] ** -(self.param_pareto[1,1]+1)) *
            self.param_pareto[1,2]*(x_test[i,56] ** -(self.param_pareto[1,2]+1)))
            if self.posterior[0,i] > self.posterior[1, i]:
                pred[i] = 0
            else:
                pred[i] = 1 
        return pred

bayes = BayesClassifier()
bayes.train(x_train, y_train)
pred_BC = bayes.predict(x_test)
Final_Accuracy = 1 - (np.mean(np.abs(pred_BC - y_test)))
print pd.crosstab(pred_BC,y_test) 
print "Accuracy is equal to " , round(Final_Accuracy*100,2), "%"

## Question 2
class BayesClassifier_parameters(object):
    def __init__(self):
        self.param_bernoullis = np.zeros((2,54))
    def train_0(self, x_train, y_train):
        self.param_prior = np.divide(float(sum(y_train)),float(len(y_train)))
        # Bernoulli/Pareto parameters when y = 1
        self.param_bernoullis[0,:] = np.mean(x_train[y_train == 0, 0:54], axis = 0)
        return self.param_bernoullis[0,:]
    def train_1(self, x_train, y_train):
        # Bernoulli/Pareto parameters when y = 0
        self.param_bernoullis[1,:] = np.mean(x_train[y_train == 1, 0:54], axis = 0)
        return self.param_bernoullis[1,:]
        
bayes_para = BayesClassifier_parameters()

parameters_Bernoulli_1 = bayes_para.train_1(x_train, y_train)
parameters_Bernoulli_0 = bayes_para.train_0(x_train, y_train)
plt.figure(1)
plt.stem(range(0,54),parameters_Bernoulli_1,markerfmt='o', label='y=1')
plt.stem(range(0,54),parameters_Bernoulli_0,markerfmt='x', label='y=0')
print parameters_Bernoulli_1[15]
print parameters_Bernoulli_0[15]
print parameters_Bernoulli_1[51]
print parameters_Bernoulli_0[51]
plt.legend()
plt.show()

## Question 3
class KNN(object):
    def __init__(self):
        pass
    def train(self, x_train, y_train):
        self.x_train = x_train # (x_train - np.mean(x_train, axis = 0))/np.std(x_train, axis = 0)
        self.y_train = y_train
        self.length = len(y_train)
    def classify_one_point(self, x, n_neighbors):
        sum_neighbors = 0
        dist = np.zeros(self.length)
        for i in range(self.length):
            dist[i] = np.sum(np.abs(x - x_train[i]))
        for j in range(n_neighbors):
            idx = np.argmin(dist)
            sum_neighbors += y_train[idx]
            dist[idx] = float("inf")
        if float(sum_neighbors)/ float(n_neighbors) > 0.5:
            return 1
        else:
            return 0
    def predict(self, x_test, n_neighbors):
        pred = np.zeros(len(x_test), dtype="int")
        for i in range(len(x_test)):
            pred[i] = self.classify_one_point(x_test[i], n_neighbors)
        return pred  
    def plotknn(self, accuracy):
        for i in range(1,21):
            accuracy[i-1] = round((1 - (np.mean(np.abs(self.predict(x_test,i) - y_test))))*100,2)
        return accuracy
accuracy_var = [0]*20 
knn = KNN()
knn.train(x_train, y_train)
knn.plotknn(accuracy_var)
print accuracy_var
plt.figure(2)
plt.plot(accuracy_var)
plt.show()

## Question 4

# def new data
def new_y_data(y_train):
    new_y_train = [0] * len(y_train)
    for i in range(len(y_train)):
        if y_train[i] == 0:
            new_y_train[i] = -1
        else:
            new_y_train[i] = y_train[i]
    return new_y_train
def new_x_data(x_train):
    new_x_train =  [0] * len(x_train)   
    for i in range(len(x_train)):
        new_x_train[i] = np.insert(x_train[i],0,1)
    return pd.DataFrame(new_x_train)
new_x_train = new_x_data(x_train)
new_y_train = new_y_data(y_train)

class SAA(object):
    def __init__(self):
        self.x_train = x_train 
        self.y_train = y_train
        self.length = len(y_train)
        self.transpose = np.transpose(x_train)
    def iteration(self, x_train, y_train, t):
        self.param_w = np.zeros(x_train.shape[1],dtype='float64')
        self.objective_function = []
        for i in range(0,t):
            iteration = 1 / ((10**5)*np.sqrt(i+1))
            matrix_multiplication = (x_train.dot(self.param_w)) * y_train
            matrix_multiplication_param = matrix_multiplication.apply(lambda x : 1 - scipy.special.expit(x))*y_train
            matrix_multiplication_param = matrix_multiplication_param.dot(x_train)
            self.param_w += iteration * matrix_multiplication_param           
            matrix_multiplication_updated_w = (x_train.dot(self.param_w)) * y_train
            calcul_of = matrix_multiplication_updated_w.apply(lambda x : np.log(scipy.special.expit(x)+ 0.00001))
            objective_ft = calcul_of.sum()
            self.objective_function.append(objective_ft)
        x = [i+1 for i in range(t)]
        plt.figure(3)        
        plt.plot(x,self.objective_function)
        plt.show()
        return self.param_w, self.objective_function

saa = SAA()
para_10000,of_10000 = saa.iteration(new_x_train,new_y_train,10000)


# Question 5
class Newton(object):
    def __init__(self):
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.length = len(y_train)
        self.transpose = np.transpose(x_train)
    def iteration(self, x_train, y_train, t):
        self.param_w = np.zeros(x_train.shape[1],dtype='float64')
        self.objective_function = []
        for i in range(0,t):
            iteration = 1 / (np.sqrt(i+1))
            matrix_multiplication = (x_train.dot(self.param_w)) * y_train
            matrix_second_ordre_inverse = matrix_multiplication.apply(lambda x : -1 * scipy.special.expit(x)*(1-scipy.special.expit(x)))*np.transpose(x_train)
            matrix_second_ordre = matrix_second_ordre_inverse.dot(x_train)
            matrix_second_ordre_inverse = np.linalg.inv(matrix_second_ordre)
            matrix_multiplication_param = matrix_multiplication.apply(lambda x : 1 - scipy.special.expit(x))*y_train
            matrix_multiplication_param = matrix_multiplication_param.dot(x_train)
            self.param_w += - iteration * matrix_second_ordre_inverse.dot(matrix_multiplication_param)
            matrix_multiplication_updated_w = (x_train.dot(self.param_w)) * y_train
            calcul_of = matrix_multiplication_updated_w.apply(lambda x : np.log(scipy.special.expit(x)))
            objective_ft = calcul_of.sum()
            self.objective_function.append(objective_ft)
        x = [i+1 for i in range(t)]
        plt.figure(4)        
        plt.plot(x,self.objective_function)
        plt.show()        
        return self.param_w, self.objective_function
    def predict_newton(self,x_test,w): 
        accu = x_test.dot(w)
        predicted_y = [0] * len(accu)
        for i in range(len(accu)):
            if accu[i] > 0:
                predicted_y[i] = 1
            else:
                predicted_y[i] = -1
        return predicted_y
   

newton = Newton()
para_newton_100,of_newton_100 = newton.iteration(new_x_train,new_y_train,100)
maxi_ob = of_newton_100.index(max(of_newton_100))
maxi_w,of_newton_98 = newton.iteration(new_x_train,new_y_train,maxi_ob) 
new_x_test = new_x_data(x_test)
new_y_test = new_y_data(y_test)
accuracy_newton = newton.predict_newton(new_x_test,maxi_w)
Final_compare = [0] * len(accuracy_newton)
for i in range(len(accuracy_newton)):
    if accuracy_newton[i] == new_y_test[i]:
        Final_compare[i] = 0
    else:
        Final_compare[i] =1

print Final_compare
Counter(Final_compare)    


