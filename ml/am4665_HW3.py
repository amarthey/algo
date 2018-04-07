# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:46:26 2017

@author: antoinemarthey
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#loads the datasets
y_train = np.genfromtxt('/Users/antoinemarthey/Desktop/gaussian_process/y_train.csv',delimiter=",", dtype="float")
y_test = np.genfromtxt('/Users/antoinemarthey/Desktop/gaussian_process/y_test.csv',delimiter=",", dtype="float")
x_train = np.genfromtxt('/Users/antoinemarthey/Desktop/gaussian_process/X_train.csv', delimiter=",", dtype="float")
x_test = np.genfromtxt('/Users/antoinemarthey/Desktop/gaussian_process/X_test.csv', delimiter=",", dtype="float")

class Gaussian_Process_RMSE(object):
    def __init__(self):
        pass

    def Dist(self,x,y,b):      
        dist = 0
        dist_vect = x - y
        for i in range(len(dist_vect)):
            dist += dist_vect[i]**2
        return dist
    def Kernel(self,point,dataset,b):
        l = len(dataset)
        param = -1/b 
        K = np.zeros(l)
        for i in range(l):
            K[i] = np.exp(param * self.Dist(point,dataset[i,:],b))
        return K

    def Kernel_n(self,x_data,b):
        l = len(x_data)
        Kn = np.zeros((l,l))
        for i in range(l):
            Kn[i] = self.Kernel(x_data[i,:],x_data,b)
        return Kn
        
    def Mu(self,x_test, x_train, y_train, b,sigma, Kn):        
        sigma_mat = sigma * np.identity(len(x_train))
        inverse = np.linalg.inv(Kn + sigma_mat )
        res = np.zeros(len(x_test))
        for i in range(len(x_test)):
            multi = self.Kernel(x_test[i,:],x_train,b).dot(inverse)
            res[i] = multi.dot(y_train)
        return res

    def RMSE(self, x_test, x_train, y_train, y_test, b , sigma, Kn):
        y_pred = self.Mu(x_test, x_train, y_train, b,sigma,Kn)
        diff = y_pred - y_test
        somme = []
        addition = 0
        for i in range(len(diff)):
            somme.append(diff[i]**2)
        for i in range(len(somme)):
            addition += somme[i]
        somme_final = addition / len(somme)
        somme_sqrt = np.sqrt(somme_final)
        return somme_sqrt
        
    def RMSE_var(self, x_test, x_train, y_train, y_test, b , sigma):
        RMSE_matrix = np.zeros((len(b), len(sigma)))
        for i in range(len(b)):
            ker = self.Kernel_n(x_train,b[i])
            for j in range(len(sigma)):
                RMSE_matrix[i,j] = self.RMSE(x_test, x_train, y_train, y_test, b[i] , sigma[j], ker)
        return RMSE_matrix

class Gaussian_Process(object):
    def __init__(self):
        pass
    def Dist(self,x,y,b):      
        dist_vect = x - y
        return dist_vect**2
    def Kernel(self,point,dataset,b):
        l = len(dataset)
        param = -1/b 
        K = np.zeros(l)
        for i in range(l):
            K[i] = np.exp(param * self.Dist(point,dataset[i],b))
        return K     
    def Kernel_n(self,x_data,b):
        l = len(x_data)
        Kn = np.zeros((l,l))
        for i in range(l-1):
            Kn[i] = self.Kernel(x_data[i],x_data,b)
        return Kn

    def Mu(self,x_test, x_train, y_train, b,sigma, Kn):        
        sigma_mat = sigma * np.identity(len(x_train))
        inverse = np.linalg.inv(Kn + sigma_mat )
        res = np.zeros(len(x_test))
        for i in range(len(x_test)):
            multi = self.Kernel(x_test[i],x_train,b).dot(inverse)
            res[i] = multi.dot(y_train)
        return res

GP = Gaussian_Process()
GP_RMSE = Gaussian_Process_RMSE()

# Question 2 

beta = [5, 7, 9, 11, 13, 15]
sigm = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

answer = GP_RMSE.RMSE_var(x_test, x_train, y_train, y_test, beta , sigm)
print answer
# Question 4

x_4_train = x_train[:,3]
x_4_test = x_test[:,3]

bet = 5
sigma = 2

K = GP.Kernel_n(x_4_train,bet)

y_test_pred = GP.Mu(x_4_test, x_4_train, y_train, bet , sigma, K)

list_sum=[list(x_4_train),list(y_train),y_test_pred]
list_sum=zip(*list_sum)
list_sum.sort(key=lambda x: x[0])
list_sum=zip(*list_sum)
x_Train_4,y_Train,y_Train_4=list_sum

plt.figure(1)
plt.scatter(x_4_train,y_train)
plt.plot(x_Train_4,y_Train_4)

## EXERCICE 2

#loads the datasets
y_train = np.genfromtxt('/Users/antoinemarthey/Desktop/y_train.csv',delimiter=",", dtype="float")
y_test = np.genfromtxt('/Users/antoinemarthey/Desktop/y_test.csv',delimiter=",", dtype="float")
x_train = np.genfromtxt('/Users/antoinemarthey/Desktop/X_train.csv', delimiter=",", dtype="float")
x_test = np.genfromtxt('/Users/antoinemarthey/Desktop/X_test.csv', delimiter=",", dtype="float")

def new_x_data(x_train):
    new_x_train =  [0] * len(x_train)   
    for i in range(len(x_train)):
        new_x_train[i] = np.insert(x_train[i],0,1)
    return new_x_train

new_x_train = np.array(new_x_data(x_train))
new_x_test = np.array(new_x_data(x_test))

class AdaBoost(object):
    def bootstrap(self,x_train,y_train,weights):
        x = np.array(x_train)
        y = np.array(y_train)
        idx = np.random.choice(len(x),len(x), p = weights)
        x = x[idx,:]
        y = y[idx]
        return x,y,idx
    
    def train(self,x_train,y_train):
        param_w = np.zeros((1,x_train.shape[1]))
        transpose = np.transpose(x_train)
        multi = transpose.dot(x_train)
        inverse_multi = np.linalg.inv(multi)
        multi_2 = inverse_multi.dot(transpose)
        param_w = multi_2.dot(y_train)
        return param_w

    def predict(self, x_test, p):
        pred = np.zeros(len(x_test), dtype="float")
        for i in range(len(x_test)):
            trans = np.transpose(x_test[i,:])
            pred[i] = np.sign(trans.dot(p))
        return pred

    def return_1(self,x,y):
        sol = []
        for i in range(len(x)):
            if x[i] == y[i]:
                sol.append(0)
            else:
                sol.append(1)
        return sol

    def error(self,x_train, y_train, x_test, y_test,t):
        alpha = []
        errors = []
        size_train = len(x_train)
        size_test = len(x_test)
        weights = np.zeros((t,size_train))
        w = np.zeros((t,size_train))
        weights[0,:] = np.ones(size_train)/size_train
        bootstrap_data = []
        Final_Accuracy = []
        final_res = [0] * size_test
        final_res_sign = [0] * size_test
        upper_bond = []
        calcul = 0
        for i in range(1,t):
            x,y,index = self.bootstrap(x_train,y_train,weights[i-1,:])
            bootstrap_data.append(index)
            param_w = self.train(x,y)
            prediction_train = self.predict(x_train,param_w)
            calcul_error = self.return_1(prediction_train,y_train)
            err = (calcul_error*weights[i-1,:]).sum()
            errors.append(err)
            alpha.append(0.5 * np.log((1-err)/err))
            final = 0.5 * np.log((1-err)/err) * self.predict(x_test,param_w)
            final_res += final
            calcul += (0.5 - err)**2
            upper_bond.append(np.exp(-2*calcul))
            for d in range(len(final_res)):
                final_res_sign[d] = np.sign(final_res[d])
            diff = self.return_1(final_res_sign, y_test)
            Final_Accuracy.append(np.mean(np.abs(diff)))
            for j in range(size_train):
                if calcul_error[j] == 1:
                    w[i,j] = weights[i-1,j] * np.exp(alpha[i-1])
                else: 
                    w[i,j] = weights[i-1,j] * np.exp(-alpha[i-1])
            weights[i,:] = w[i,:] / w[i,:].sum() 
        return errors, weights, alpha, bootstrap_data, Final_Accuracy, upper_bond
    

ada = AdaBoost()
e,w,a,b,f,u = ada.error(new_x_train,y_train,new_x_train,y_train,1500)
e_test,w_test,a_test,b_test,f_test,u_test = ada.error(new_x_train,y_train,new_x_test,y_test,1500)

# Question 1

plt.figure(2)  
plt.plot(f,label='Training error')
plt.plot(f_test,label='Testing Error')
plt.show

# Question 2
plt.figure(3)
plt.plot(u)
plt.show
# Question 3 

def count_occurence(data):
    empty = []      
    dico = Counter(empty)
    for i in range(len(data)):
        dico += Counter(data[i])
    return dico
final = count_occurence(b)
plt.figure(4)
plt.bar(range(len(final)), final.values(), align='center')
plt.show

# Quesiton 4

plt.figure(5)
plt.plot(a)
plt.show
plt.figure(6)
plt.plot(e)
plt.show