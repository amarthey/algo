# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:17:26 2017

@author: antoinemarthey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init(mu_1,mu_2,mu_3,cov_1,cov_2,cov_3):
    mu=[]
    mu.append(mu_1)
    mu.append(mu_2)
    mu.append(mu_3)
    cov=[]
    cov.append(cov_1)
    cov.append(cov_2)
    cov.append(cov_3)
    return mu,cov

def points(mu,cov,weights):
    index = np.random.choice(len(weights),500, p = weights)
    obs = []
    for i in range(len(index)):
        x = np.random.multivariate_normal(mu[index[i]], cov[index[i]])
        obs.append(x)
    observations = np.array(obs)
    return observations,index

def init_centroids(observations, k):
    init_centro = []
    index = np.random.choice(len(observations),k)
    for i in range(len(index)):
        init_centro.append(observations[index[i]])
    init_centroid = np.array(init_centro)
    return init_centroid

def euclidean_dist(observations, centroids):
    dist = np.zeros((len(observations),len(centroids)))
    idx = np.zeros((len(observations),1))
    objective_function = 0
    for i in range(len(observations)): 
        for j in range(len(centroids)):
            dist[i,j] = np.linalg.norm(observations[i]-centroids[j])
        idx[i] = np.argmin(dist[i,:])     
        objective_function += np.min(dist[i,:])
    return dist,idx,objective_function

def update_centroids(observations,index,centroids):   
    unique, counts = np.unique(index, return_counts=True)
    dict_n = dict(zip(unique, counts))
    new_centroids = np.zeros((len(centroids),2))
    new_k = []
    observations_2 = pd.DataFrame(observations)
    observations_2['index'] = index
    for k in range(len(centroids)):
        new_k.append(dict_n.get(k))
        new_centroids[k,0] =  1.0 / (dict_n.get(k)) * observations_2[index==k][0].sum()
        new_centroids[k,1] =  1.0 / (dict_n.get(k)) * observations_2[index==k][1].sum()
    return new_centroids
    
def iteration(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3,weights,K,iteration_value): 
    obj_fun = []
    mu , cov = init(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3)
    observations, index = points(mu,cov,weights)
    centroids = init_centroids(observations, K)
    for i in range(iteration_value):
        d,idx,o = euclidean_dist(observations, centroids)
        index = idx
        centroids = update_centroids(observations,index,centroids)
        obj_fun.append(o)
    return observations, obj_fun, idx
         

mean_1 = [0, 0]
cov_1 = [[1, 0], [0, 1]]
mean_2 = [3, 0]
cov_2 = [[1, 0], [0, 1]]
mean_3 = [0, 3]
cov_3 = [[1, 0], [0, 1]]
weights = [0.2,0.5,0.3]

observations_K2, final_objective_K2, final_index_K2 = iteration(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3,weights,2,20)
observations_K3, final_objective_K3, final_index_K3 = iteration(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3,weights,3,20)
observations_K4, final_objective_K4, final_index_K4 = iteration(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3,weights,4,20)
observations_K5, final_objective_K5, final_index_K5 = iteration(mean_1,mean_2,mean_3,cov_1,cov_2,cov_3,weights,5,20)

plt.figure(1)
plt.plot(final_objective_K2,label ='K=2')
plt.plot(final_objective_K3,label ='K=3')
plt.plot(final_objective_K4,label ='K=4')
plt.plot(final_objective_K5,label ='K=5')
plt.legend()
plt.show()

fig_1 = plt.figure(2)
K_means_3_plot = fig_1.add_subplot(111)
scatter = K_means_3_plot.scatter(*zip(*observations_K3),c=final_index_K3)
K_means_3_plot.set_xlabel('x_coor')
K_means_3_plot.set_ylabel('y_coor')

fig_1.show()

fig_2 = plt.figure(3)
K_means_5_plot = fig_2.add_subplot(111)
scatter = K_means_5_plot.scatter(*zip(*observations_K5),c=final_index_K5)
K_means_5_plot.set_xlabel('x_coor')
K_means_5_plot.set_ylabel('y_coor')

fig_2.show()



#loads the datasets
ratings = pd.read_csv("/Users/antoinemarthey/Desktop/COMS4721_hw4-data/ratings.csv", sep=',',names=['user', 'movie', 'rating'])
ratings_test = pd.read_csv("/Users/antoinemarthey/Desktop/COMS4721_hw4-data/ratings_test.csv", sep=',',names=['user', 'movie', 'rating'])


def initialization(l,d,number):
    x = np.random.normal(0, 1.0/l, size=(d, number))
    return x


def facto(ratings, l, d, sigma, number_of_iteration):
 
    ratings_pivot = ratings.pivot(index='user', columns='movie', values='rating') 
    ratings_pivot_value = ratings_pivot.values
    number_user, number_video = ratings_pivot_value.shape
    v = initialization(l,d,number_video)
    u = initialization(l,d,number_user)    
    objective_function = []
    for number in range(number_of_iteration):
        for i in xrange(number_user):
            list_of_components = np.where(~np.isnan(ratings_pivot_value[i]))[0]
            matrix_multi = sum([np.dot(v[:, [k]], v[:, [k]].T) for k in list_of_components])
            scalar_matrix = sum([ratings_pivot_value[i, k] * v[:, [k]] for k in list_of_components])
            u[:, [i]] = np.dot(np.linalg.inv(l * sigma * np.identity(d) + matrix_multi), scalar_matrix)
        for j in xrange(number_video):
            list_of_components_2 = np.where(~np.isnan(ratings_pivot_value[:, j]))[0]
            if len(list_of_components_2) == 0:
                continue
            matrix_multi_2 = sum(np.dot(u[:, [k]], u[:, [k]].T) for k in list_of_components_2)
            scalar_matrix_2 = sum([ratings_pivot_value[k, j] * u[:, [k]] for k in list_of_components_2])
            v[:, [j]] = np.dot(np.linalg.inv(l * sigma * np.identity(d) + matrix_multi_2), scalar_matrix_2)    
        l =  - (0.5* (sigma**2) * np.nansum(np.square(ratings_pivot_value - (np.transpose(u)).dot(v)))) - (np.sum(0.5*l * np.power(np.linalg.norm(u, axis = 1), 2))) - (np.sum(0.5*l * np.power(np.linalg.norm(v, axis = 0), 2))) 
        objective_function.append(l)
    return objective_function

    
l = 1
d = 10
sigma = 0.25
number_of_iteration = 100
loop = 10

a = facto(ratings, l, d, sigma,number_of_iteration)

plt.figure(1)
plt.plot(a)
plt.show()


ratings_pivot_test = ratings.pivot(index='user', columns='movie', values='rating') 
ratings_pivot_test_value = ratings_pivot_test.values


def final(ratings, l, d, sigma, number_of_iteration, loop):
    final_obj =  []   
    for i in range(loop):
        o = facto(ratings, l, d, sigma, number_of_iteration)
        final_obj.append(o[99])
    return final_obj

#final(ratings, l, d, sigma, number_of_iteration, loop)      
    
def RMSE(ratings, ratings_test, l, d, sigma, number_of_iteration):
    ratings_pred = facto(ratings, l, d, number_of_iteration)
    diff = ratings_test[:,2] - ratings_pred
    somme = []
    addition = 0
    for i in range(len(diff)):
        somme.append(diff[i]**2)
    for i in range(len(somme)):
        addition += somme[i]
    somme_final = addition / len(somme)
    somme_sqrt = np.sqrt(somme_final)
    return somme_sqrt