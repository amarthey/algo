from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg

points = pd.read_csv("/Users/antoinemarthey/Desktop/CFB2016_scores.csv", sep=',',names=['Team_A', 'Team_A_pt', 'Team_B', 'Team_B_pt'])
teams = pd.read_table('/Users/antoinemarthey/Desktop/TeamNames.txt', names=['Teams'])

points['Victory_A'] = np.where(points['Team_A_pt'] > points['Team_B_pt'], 1, 0)

M_t = np.zeros((760,760))

def update(points):
    for index, row in points.iterrows():
        M_t[row['Team_A']-1,row['Team_A']-1] = M_t[row['Team_A']-1,row['Team_A']-1] + row['Victory_A'] + (row['Team_A_pt'] / (row['Team_B_pt'] + row['Team_A_pt']))
        M_t[row['Team_B']-1,row['Team_B']-1] = M_t[row['Team_B']-1,row['Team_B']-1] + (1-row['Victory_A']) + (row['Team_B_pt'] / (row['Team_B_pt'] + row['Team_A_pt']))
        M_t[row['Team_A']-1,row['Team_B']-1] = M_t[row['Team_A']-1,row['Team_B']-1] + (1-row['Victory_A']) + (row['Team_B_pt'] / (row['Team_B_pt'] + row['Team_A_pt']))
        M_t[row['Team_B']-1,row['Team_A']-1] = M_t[row['Team_B']-1,row['Team_A']-1] + row['Victory_A'] + (row['Team_A_pt'] / (row['Team_B_pt'] + row['Team_A_pt']))
    return M_t

M_updated = update(points)

def normalize_line(M):
    return np.divide(np.transpose(M),M.sum(axis=1))

M_updated_normalized = np.transpose(normalize_line(M_updated))

w_0 = np.random.uniform(0,1,760)
w_0 /= sum(w_0)

team_name = pd.read_csv('/Users/antoinemarthey/Desktop/TeamNames.txt', header=None)
matrix_v=np.zeros((760,4))
index=0
for t in [10, 100, 1000, 10000]:
    matrix_v[:,index] = np.dot(np.linalg.matrix_power(np.transpose(M_updated_normalized), t), w_0)
    index += 1

teams_10 = pd.DataFrame(matrix_v[:,0], columns=['rank'])
teams_10['team_name'] = team_name
teams_100 = pd.DataFrame(matrix_v[:,1], columns=['rank'])
teams_100['team_name'] = team_name
teams_1000 = pd.DataFrame(matrix_v[:,2], columns=['rank'])
teams_1000['team_name'] = team_name
teams_10000 = pd.DataFrame(matrix_v[:,3], columns=['rank'])
teams_10000['team_name'] = team_name

ranking_10 = teams_10.sort_values(by='rank', ascending=False)[:25]
ranking_100 = teams_100.sort_values(by='rank', ascending=False)[:25]
ranking_1000 = teams_1000.sort_values(by='rank', ascending=False)[:25]
ranking_10000 = teams_10000.sort_values(by='rank', ascending=False)[:25]
print ranking_10
print ranking_100
print ranking_1000
print ranking_10000

Matrix = np.transpose(M_updated_normalized)
a,b = scipy.sparse.linalg.eigs(Matrix, k=1, sigma=1.0)
v_inf = b / sum(x for x in b)

matrix_update = np.zeros((760,10000))
matrix_update[:,0] = w_0
for t in range(1,10000):
    matrix_update[:,t] = np.dot(np.transpose(M_updated_normalized), matrix_update[:,t-1])

diff=[np.sum(np.absolute((v_inf - matrix_update[:,t]/np.sum(matrix_update[:,t])))) for t in range(10000)]

plt.figure(1)
plt.plot(diff)
plt.show()


X = np.zeros((3012,8447))
N1,N2 = X.shape
file = open('/Users/antoinemarthey/Desktop/nyt_data.txt', 'r')
i = 0
for line in file:
    currentline = line.split(",")
    for j in range(len(currentline)):
        X[int(currentline[j].split(":")[0])-1,i] = int(currentline[j].split(":")[1])
    i += 1 
nyt_vocab = pd.read_table('/Users/antoinemarthey/Desktop/nyt_vocab.txt')

K = 25
W = np.random.uniform(1,2,size=(N1,K))
H = np.random.uniform(1,2,size=(K,N2))

def objective_function(X,W,H):
    WH = W.dot(H)
    somme = X * np.log(WH + 10**-16) - WH
    objective_function = - np.sum(somme)
    return objective_function

def NMF(H,W,X,K,iteration_value):
    objective = []
    for i in range(iteration_value):
        matrix_1 = np.transpose(W)/(np.expand_dims(np.sum(np.transpose(W), axis=1), 1)+10**-16)
        matrix_2 = X / (np.dot(W, H)+10**-16)
        H = H*np.dot(matrix_1, matrix_2)
        matrix_3 = X / (np.dot(W, H)+10**-16)
        matrix_4 = np.transpose(H)/(np.transpose(np.expand_dims(np.sum(np.transpose(H), axis=0), 1))+10**-16)
        W = W*np.dot(matrix_3, matrix_4)
        objective.append(objective_function(X,W,H))
    return objective, W

result, W_result = NMF(H,W,X,25,100)

plt.figure(2)
plt.plot(result)
plt.show()

def normalize_col(M):
    return np.divide(M,M.sum(axis=0))

W_Norm = normalize_col(W_result)
liste = []
for i in range(W_Norm.shape[1]):
    to_sort_list=zip(np.linspace(0,W_Norm.shape[0]-1,W_Norm.shape[0]-1),W_Norm[:,i])
    sorted_list=sorted(to_sort_list,key=lambda item:item[1],reverse=True)
    liste.append(sorted_list)
a = pd.DataFrame(liste)
b = pd.DataFrame.transpose(a)
c = b.iloc[0:10,:]
for x in range(25):
    for y in range(10):
        t = list(b.iloc[y,x])
        lst = list(t)
        lst[0] = nyt_vocab.iloc[int(lst[0])-1]
        t = tuple(lst)
        c.iloc[y,x] = t

pd.set_option('max_rows', 99)
print c.iloc[:,0:10]
print c.iloc[:,10:30]

