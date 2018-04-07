import numpy as np
from matplotlib import pyplot as plt

def import_data(y_train,x_train,y_test,x_test):
    x_Train = np.genfromtxt(x_train, delimiter=',')
    y_Train = np.genfromtxt(y_train, delimiter=',')
    x_Test = np.genfromtxt(x_test, delimiter=',')
    y_Test = np.genfromtxt(y_test, delimiter=',')
    return x_Train,y_Train,x_Test,y_Test
    
def w_RidReg(y,x,lambda_var):
    u_matrix,s_matrix,v_matrix=np.linalg.svd(x,full_matrices=False)
    s_lambda=np.fromiter((item/(lambda_var+item*item) for item in s_matrix),float)
    s_inverse_lambda=np.linalg.inv(np.diag(s_lambda))
    w_ridreg=np.transpose(v_matrix).dot(np.linalg.inv(s_inverse_lambda)).dot(np.transpose(u_matrix)).dot(y)
    return w_ridreg
    
def deg_freedom_lambda(y,x,lambda_var):
    u_matrix,s_matrix,v_matrix=np.linalg.svd(x,full_matrices=False)
    s_lambda=np.fromiter((item/(lambda_var+item*item) for item in s_matrix),float)
    deg_freedom=s_lambda.sum()
    return deg_freedom
    
def w_df(y,x,l):
    lam=np.linspace(0,l,l+1,endpoint=True)
    w_RR=np.asarray([w_RidReg(y,x,lambda_i) for lambda_i in lam])
    df_list=np.asarray([deg_freedom_lambda(y,x,lambda_i) for lambda_i in lam])
    w_rr_plot=plt.plot(df_list,w_RR)
    plt.legend(iter(w_rr_plot),['dimension '+str(cnt) for cnt in range(0,w_RR.shape[1])],bbox_to_anchor=(1, 1))
    plt.show()
    
def value(x,w):
    return x.dot(w)
    
def RMSE(y_prediction,y_true):
    y_error = y_true - y_prediction
    def error(x):
        return x*x
    y_error_square=np.apply_along_axis(error,axis=0,arr=y_error)
    final_RMSE=np.sqrt(y_error_square.sum()/y_error_square.shape[0])
    return final_RMSE
    
def x_polynomial(x,order):
    x_factor,x_intercept = np.hsplit(x,[x.shape[1]-1])
    fac=x_factor
    for i in range(1,order):
        fac=np.hstack((fac,np.power(x_factor,i+1)))
    fac=np.hstack((fac,x_intercept))
    return fac
    
def rmse_lp(y_train,x_train,y_test,x_test,lambda_range,p_order):
    x_new1=x_polynomial(x_train,p_order)
    x_new2=x_polynomial(x_test,p_order)
    lambda_list=np.linspace(0,lambda_range,lambda_range+1,endpoint=True)
    w_RR=np.asarray([w_RidReg(y_train,x_new1,lambda_i) for lambda_i in lambda_list])
    y_predict=np.asarray([value(x_new2,w_i) for w_i in w_RR])
    rmse=[RMSE(y_predict_i,y_test) for y_predict_i in y_predict]
    return rmse
    
def RMSE_lambda(y_train,x_train,y_test,x_test,lambda_range,p_order):
    lambda_list=np.linspace(0,lambda_range,lambda_range+1,endpoint=True)
    rmse_list=rmse_lp(y_train,x_train,y_test,x_test,lambda_range,p_order)
    plt.plot(lambda_list,rmse_list,label='order={},lambda between [0,{}]'.format(p_order,lambda_range))
    plt.show()
    
def answer_question_a():
    x_Train,y_Train,x_Test,y_Test=import_data('Y_train.csv','X_train.csv','Y_test.csv','X_test.csv')
    w_df(y_Train,x_Train,5000)
    
def answer_question_c():
    x_Train,y_Train,x_Test,y_Test=import_data('Y_train.csv','X_train.csv','Y_test.csv','X_test.csv')
    RMSE_lambda(y_Train,x_Train,y_Test,x_Test,50,1)
    
def answer_question_d():
    x_Train,y_Train,x_Test,y_Test=import_data('Y_train.csv','X_train.csv','Y_test.csv','X_test.csv')
    rmse_1=rmse_lp(y_Train,x_Train,y_Test,x_Test,500,1)
    rmse_2=rmse_lp(y_Train,x_Train,y_Test,x_Test,500,2)
    rmse_3=rmse_lp(y_Train,x_Train,y_Test,x_Test,500,3)
    lambda_list_500=np.linspace(0,500,501,endpoint=True)
    plt.plot(lambda_list_500,rmse_1,label='Order 1')
    plt.plot(lambda_list_500,rmse_2,label='Order 2')
    plt.plot(lambda_list_500,rmse_3,label='Order 3')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    
answer_question_a()

answer_question_c()

answer_question_d()
