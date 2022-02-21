import numpy as np 
import multiprocessing as mp
from itertools import product
from fast_train import RBM
from training_reconstruction import RBM_train_recon
import functools
from scipy.sparse import diags

def QGenerator(K):#Generate Q matrix for simulation. K is the number of latent variables, we set no of items to be 3K
    Q1=np.diag([1]*K)
    Q2=diags([1, 1], [0,1], shape=(K, K)).toarray()
    Q3=diags([1, 1, 1], [-1, 0, 1], shape=(K, K)).toarray()
    Q=np.vstack((Q1, Q2, Q3))
    return Q

def alphaGenerator(N, K, rho):#Generate latent attributes matrix for simulation. N is the number of people
    sigma=np.full((K,K), rho)
    sigma=sigma-rho*np.identity(K)+np.identity(K)
    z=np.random.multivariate_normal(mean=[0]*K, cov=sigma, size=N)
    threshold=np.linspace(-0.5,0.5, K)
    thresMatrix=np.outer(np.ones(N),threshold)
    alp=np.zeros((N,K))
    alp[z>thresMatrix]=1
    return alp

def thetaGenerator(Q, alpha, g, s):#Q matrix, alpha matrix, guessing parameter, slipping parameter
    J, K = np.shape(Q)
    N=np.shape(alpha)[0]
    theta=np.zeros((N, J))
    for i in range(N):
        a=alpha[i,:]
        for j in range(J):
            if np.product(np.power(a, Q[j,:]))==1:
                theta[i,j]=1-s
            else:
                theta[i,j]=g
    return theta

def dinaGenerator(N, K, J, rho, g, s, seed):#generate DINA model data
    np.random.seed(seed)
    Q=QGenerator(K)
    alpha=alphaGenerator(N, K, rho)
    theta=thetaGenerator(Q, alpha, g, s)
    data=np.zeros((N,J))
    for i in range(N):
        for j in range(J):
            u=np.random.uniform(0, 1, 1)
            if u < theta[i,j]:
                data[i,j]=1
    return [data, alpha]               
            
                
def costMatrix(Q, W): #Compute cost matrix. W is an estimate for Q, both numpy matrix
    r,c =Q.shape
    wrongEntry=np.array([[0]*c for i in range(c)])
    for i in range(c):
        for j in range(c):
            wrongEntry[i][j]=sum((Q[:,i]-W[:,j])**2)
    return wrongEntry


def costComputation(costmatrix): #Compute out of all proportion estimation errors using Hungarian algorithm 
    K=np.shape(costmatrix)[0]
    from scipy import optimize
    costInd=optimize.linear_sum_assignment(costmatrix)
    row_ind, col_ind=costInd
    summ=0
    for i in range(len(row_ind)):
        summ+=costmatrix[int(row_ind[i]), int(col_ind[i])]
    percentageError=round(summ/(K*(3*K)),4)
    return percentageError

    
def propEstErr(costmatrix, N): #Compute out of all proportion estimation errors using Hungarian algorithm 
   # K=np.shape(costmatrix)[0]
    from scipy import optimize
    costInd=optimize.linear_sum_assignment(costmatrix)
    row_ind, col_ind=costInd
    ratio_error=[]
    for i in range(len(row_ind)):
        prop=costmatrix[int(row_ind[i]), int(col_ind[i])]/N
        ratio_error.append(prop)
    return ratio_error


#Returns two dimensional np arrays 
#First dimension represents the column index in Q
#Second dimension represents the corresponding column index in W
def costComputationInd(costmatrix): #Returns two dimensional np arrays of optimal matches of column indices of Q and W
   # K=np.shape(costmatrix)[0]
    from scipy import optimize
    costInd=optimize.linear_sum_assignment(costmatrix)
    row_ind, col_ind=costInd
    #summ=0
    ind=np.array([row_ind, col_ind])
    return ind

def costPositive(ind, Q, W):#Compute out of true positive errors
    nr, nc= np.shape(Q)
    cost=0
    for i in range(nc):
        x,y=ind[:,i]
        for j in range(nr):
            if Q[j,x]==1:
                if W[j,y]!=1:
                    cost+=1
    positiveTotal=6*nc-3
    return cost/positiveTotal

def costNegative(ind, Q, W):#Compute out of true negative errors
    nr, nc= np.shape(Q)
    cost=0
    for i in range(nc):
        x,y=ind[:,i]
        for j in range(nr):
            if Q[j,x]==0:
                if W[j,y]!=0:
                    cost+=1
    totalNegative=nc*3*nc-(6*nc-3)
    return cost/totalNegative

def costFDR(ind, Q, W):#Compute FDR
    nr, nc= np.shape(Q)
    cost=0
    count=0
    for i in range(nc):
        x,y=ind[:,i]
        for j in range(nr):
            if W[j,y]==1:
                count+=1
                if Q[j,x]!=1:
                    cost+=1
    return cost/count

#Overall error rate for a list of Q matrix (k specifying a list containing the numbers of latent attributes)
#w is a list of estimated W matrix
def errors(k,w):
    trueQ=QGenerator(k)
    estimatedQ=w
    costmatrix=costMatrix(trueQ,estimatedQ)
    e=costComputation(costmatrix)
    return e

#Similar as above, return a list of out of true positive errors
def errorPositive(k,w):
    trueQ=QGenerator(k)
    estimatedQ=w
    costmatrix=costMatrix(trueQ,estimatedQ)
    ind=costComputationInd(costmatrix)
    e=costPositive(ind, trueQ, estimatedQ)
    return e

#Similar as above, return a list of out of true negative errors
def errorNegative(k,w):
    trueQ=QGenerator(k)
    estimatedQ=w
    costmatrix=costMatrix(trueQ,estimatedQ)
    ind=costComputationInd(costmatrix)
    e=costNegative(ind, trueQ, estimatedQ)
    return e

#Similar as above, return a list of FDRs
def errorFDR(k,w):
    trueQ=QGenerator(k)
    estimatedQ=w
    costmatrix=costMatrix(trueQ,estimatedQ)
    ind=costComputationInd(costmatrix)
    e=costFDR(ind, trueQ, estimatedQ)
    return e
def compute_hidden_est_err(alpha, est):
    N,K=np.shape(alpha)
    ratio_error=[]
    for k in range(K):
        prop=np.sum(np.abs(alpha[:,k]-est[:,k]))
        ratio_error.append(prop)
    return np.array([ratio_error])

def sigmoid(x): #sigmoid function
    return 1 / (1 + np.exp(-x))

def visible_to_hidden(v, W, h_bias): #sampling hidden units and probability
        p_h = np.matmul(v, W)+h_bias #Compute p(h|v)
        p_h=sigmoid(p_h)
        return p_h

def evaluate_ACC(weight, N, data, hbias, alp,  nu): #Return classification accuracy of latent attributes
    res=[]
    for j in range(nu):   
        dat=data[j*N:(j+1)*N]
        al=alp[j*N:(j+1)*N]
        estimated_alpha=visible_to_hidden(v=dat, W=weight, h_bias=hbias)
        estimated_alpha[estimated_alpha>=0.5]=1
        estimated_alpha[estimated_alpha<0.5]=0
        cost=costMatrix(al, estimated_alpha)
        r=propEstErr(costmatrix=cost, N=N)
        res.append(r)
    res=np.array(res)
    return res.mean(0)
    




#Return cross-validation error and the corresponding trained Q given a combination of lambda and gamma
#We seek to do parallel computing on this function across different combination of hyper-parameters
def tu(lambd, gamma, visible_units, hidden_units, training, test, ga=[0.25,0.5,1,1.5], batch_size=50, epochs=100):
    rbm_dina = RBM(visible_units, hidden_units, lambd, batch_size, gamma) 
    cost=rbm_dina.train(training, epochs)
    err=float('Inf')
    for gamm in ga:     
        trainRecon=RBM_train_recon(visible_units, hidden_units, batch_size, cost[4], cost[3], eta=gamm)
        t_c=trainRecon.train(training, epochs)
        if t_c[2]<err:
            weight=t_c[0]
            h_bias=t_c[1]
            err=t_c[2]
    return [err, cost[3], weight, h_bias]


def exe(data, K, N, num,alpha): #parallel computing multiple CPUs, num--integer--num fold cross validation
    pool = mp.Pool(10) #use 10 cores for parallel computing
    lambd=[0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016] #penalties
    gamma=[0.5, 1.5, 2.5, 3.5, 4.5] #learning rate
    n_lam=len(lambd)
    n_gam=len(gamma)
    overal_e=[] #overall error
    positive_e=[] #positve errors
    FDR_e=[] #FDR
    negative_e=[] #negative errors
    ACC=np.array([[0]*K])
    #hidden_e=np.array([[0]*K])
    for i in range(100):#possibly iterate through different data set combined to run replicas
        dat=data[i*N:(i+1)*N,:]
        alp=alpha[i*N:(i+1)*N,:]
        err=float('Inf')
        for j in range(num):#num-fold cross validation
            np.random.seed(seed=10*i+j)
            n1=np.shape(dat)[0]
            n_test=n1/num
            te=dat[int(j*n_test):int((j+1)*n_test),:]
            train=np.delete(dat,range(int(j*n_test),int((j+1)*n_test)), axis=0)
            tune=functools.partial(tu,visible_units=3*K, hidden_units=K, training=train, test=te)##
            if __name__ == '__main__': 
                re=pool.starmap(tune, product(lambd,gamma))  #parallel computing
                for m in range(n_gam*n_lam):
                    if re[m][0]< err:
                        err=re[m][0]
                        w_opt=re[m][1]
                        weight=re[m][2]
                        h_bias=re[m][3]
        overal_e.append(errors(K,w_opt)) #compute overall error of this optimal trained Q
        positive_e.append(errorPositive(K,w_opt)) #compute out of positive error of this optimal trained Q
        FDR_e.append(errorFDR(K,w_opt))#compute FDR of this optimal trained Q
        negative_e.append(errorNegative(K,w_opt))#compute out of negative error of this optimal trained Q
        acc=np.array([evaluate_ACC(weight=weight, N=N, data=dat, hbias=h_bias, alp=alp,  nu=1)])
        ACC=np.concatenate((ACC, acc))
    pool.close()#close parallel computing
    ACC=np.delete(ACC, obj=0, axis=0)
   # hidden_e=np.delete(hidden_e, (0), axis=0)                
    res=np.array([overal_e,positive_e,FDR_e,negative_e]), ACC, w_opt #save results in an np array
    return res

#N=2000, rho=0.25, K=25
N=2000*10 #2000 observations with 10 replica
K=25 #25 latent attributes
J=3*K #75 items
rho=0.25 #correlations between latent attributes
g=0.1 #guessing parameter
s=0.1 #slipping parameter
dat, alp=dinaGenerator(N, K, J, rho, g, s, seed=2005)
result2000_25_1=exe(data=dat, K=25, N=2000, num=5, alpha=alp) 
np.savetxt('ACC_DINA0.1_K25_N2000_rho25.csv', result2000_25_1[1], delimiter=',') 
np.savetxt('Q_DINA0.1_K25_N2000_rho25.csv', result2000_25_1[2], delimiter=',')
np.savetxt('Res_DINA0.1_K25_N2000_rho25.csv', result2000_25_1[0], delimiter=',')














