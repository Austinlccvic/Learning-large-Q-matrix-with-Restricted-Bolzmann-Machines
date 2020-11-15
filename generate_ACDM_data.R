library(MASS)

#Generate Q matrix with K is the number of latent attributes
QGenerator=function(K){ 
  Q1=diag(1,K) #Identity
  Q2=diag(1,K)
  Q2[row(Q2) - col(Q2) == -1] = 1 #upper bi-diagonal
  Q3=diag(1,K)
  Q3[abs(row(Q3)-col(Q3))==1]=1 #tri-diagonal
  Q=rbind(Q1, Q2, Q3)
  return(Q)
}

#Generate alpha matrix, dimensiton N by K
alphaGenerator=function(N, K, rho){#N individuals, K attributes, correlation is rho
  sigma=matrix(rho, K, K)
  sigma=sigma-diag(rep(rho, K))+ diag(K)
  z=mvrnorm(n = N, mu=rep(0, K), Sigma=sigma)
  threshold=seq(-0.5, 0.5, 1/(K-1)) #threshold on acquring each latent attributes differ
  m_threshold=rep(1,N)%*%t(threshold)
  alp=matrix(NA, N, K)
  alp[z<m_threshold]=0
  alp[z>=m_threshold]=1
  return(alp)
}

#Generate the success probability matrix, dimension N by J
#theta_{ij} represent the success probability of individual i answers item j correctly
thetaGenerator=function(Q, alpha, g, s){
  J=dim(Q)[1]
  K=dim(Q)[2]
  n=dim(alpha)[1]
  guessing=rep(g, J)
  for (i in 1:J){
    Q_c=Q[i, ]
    Q[i,]=(1-g-s)*Q[i,]/sum(Q_c)
  }
  theta=alpha%*%t(Q)+rep(1, n)%*%t(guessing)
  return(theta)
}


#Generate ACDM model data
ACDM_data = function(Q, theta){
  Nitem=dim(Q)[1]
  n=dim(theta)[1]
  data=matrix(rep(0, n*Nitem), nrow = n, ncol = Nitem)
  for (i in 1:n){
    for (j in 1:Nitem){
      u=runif(1)
      if (u<theta[i,j]){data[i,j]=1}
      else{data[i,j]=0}
    }
  }
  return(data)
}



#Executions generate and save ACDM data.
g=0.1
s=0.1
N=20000
K=5
Q=QGenerator(K)

rho=0.75
alpha = alphaGenerator(N, K, rho)
theta=thetaGenerator(Q, alpha,g,s)
dat=ACDM_data(Q,theta)
write.table(dat, sep=",",  col.names=FALSE,row.names=FALSE,file = 'ACDM0.1_K5_2000rho75.csv')
write.table(alpha, sep=",",  col.names=FALSE,row.names=FALSE,file = 'ACDM0.1_K5_2000_alprho75.csv')

