import numpy as np

class VARSampler:
    def __init__(self,size,degree,coefficients,covariance_matrix,X_initial=0,seed=0):
        self.deg=degree
        self.size=size
        self.coefficients=coefficients
        self.covariance_matrix=covariance_matrix
        self.rng=np.random.default_rng(seed)
        if X_initial == 0:
            self.X=np.zeros((size,degree))
        else:
            self.X=X_initial
    def propagate(self,random_variable=0):
        if isinstance(random_variable,(int,float)) and random_variable==0: #If nothing is given, create new random data
            random_variable=self.rng.multivariate_normal(np.zeros(self.size),self.covariance_matrix)
        X_predict=np.zeros(self.size)
        for i in range(self.deg):
            X_predict+=self.coefficients[i]@self.X[:,i]
        self.X=np.roll(self.X,1,axis=1)
        self.X[:,0]=X_predict+random_variable
        return self.X[0]
    def sample_series(self,t=52,returnRandom=False):
        returnX=np.zeros((self.size,t))
        random_variables=self.rng.multivariate_normal(np.zeros(self.size),self.covariance_matrix,size=t)
        for i in range(t):
            self.propagate(random_variables[i])
            returnX[:,i]=self.X[:,0]
        if returnRandom:
            return returnX, random_variables
        else:
            return returnX
def water_sampler(load_randoms,covariance_matrix,reg_coef=0.919520,seed=123):
    rng=np.random.default_rng(seed)

    sigmawater=np.sqrt(covariance_matrix[1,1])
    sigmaload=np.sqrt(covariance_matrix[0,0])
    p=covariance_matrix[1,0]/(sigmawater*sigmaload)
    own_randoms=np.zeros(len(load_randoms))
    Xvals=np.zeros(len(load_randoms))
    for i in range(len(load_randoms)):
        own_randoms[i]=rng.normal(sigmawater/sigmaload*p*load_randoms[i],(1-p**2)*sigmawater**2)
    num_months=int(len(load_randoms)/4)
    Xvals=own_randoms
    for i in range(1,num_months):
        mean=np.mean(Xvals[(i-1)*4:i*4])
        meanarr=np.array([mean]*4)
        Xvals[i*4:(i+1)*4]+=reg_coef*meanarr
    return Xvals
