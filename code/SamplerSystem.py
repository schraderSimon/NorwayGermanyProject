import numpy as np
import sys
class VARSampler:
    def __init__(self,size,degree,coefficients,covariance_matrix,X_initial=0,seed=0):
        self.deg=degree
        self.size=size
        self.coefficients=coefficients
        self.covariance_matrix=covariance_matrix
        self.rng=np.random.default_rng(seed)
        if X_initial == 0:
            self.X=np.zeros((size,degree)) #Store the "degree" last variables as they are needed
        else:
            self.X=X_initial
    def propagate(self,random_variable=0):
        """Calculates the next time step"""
        if isinstance(random_variable,(int,float)) and random_variable==0: #If nothing is given, create new random data
            random_variable=self.rng.multivariate_normal(np.zeros(self.size),self.covariance_matrix) #Create self.size multivariate normal distributed data
        X_predict=np.zeros(self.size) #The next time step

        for i in range(self.deg):
            X_predict+=self.coefficients[i]@self.X[:,i] #Calculate the non-stochastic next step
        self.X=np.roll(self.X,1,axis=1) #Move all variables one to the right (such that X[:,0]->X[:,1])
        self.X[:,0]=X_predict+random_variable #Update the "newest" X
        return self.X[:,0] #Return the newest estimate
    def sample_series(self,t=52,returnRandom=False):
        """
        Returns
        """
        returnX=np.zeros((self.size,t)) #Empty array for t time steps of X data
        random_variables=self.rng.multivariate_normal(np.zeros(self.size),self.covariance_matrix,size=t) #Create t multivariate normal distributed variables (this is faster)
        for i in range(t):
            #Calculate next time step
            self.propagate(random_variables[i]) #
            if(i==5):
                #sys.exit(1)
                pass
            returnX[:,i]=self.X[:,0]
        #print((returnX.T))
        #sys.exit(1)
        if returnRandom: #If the random variables should be returned to
            return returnX, random_variables
        else:
            return returnX
def water_sampler(load_randoms,covariance_matrix,reg_coefs=[1.189324,-0.484997],seed=123,watertype="dependent"): #Sampling for water
    rng=np.random.default_rng(seed)
    if watertype=="dependent": #Wether the random variable from water depends on load
        sigmawater=np.sqrt(covariance_matrix[1,1]) #standard deviation for water
        sigmaload=np.sqrt(covariance_matrix[0,0]) #Standard deviation for load
        p=covariance_matrix[1,0]/(sigmawater*sigmaload) #rho in the covariance matrix
    own_randoms=np.zeros(len(load_randoms)) #Random values for the water
    Xvals=np.zeros(len(load_randoms)) #Return values
    sigma_alone=np.sqrt(0.001406/4) #sigma From the AR fit, divided by 4
    if watertype=="dependent":
        for i in range(len(load_randoms)):
            own_randoms[i]=rng.normal(sigmawater/sigmaload*p*load_randoms[i],np.sqrt((1-p**2))*sigmawater) #Conditional random value (this is from Wikipedia)
    elif watertype=="independent":
        for i in range(len(load_randoms)):
            own_randoms=rng.normal(0,sigma_alone,size=len(own_randoms))
    else:
        raise NameError("Water type does not exist, %s"%watertype)
    num_months=int(len(load_randoms)/4) #Number of months
    Xvals=np.copy(own_randoms) #Needs to be a copy, FUCK NUMPY
    mean=np.mean(Xvals[0:4])
    meanarr=np.array([mean]*4)
    Xvals[4:8]+=reg_coefs[0]*meanarr
    for i in range(2,num_months): #For each month
        mean_prev=np.mean(Xvals[(i-1)*4:i*4])
        mean_prevprev=np.mean(Xvals[(i-2)*4:(i-1)*4])
        meanarr_prev=np.array([mean_prev]*4)
        meanarr_prevprev=np.array([mean_prevprev]*4)
        Xvals[i*4:(i+1)*4]+=reg_coefs[0]*meanarr_prev+reg_coefs[1]*meanarr_prevprev
    return Xvals
