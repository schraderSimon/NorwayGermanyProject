import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from SamplerSystem import *

def coefs_to_function(trend_coef,season_coef,period=52):
    trendfunc=np.poly1d(trend_coef)
    seasonfunc=np.poly1d(season_coef)
    def seasonfunc_periodic(t):
        return seasonfunc((t-1)%period)
    return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))

class case0:

    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0):
        self.CO2_fossil_germany=710 #kg per MWh
        self.CO2_solar_germany=45
        self.CO2_wind_germany=11
        self.CO2_wind_noway=11
        self.CO2_water_norway=18.5
        self.CO2_rest_germany=(12.6*12+9.4*43+3.8*18.5)/(12.6+9.4+3.8)
        self.CO2_platforms=480 #kg per MWh
        self.sendable_max=1400 # Cable's load in MW
        self.cable_loss=0.05
        self.coefs=coefs
        self.seed=seed
        self.order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
        self.periods=[52,52,52,52,13,52]
        self.colors=["cyan","black","green","red","blue","orange"]
        self.num_steps=num_years*52
        self.num_years=num_years
        self.start_year=start_year-2017
        self.time=np.linspace(self.start_year*52,(self.start_year+num_years)*52,self.num_steps)
        functions=[]
        for i in range(len(self.order)):
            trend=trend_coefs[self.order[i]]
            season=season_coefs[self.order[i]]
            functions.append(self.coefs_to_function(trend,season,period=self.periods[i]))
        self.functions=functions
        self.CO2=[] #Co2 per simulation
    def coefs_to_function(self,trend_coef,season_coef,period):
        trendfunc=np.poly1d(trend_coef)
        seasonfunc=np.poly1d(season_coef)
        def seasonfunc_periodic(t):
            return seasonfunc((t-1)%period)
        return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))
    def simulate_one_year(self,seed):
        self.simulation_results=np.zeros((6,self.num_steps))
        wind_sun=VARSampler(3,1,self.coefs["windsun_coefs"],self.coefs["windsun_sigma"],seed=seed+1)
        timeseries_wind_wind_sun=wind_sun.sample_series(self.num_steps)
        self.simulation_results[0]=np.exp(timeseries_wind_wind_sun[0]+self.functions[0](self.time))
        self.simulation_results[1]=np.exp(timeseries_wind_wind_sun[1]+self.functions[1](self.time))
        self.simulation_results[5]=np.exp(timeseries_wind_wind_sun[2]+self.functions[5](self.time))
        loadings=VARSampler(2,3,coefs["load_coefs"],coefs["load_sigma"],seed=seed+2)
        timeseries_loads,loadingstrandom=loadings.sample_series(self.num_steps,returnRandom=True)
        self.simulation_results[2]=np.exp(timeseries_loads[0]+functions[2](self.time))
        self.simulation_results[3]=np.exp(timeseries_loads[1]+functions[3](self.time))
        water_data=water_sampler(loadingstrandom[:,0],coefs["water_sigma"],seed=seed+3)
        self.simulation_results[4]=np.exp(water_data+functions[4](self.time))

    def simulate_n_years(self,n=1):
        self.CO2=np.zeros(n)
        for i in range(n):
            self.simulation_step=i
            self.simulate_one_year(self.seed)
            self.seed+=1
            self.sample_CO2()
    def plotlast(self):
        try:
            self.simulation_results[0]
        except:
            self.simulate_one_year(0)
        for i in [0,2,3,4]:
            plt.plot(self.simulation_results[i],label=self.order[i],color=self.colors[i])
        plt.plot(self.simulation_results[1]+self.simulation_results[5],label="Total German clean")


        #plt.axvline(52*num_years,linestyle="--",color="grey",label="future line")
        plt.title("System simulated from %d to %d"%(self.start_year+2017,self.start_year+2017+self.num_years))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlabel("week")
        plt.ylabel("MW")
        #plt.savefig("../graphs/testing_predictions_enhanced.pdf")
        plt.show()
    def sample_CO2(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany
        burns_germany=burns_germany.clip(min=0)
        CO2_germany=15000*self.CO2_rest_germany+np.sum(burns_germany)*self.CO2_fossil_germany \
        +np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[3])*self.CO2_solar_germany
        self.CO2[self.simulation_step]=CO2_germany/self.num_years
    def get_CO2(self):
        return self.CO2

class case1(case0):
    """Idea: We cannot _store_ water, but we can keep German water for 'one week'"""
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
    def sample_CO2(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7
        german_overproduction_nextstep=-(self.simulation_results[3][i-1]-(self.simulation_results[1][i-1]+self.simulation_results[5][i-1]))
        norwegian_overproduction_nextstep=-(self.simulation_results[2][i-1]-self.simulation_results[0][i-1]-self.simulation_results[4][i-1])
        for i in range(self.num_years*52):

            german_overproduction=german_overproduction_nextstep
            german_overproduction_nextstep=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            norwegian_overproduction=norwegian_overproduction_nextstep
            norwegian_overproduction_nextstep=-(self.simulation_results[2][i]-self.simulation_results[2][i]-self.simulation_results[4][i])
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                print("Germany overproduced week %d"%i)
                if(german_overproduction<self.sendable_max*24*7): #If the cable is capable of sending
                    self.simulation_results[2][i]-=german_overproduction*(1-self.cable_loss) #Send German extra to Norway
                else:
                    self.simulation_results[2][i]-=self.sendable_max*(1-self.cable_loss)*24*7
            if norwegian_overproduction>0 and german_overproduction<0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                print("Norway overproduced week %d"%i)
                if(norwegian_overproduction<self.sendable_max*24*7): #If the cable is capable of sending
                    self.simulation_results[3][i-1]-=norwegian_overproduction*(1-self.cable_loss) #Send Norwegian extra to Germany
                else:
                    self.simulation_results[3][i-1]-=self.sendable_max*(1-self.cable_loss)*24*7
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany
        burns_germany=burns_germany.clip(min=0)
        CO2_germany=15000*self.CO2_rest_germany+np.sum(burns_germany)*self.CO2_fossil_germany \
        +np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[3])*self.CO2_solar_germany
        self.CO2[self.simulation_step]=CO2_germany/self.num_years
order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
periods=[52,52,52,52,13,52]
colors=["cyan","black","green","red","blue","orange"]
coefs=scipy.io.loadmat("../data/timeseries.mat")
trend_coefs=pd.read_csv("../data/trends.csv")
#trend_coefs["water NO"][0]=trend_coefs["load NO"][0]*4 #make water raise as high as production
season_coefs=pd.read_csv("../data/season.csv")

functions=[]
for i in range(len(order)):
    trend=trend_coefs[order[i]]
    season=season_coefs[order[i]]
    functions.append(coefs_to_function(trend,season,period=periods[i]))
try:
    start_year=int(sys.argv[1])
    num_years=int(sys.argv[2])
except:
    start_year=2021
    num_years=1
case1_simulator=case1(coefs,trend_coefs,season_coefs,seed=1,start_year=start_year,num_years=num_years)
case1_simulator.simulate_n_years(n=1000)
case1_simulator.plotlast()
case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=1,start_year=start_year,num_years=num_years)
case0_simulator.simulate_n_years(n=1000)
case0_simulator.plotlast()

CO2_hist_case0=case0_simulator.get_CO2()/1e9
CO2_hist_case1=case1_simulator.get_CO2()/1e9

import seaborn as sns
sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Case 0")
sns.kdeplot(CO2_hist_case1,x=r"Million tons CO$_2$",label="Case 1")
plt.legend()
plt.show()
