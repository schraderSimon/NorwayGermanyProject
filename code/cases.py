import numpy as np
import matplotlib.pyplot as plt
from SamplerSystem import *
import seaborn as sns

class case0:

    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0):
        self.CO2_fossil_germany=710 #kg per MWh
        self.CO2_solar_germany=45
        self.CO2_wind_germany=11
        self.CO2_wind_norway=11
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
        wind_sun=VARSampler(3,1,self.coefs["windsun_coefs"],self.coefs["windsun_sigma"],seed=seed)
        timeseries_wind_wind_sun=wind_sun.sample_series(self.num_steps)
        self.simulation_results[0]=np.exp(timeseries_wind_wind_sun[0]+self.functions[0](self.time))
        self.simulation_results[1]=np.exp(timeseries_wind_wind_sun[1]+self.functions[1](self.time))
        self.simulation_results[5]=np.exp(timeseries_wind_wind_sun[2]+self.functions[5](self.time))
        loadings=VARSampler(2,3,coefs["load_coefs"],coefs["load_sigma"],seed=seed+1)
        timeseries_loads,loadingstrandom=loadings.sample_series(self.num_steps,returnRandom=True)
        self.simulation_results[2]=np.exp(timeseries_loads[0]+functions[2](self.time))
        self.simulation_results[3]=np.exp(timeseries_loads[1]+functions[3](self.time))
        water_data=water_sampler(loadingstrandom[:,0],coefs["water_sigma"],seed=seed+2)
        self.simulation_results[4]=np.exp(water_data+functions[4](self.time))

    def simulate_n_years(self,n=1):
        self.setUpValuesOfInterest(n)
        for i in range(n):
            self.simulation_step=i
            self.simulate_one_year(self.seed)
            self.seed+=4
            self.sample()
    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
        self.wind_surplus=np.zeros(n)
        self.wind_toNorway=np.zeros(n)
    def plotlast(self):
        try:
            self.simulation_results[0]
        except:
            self.simulate_one_year(0)
        for i in range(6):
            plt.plot(self.simulation_results[i],label=self.order[i],color=self.colors[i])
        #plt.plot(self.simulation_results[1]+self.simulation_results[5],label="Total German clean")
        if (2020-start_year>=0):
            plt.axvline(52*(2020-start_year),linestyle="--",color="grey",label="future line")


        #plt.axvline(52*num_years,linestyle="--",color="grey",label="future line")
        plt.title("System simulated from %d to %d"%(self.start_year+2017,self.start_year+2017+self.num_years))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlabel("week")
        plt.ylabel("MWh")
        plt.savefig("../graphs/testing_predictions_enhanced.pdf")
        plt.show()
    def sample(self):
        wind_surplus=0
        wind_toNorway=0
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany
        german_overproduction=-burns_germany.clip(max=0)
        wind_surplus=np.sum(german_overproduction)
        wind_toNorway=np.sum(german_overproduction.clip(max=self.sendable_max*24*7))
        burns_germany=burns_germany.clip(min=0)
        CO2_germany=15000*self.CO2_rest_germany+np.sum(burns_germany)*self.CO2_fossil_germany \
        +np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway
        self.CO2[self.simulation_step]=(CO2_germany)/self.num_years
        self.norwegian_balance[self.simulation_step]=-np.sum(self.simulation_results[2]-self.simulation_results[0]-self.simulation_results[4])/self.num_years
        self.wind_surplus[self.simulation_step]=wind_surplus
        self.wind_toNorway[self.simulation_step]=wind_toNorway
    def get_CO2(self):
        return self.CO2

class case1(case0):
    """Idea: We cannot _store_ water, but we can keep German water for 'one week'"""
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
        self.delay_DEtoNO=delay_DEtoNO
        self.delay_NOtoDE=delay_NOtoDE

    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.import_export_balance=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway
        german_overproduction_nextstep   =-(self.simulation_results[3][-1]-(self.simulation_results[1][-1]+self.simulation_results[5][-1])) #Cheating a tiny little bit by making year circulars
        norwegian_overproduction_nextstep=-(self.simulation_results[2][-1]-(self.simulation_results[0][-1]+self.simulation_results[4][-1]))
        for i in range(self.num_years*52):
            german_overproduction=german_overproduction_nextstep
            german_overproduction_nextstep=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]+toGermany[i])) #load - wind - sun
            norwegian_overproduction=norwegian_overproduction_nextstep
            norwegian_overproduction_nextstep=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]+toNorway[i])) #load - wind - water
            norwegian_overproduction_arrivestep=norwegian_overproduction
            german_overproduction_arrivestep=german_overproduction
            if self.delay_DEtoNO==1:
                norwegian_overproduction_arrivestep=norwegian_overproduction_nextstep
            if self.delay_NOtoDE==1:
                german_overproduction_arrivestep=german_overproduction_nextstep

            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                if(german_overproduction<self.sendable_max*24*7): #If the cable is capable of sending everything
                    toNorway[i-1+self.delay_DEtoNO]+=german_overproduction*(1-self.cable_loss) #Send German extra to Norway
                    balance+=german_overproduction
                else:
                    toNorway[i-1+self.delay_DEtoNO]+=self.sendable_max*(1-self.cable_loss)*24*7
                    balance+=self.sendable_max*24*7
            """Check if Norway can send to Germany"""
            if norwegian_overproduction>0 and german_overproduction_arrivestep<=0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                #Use as much as possible to make German energy green
                if(norwegian_overproduction<self.sendable_max*24*7): #If the cable is capable of sending (if its cable-ble, hehe)
                    toGermany[i-1+self.delay_NOtoDE]+=norwegian_overproduction*(1-self.cable_loss) #Send Norwegian extra to Germany
                    balance-=norwegian_overproduction
                else:
                    toGermany[i-1+self.delay_NOtoDE]+=self.sendable_max*(1-self.cable_loss)*24*7
                    balance-=self.sendable_max*24*7
        """Optional - 'load' remaining water into next day - but I guess it's more reasonable to "let it go to waste" otherwise whats the purpose of the time series """
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        self.CO2[self.simulation_step]=(CO2_germany)/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
    '''
    def sample_old(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway
        german_overproduction_nextstep   =-(self.simulation_results[3][-1]-(self.simulation_results[1][-1]+self.simulation_results[5][-1])) #Cheating a tiny little bit by making year circulars
        norwegian_overproduction_nextstep=-(self.simulation_results[2][-1]-(self.simulation_results[0][-1]+self.simulation_results[4][-1]))
        for i in range(self.num_years*52):

            german_overproduction=german_overproduction_nextstep
            german_overproduction_nextstep=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i])) #load - wind - sun
            norwegian_overproduction=norwegian_overproduction_nextstep
            norwegian_overproduction_nextstep=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i])) #load - wind - water
            norwegian_overproduction_arrivestep=norwegian_overproduction
            german_overproduction_arrivestep=german_overproduction
            if self.delay_DEtoNO==1:
                norwegian_overproduction_arrivestep=norwegian_overproduction_nextstep
            if self.delay_NOtoDE==1:
                german_overproduction_arrivestep=german_overproduction_nextstep
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                if(german_overproduction<self.sendable_max*24*7): #If the cable is capable of sending
                    self.simulation_results[2][i-1+self.delay_DEtoNO]-=german_overproduction*(1-self.cable_loss) #Send German extra to Norway
                    self.simulation_results[1][i-1]-=german_overproduction #remove what is send from german wind
                    balance+=german_overproduction
                else:
                    self.simulation_results[2][i-1+self.delay_DEtoNO]-=self.sendable_max*(1-self.cable_loss)*24*7
                    self.simulation_results[1][i-1]-=self.sendable_max*24*7
                    balance+=self.sendable_max*24*7
            if norwegian_overproduction>0 and german_overproduction_arrivestep<=0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                if(norwegian_overproduction<self.sendable_max*24*7): #If the cable is capable of sending (if its cable-ble, hehe)
                    self.simulation_results[3][i-1+self.delay_NOtoDE]-=norwegian_overproduction*(1-self.cable_loss) #Send Norwegian extra to Germany
                    balance-=norwegian_overproduction
                    self.simulation_results[4][i-1]-=norwegian_overproduction
                else:
                    self.simulation_results[3][i-1+self.delay_NOtoDE]-=self.sendable_max*(1-self.cable_loss)*24*7
                    balance-=self.sendable_max*24*7
                    self.simulation_results[4][i-1]-=self.sendable_max*24*7
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany
        burns_germany=burns_germany.clip(min=0)
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway)/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
        self.norwegian_balance[self.simulation_step]=-np.sum(self.simulation_results[2]-self.simulation_results[0]-self.simulation_results[4])/self.num_years
    '''
