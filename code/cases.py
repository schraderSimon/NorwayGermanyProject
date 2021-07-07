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
        self.CO2_platforms_good=480 #kg per MWh
        self.CO2_platforms_bad=333
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
        loadings=VARSampler(2,3,self.coefs["load_coefs"],self.coefs["load_sigma"],seed=seed+1)
        timeseries_loads,loadingstrandom=loadings.sample_series(self.num_steps,returnRandom=True)
        self.simulation_results[2]=np.exp(timeseries_loads[0]+self.functions[2](self.time))
        self.simulation_results[3]=np.exp(timeseries_loads[1]+self.functions[3](self.time))
        water_data=water_sampler(loadingstrandom[:,0],self.coefs["water_sigma"],seed=seed+2)
        self.simulation_results[4]=np.exp(water_data+self.functions[4](self.time))

    def simulate_n_years(self,n=1):
        self.setUpValuesOfInterest(n)
        for i in range(n): #For each year
            self.simulation_step=i #Advance the simulation step by one
            self.simulate_one_year(self.seed) #Simulate for one year
            self.seed+=4 #Increase seed by four [To get new values]
            self.sample() #update samples for that year
    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.profiles=np.zeros((n,6))
        self.norwegian_balance=np.zeros(n)
        self.norwegian_balance_nowind=np.zeros(n) #If wind is ignored
        self.CO2_nowind=np.zeros(n) #If wind is ignored
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
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        CO2_norway_nowind=np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway)/self.num_years
        self.CO2_nowind[self.simulation_step]=(CO2_germany+CO2_norway_nowind)/self.num_years
        self.norwegian_balance[self.simulation_step]=-np.sum(self.simulation_results[2]-self.simulation_results[0]-self.simulation_results[4])/self.num_years
        self.norwegian_balance_nowind[self.simulation_step]=-np.sum(self.simulation_results[2]-self.simulation_results[4])/self.num_years
        self.wind_surplus[self.simulation_step]=wind_surplus
        self.wind_toNorway[self.simulation_step]=wind_toNorway
        for i in range(6):
            self.profiles[self.simulation_step,i]=np.sum(self.simulation_results[i])
    def get_CO2(self):
        return self.CO2

class case1(case0):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
        self.delay_DEtoNO=delay_DEtoNO
        self.delay_NOtoDE=delay_NOtoDE

    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.import_export_balance=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
        self.num_days_DEtoNO=np.zeros(n)
        self.num_days_NOtoDE=np.zeros(n)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        num_days_DEtoNO=0
        num_days_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9

        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]))
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                num_days_DEtoNO+=1
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            if norwegian_overproduction>0 and german_overproduction<=0:
                num_days_NOtoDE+=1
                send_amount=np.min([-german_overproduction,norwegian_overproduction,self.sendable_max*24*7])
                toGermany[i]+=send_amount*(1-self.cable_loss) #Send Norwegian extra to Germany
                balance-=send_amount
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+toNorway/(1-self.cable_loss)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway)/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
        self.num_days_DEtoNO[self.simulation_step]=num_days_DEtoNO
        self.num_days_NOtoDE[self.simulation_step]=num_days_NOtoDE
class case1_delay1(case0):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=1,delay_DEtoNO=1):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
        self.delay_DEtoNO=delay_DEtoNO
        self.delay_NOtoDE=delay_NOtoDE

    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.import_export_balance=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
        self.num_days_DEtoNO=np.zeros(n)
        self.num_days_NOtoDE=np.zeros(n)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        num_days_DEtoNO=0
        num_days_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0])+self.delay_DEtoNO)
        toGermany=np.zeros(len(self.simulation_results[0])+self.delay_NOtoDE)
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9

        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]+toNorway[i])) #ignore toNorway, consider it as "independent"-ish of the elecricity production
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]+toGermany[i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                num_days_DEtoNO+=1
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i+1]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE

            if norwegian_overproduction>0 and german_overproduction<=0:
                #Norway overproduces less than what Germany underproduces that day
                num_days_NOtoDE+=1
                send_amount=np.min([norwegian_overproduction,self.sendable_max*24*7])
                toGermany[i+1]+=send_amount*(1-self.cable_loss) #Send Norwegian extra to Germany
                balance-=send_amount
        toNorway=np.delete(toNorway,0)
        toNorway=np.roll(toNorway,1)
        toGermany=np.delete(toGermany,0)
        toGermany=np.roll(toGermany,1)
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+toNorway/(1-self.cable_loss)-toGermany #Rolling is necessary because
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway)/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
        self.num_days_DEtoNO[self.simulation_step]=num_days_DEtoNO
        self.num_days_NOtoDE[self.simulation_step]=num_days_NOtoDE

class case2(case0):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0,mean_wind=0.431):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
        self.delay_DEtoNO=delay_DEtoNO
        self.delay_NOtoDE=delay_NOtoDE
        self.mean_wind=mean_wind*1e6*(1-self.cable_loss) #Twh
    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.import_export_balance=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        total_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        printout=False
        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]))
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            """Check if Norway can send to Germany"""
            if norwegian_overproduction>0 and german_overproduction<0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                if(total_NOtoDE>=self.mean_wind):
                    if printout:
                        print("Done sending after week %d, simulation %d"%(i,self.simulation_step))
                        printout=False
                    continue
                left_sendable_energy=self.mean_wind-total_NOtoDE
                send_amount=np.min([norwegian_overproduction,self.sendable_max*24*7,left_sendable_energy,-german_overproduction])
                toGermany[i]+=send_amount*(1-self.cable_loss)
                balance-=send_amount
                total_NOtoDE+=send_amount
        green_germany=self.simulation_results[1]+self.simulation_results[5]

        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)/(1-self.cable_loss)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway)/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
class case3_1(case0):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0,mean_wind=0.431):
        case0.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed)
        self.delay_DEtoNO=delay_DEtoNO
        self.delay_NOtoDE=delay_NOtoDE
        self.platform_restriction=15/52*1e6 #MWh
    def setUpValuesOfInterest(self,n):
        self.CO2=np.zeros(n)
        self.CO2_bad=np.zeros(n)
        self.import_export_balance=np.zeros(n)
        self.norwegian_balance=np.zeros(n)
    def sample(self):
        for i in range(len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        total_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toPlatforms=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        printout=False
        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]))
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            """Check if Norway can send to Germany"""
            """Difference to the previous scenarios: Here, what is send from Germany can go DIRECTLY to the platforms, as the cable is only used one-way"""
            norwegian_overproduction_withgermany=norwegian_overproduction+toNorway[i]
            if norwegian_overproduction_withgermany>0 and norwegian_overproduction>=0: #If Norway has extra energy, it is send to the platforms
                #Use as much as possible to make the platforms green
                sendable_max=np.min([norwegian_overproduction_withgermany,self.platform_restriction])
                toPlatforms[i]+=sendable_max #Send Norwegian extra to Germany
            elif norwegian_overproduction_withgermany>0 and norwegian_overproduction<0: #Parts of what comes from Germany is used for the platforms, the other part is used
                surplus=norwegian_overproduction_withgermany
                toPlatforms[i]+=surplus
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany
        TWhtoplatform=np.sum(toPlatforms)
        CO2_saved_platform_good=TWhtoplatform*self.CO2_platforms_good
        CO2_saved_platform_bad=TWhtoplatform*self.CO2_platforms_bad
        self.CO2_bad[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_bad)/self.num_years
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_good)/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toPlatforms-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
class case3_2(case3_1):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0,mean_wind=0.431,reduction=0.3333333333):
        case3_1.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed,delay_NOtoDE,delay_DEtoNO,mean_wind)
        self.reduction=reduction #Reduce how much is sent to Germany when Germany is in need of extra :)
        self.mean_wind=mean_wind*1e6*(1-self.cable_loss) #Twh
    def sample(self):
        reduction=self.reduction
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        total_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toPlatforms=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        printout=True
        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[0][i]+self.simulation_results[4][i]))
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            """Check if Norway can send to Germany"""
            if norwegian_overproduction>0 and german_overproduction<0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                left_sendable_energy=self.mean_wind-total_NOtoDE
                if(total_NOtoDE>=self.mean_wind):
                    if printout:
                        print("Done sending after week %d, simulation %d, mean wind %f"%(i,self.simulation_step,self.mean_wind))
                        printout=False
                else:
                    send_amount=np.min([norwegian_overproduction,self.sendable_max*24*7*reduction,left_sendable_energy,-german_overproduction])
                    toGermany[i]+=send_amount*(1-self.cable_loss)
                    balance-=send_amount
                    total_NOtoDE+=send_amount
            """Difference to the previous scenarios: Here, what is send from Germany can go DIRECTLY to the platforms, as the cable is only used one-way"""
            norwegian_overproduction-=toGermany[i]
            norwegian_overproduction_withgermany=norwegian_overproduction+toNorway[i]
            if norwegian_overproduction_withgermany>0 and norwegian_overproduction>=0: #If Norway has extra energy, it is send to the platforms
                #Use as much as possible to make the platforms green
                if(norwegian_overproduction_withgermany<self.platform_restriction): #If the cable is capable of sending (if its cable-ble, hehe)
                    toPlatforms[i]+=norwegian_overproduction_withgermany #Send Norwegian extra to Germany
                else:
                    toPlatforms[i]+=self.platform_restriction
            elif norwegian_overproduction_withgermany>0 and norwegian_overproduction<0: #Parts of what comes from Germany is used for the platforms, the other part is used
                surplus=norwegian_overproduction_withgermany
                toPlatforms[i]+=surplus
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)/(1-self.cable_loss)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany

        TWhtoplatform=np.sum(toPlatforms)
        CO2_saved_platform_good=TWhtoplatform*self.CO2_platforms_good
        CO2_saved_platform_bad=TWhtoplatform*self.CO2_platforms_bad
        self.CO2_bad[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_bad)/self.num_years
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_good)/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-toPlatforms-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years
class case3_3(case3_1):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0,mean_wind=0.431):
        case3_1.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed,delay_NOtoDE,delay_DEtoNO,mean_wind)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        total_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toPlatforms=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[0])*self.CO2_wind_norway+np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        printout=False
        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[4][i])-self.simulation_results[0][i])
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            """Check if Norway can send to Germany"""
            if norwegian_overproduction>0 and german_overproduction<0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                send_amount=np.min([norwegian_overproduction,self.sendable_max*24*7,-german_overproduction])
                toGermany[i]+=send_amount*(1-self.cable_loss) #Send Norwegian extra to Germany
                balance-=send_amount
                total_NOtoDE+=send_amount
            """Difference to the previous scenarios: Here, what is send from Germany can go DIRECTLY to the platforms, as the cable is only used one-way"""
            norwegian_overproduction-=toGermany[i]
            norwegian_overproduction_withgermany=norwegian_overproduction+toNorway[i]
            if norwegian_overproduction_withgermany>0 and norwegian_overproduction>=0: #If Norway has extra energy, it is send to the platforms
                #Use as much as possible to make the platforms green
                send_amount=np.min([norwegian_overproduction_withgermany,self.platform_restriction])
                toPlatforms[i]+=send_amount*(1-self.cable_loss)
            elif norwegian_overproduction_withgermany>0 and norwegian_overproduction<0: #Parts of what comes from Germany is used for the platforms, the other part is used
                surplus=norwegian_overproduction_withgermany
                toPlatforms[i]+=surplus
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)/(1-self.cable_loss)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany

        TWhtoplatform=np.sum(toPlatforms)
        CO2_saved_platform_good=TWhtoplatform*self.CO2_platforms_good
        CO2_saved_platform_bad=TWhtoplatform*self.CO2_platforms_bad
        self.CO2_bad[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_bad)/self.num_years
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_good)/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-toPlatforms-self.simulation_results[2]+self.simulation_results[0]+self.simulation_results[4])/self.num_years

class case4(case3_1):
    def __init__(self,coefs,trend_coefs,season_coefs,num_years=1,start_year=2020,seed=0,delay_NOtoDE=0,delay_DEtoNO=0,mean_wind=0.431):
        case3_1.__init__(self,coefs,trend_coefs,season_coefs,num_years,start_year,seed,delay_NOtoDE,delay_DEtoNO,mean_wind)
    def sample(self):
        for i in range (len(self.simulation_results)):
            self.simulation_results[i]*=24*7 #Convert from MW to MWh
        balance=0
        total_NOtoDE=0
        toNorway=np.zeros(len(self.simulation_results[0]))
        toPlatforms=np.zeros(len(self.simulation_results[0]))
        toGermany=np.zeros(len(self.simulation_results[0]))
        """Calculate emissions from the actual productions, before considering what is send where, assuming sending is emission-free"""
        CO2_germany=np.sum(self.simulation_results[1])*self.CO2_wind_germany+np.sum(self.simulation_results[5])*self.CO2_solar_germany+15000*self.CO2_rest_germany
        CO2_norway=np.sum(self.simulation_results[4])*self.CO2_water_norway+13*1e9
        printout=False
        for i in range(self.num_years*52):
            norwegian_overproduction=-(self.simulation_results[2][i]-(self.simulation_results[4][i]))
            german_overproduction=-(self.simulation_results[3][i]-(self.simulation_results[1][i]+self.simulation_results[5][i]))
            """Check if Germany can send to Norway"""
            if german_overproduction>0: #If Germany produces too much wind, it is reduced from Norway's next week consumption
                #Use as much as possible to increase Norwegian water the next step, alternatively, to decrease Norwegian load.
                send_amount_DE=np.min([german_overproduction,self.sendable_max*24*7])
                toNorway[i]+=send_amount_DE*(1-self.cable_loss) #Send German extra to Norway
                balance+=send_amount_DE
            """Check if Norway can send to Germany"""
            if norwegian_overproduction>0 and german_overproduction<0: #If Norway has extra energy, it is reduced from Germanys this-week energy
                send_amount=np.min([norwegian_overproduction,self.sendable_max*24*7,-german_overproduction])
                toGermany[i]+=send_amount*(1-self.cable_loss) #Send Norwegian extra to Germany
                balance-=send_amount
                total_NOtoDE+=send_amount
            """Difference to the previous scenarios: Here, what is send from Germany can go DIRECTLY to the platforms, as the cable is only used one-way"""
            norwegian_overproduction-=toGermany[i]
            norwegian_overproduction_withgermany=norwegian_overproduction+toNorway[i]
            if norwegian_overproduction_withgermany>0 and norwegian_overproduction>=0: #If Norway has extra energy, it is send to the platforms
                #Use as much as possible to make the platforms green
                send_amount=np.min([norwegian_overproduction_withgermany,self.platform_restriction])
                toPlatforms[i]+=send_amount*(1-self.cable_loss)
            elif norwegian_overproduction_withgermany>0 and norwegian_overproduction<0: #Parts of what comes from Germany is used for the platforms, the other part is used
                surplus=norwegian_overproduction_withgermany
                toPlatforms[i]+=surplus
        green_germany=self.simulation_results[1]+self.simulation_results[5]
        burns_germany=self.simulation_results[3]-green_germany+np.roll(toNorway,-self.delay_DEtoNO)/(1-self.cable_loss)-toGermany
        burns_germany=burns_germany.clip(min=0) #negative numbers are ignored
        CO2_germany+=np.sum(burns_germany)*self.CO2_fossil_germany

        TWhtoplatform=np.sum(toPlatforms)
        CO2_saved_platform_good=TWhtoplatform*self.CO2_platforms_good
        CO2_saved_platform_bad=TWhtoplatform*self.CO2_platforms_bad
        self.CO2_bad[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_bad)/self.num_years
        self.CO2[self.simulation_step]=(CO2_germany+CO2_norway-CO2_saved_platform_good)/self.num_years
        self.norwegian_balance[self.simulation_step]=np.sum(toNorway-toGermany/(1-self.cable_loss)-toPlatforms-self.simulation_results[2]+self.simulation_results[4])/self.num_years
        self.import_export_balance[self.simulation_step]=balance/self.num_years
