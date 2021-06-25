import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import ar_select_order
import statsmodels.api as sm
from  statsmodels.tsa.seasonal import seasonal_decompose as se_de
from  statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima_process import arma_generate_sample
import warnings

warnings.filterwarnings("ignore")

def average_array(arr,length):
    mean=np.mean(arr.reshape(-1,length),axis=1)
    std=np.std(arr.reshape(-1,length),axis=1)
    return mean,std
def fill_nan(array):
    array_interpolated=pd.DataFrame(array).interpolate().values.ravel()
    return array_interpolated
def adf_test(timeseries,output=True):

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    if dfoutput["p-value"]>0.05:
        print("series may be nonstationary")
    if output:
        print ('Results of Dickey-Fuller Test:')
        for key,value in dftest[4].items():
           dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)
def kpss_test(timeseries,output=True):

    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    if(kpss_output["p-value"]<0.05):
        print("series is maybe not stationary")
    if output:
        print ('Results of KPSS Test:')
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)
def decompose(data,period):
    #DecomposeResult=STL(data,period=52).fit()#,extrapolate_trend="freq")
    DecomposeResult=se_de(data,period=52)#,extrapolate_trend="freq")
    trend=DecomposeResult.trend[~np.isnan(DecomposeResult.trend)]
    seasonal=DecomposeResult.seasonal[~np.isnan(DecomposeResult.seasonal)]
    resid=DecomposeResult.resid[~np.isnan(DecomposeResult.resid)]#+DecomposeResult.trend[~np.isnan(DecomposeResult.trend)]
    return trend,seasonal,resid
data=pd.read_csv("../data/time_series_60min_singleindex_2020.csv")
print(data.tail())
time=data["cet_cest_timestamp"]
begin_year=2017; begin_month=6;begin_day=1
end_year=2020;end_month=6;end_day=1
num_years=end_year-begin_year
begin_date=datetime.datetime(begin_year,begin_month,begin_day)
end_date=datetime.datetime(end_year,end_month,end_day)
onyear_index_start=np.where(data=="%sT00:00:00+0200"%(begin_date.strftime("%Y-%m-%d")))[0][0]
onyear_index_end=np.where(data=="%sT00:00:00+0200"%(end_date.strftime("%Y-%m-%d")))[0][0]-num_years*24 #drop the last day of the year for simplicity reasons I guess?
if (begin_year<=2016):
    if (begin_year==2016 and begin_month>2):
        pass
    else:
        onyear_index_end-=24
if (end_year==2020 and end_month>2):
    onyear_index_end-=24
print("Index at: ",onyear_index_start)
time=time[onyear_index_start:onyear_index_end]

wind_DE=data["DE_wind_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() # Innstead of "DE_wind_capacity" which does not matter so much
load_DE=data["DE_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()#-12000
solar_DE=data["DE_solar_generation_actual"][onyear_index_start:onyear_index_end].to_numpy()
load_NO=data["NO_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()
wind_NO=data["NO_wind_onshore_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() #Assuming there is no offshore wind?
wind_NO*=5.525/5.194 #Additional factor so it matches the official 2019 data
steplength=1
print("Wind Norway total:%f MWh"%(np.sum(wind_NO)*steplength))
print("Wind Germany total:%f MWh"%(np.sum(wind_DE)*steplength))
print("Solar Germany total:%f MWh"%(np.sum(solar_DE)*steplength))
print("Load Germany total:%f MWh"%(np.sum(load_DE)*steplength))
print("Load Norway total:%f MWh"%(np.sum(load_NO)*steplength))

water_data=pd.read_csv("../data/vann_norge_ny.csv",delimiter=";")
water_data=water_data.set_index("maaned")
oil_data=pd.read_csv("../data/olje.csv",delimiter=";")
oil_data=oil_data.set_index("maaned")
if begin_month<10:
    begin_string="%dM0%d"%(begin_year,begin_month)
else:
    begin_string="%dM%d"%(begin_year,begin_month)
if end_month<10:
    end_string="%dM0%d"%(end_year,end_month)
else:
    end_string="%dM%d"%(end_year,end_month)
water_data=water_data[begin_string:end_string]
oil_data=oil_data[begin_string:end_string]

water_data.drop(water_data.tail(1).index,inplace=True)
oil_data.drop(oil_data.tail(1).index,inplace=True)

print(water_data)
print(np.sum(water_data["Elektrisk kraft"].to_numpy()))
print(np.sum(oil_data["Elektrisk kraft"].to_numpy()))
def weekify(water_data,num_years):
    water=water_data["Elektrisk kraft"].to_numpy()
    water_peryear=np.split(water,num_years)
    num_days=[31,28,31,30,31,30,31,31,30,31,30,31]
    num_days=np.roll(num_days,-(begin_month-1))
    print(num_days)
    water_perweek=[]
    for year in range(num_years):
        num_days_copy=np.copy(num_days)
        counter=0
        for week in range(52):
            if num_days_copy[counter]>=7:
                water_perweek.append(water_peryear[year][counter]/(365*24/12))
                num_days_copy[counter]-=7
            else:
                days_left=num_days_copy[counter]
                appenderino=0
                appenderino+=water_peryear[year][counter]/(365*24/12)/7*days_left
                num_days_copy[counter]-=days_left
                counter+=1
                appenderino+=water_peryear[year][counter]/(365*24/12)/7*(7-days_left)
                num_days_copy[counter]-=(7-days_left)
                water_perweek.append(appenderino)
    return water_perweek
"""
data=pd.read_csv("../data/time_series_60min_singleindex_old.csv")
time=data["cet_cest_timestamp"]
begin_year=2012; begin_month=6;begin_day=1
end_year=2016;end_month=6;end_day=1
num_years=end_year-begin_year
begin_date=datetime.datetime(begin_year,begin_month,begin_day)
end_date=datetime.datetime(end_year,end_month,end_day)
onyear_index_start=np.where(data=="%sT00:00:00+0200"%(begin_date.strftime("%Y-%m-%d")))[0][0]
onyear_index_end=np.where(data=="%sT00:00:00+0200"%(end_date.strftime("%Y-%m-%d")))[0][0]-num_years*24 #drop the last day of the year for simplicity reasons I guess?
if (begin_year<=2012 and begin_month<=2):
    onyear_index_end-=24
if (end_year==2016 and end_month>2):
    onyear_index_end-=24
print("Index at: ",onyear_index_start)
time=time[onyear_index_start:onyear_index_end]

wind_DE=data["DE_wind_generation"][onyear_index_start:onyear_index_end].to_numpy() # Innstead of "DE_wind_capacity" which does not matter so much
load_DE=data["DE_load_old"][onyear_index_start:onyear_index_end].to_numpy()#-12000
solar_DE=data["DE_solar_generation"][onyear_index_start:onyear_index_end].to_numpy()
load_NO=data["NO_load_old"][onyear_index_start:onyear_index_end].to_numpy()
wind_NO=data["NO_wind_onshore_generation"][onyear_index_start:onyear_index_end].to_numpy() #Assuming there is no offshore wind?
"""

#time=np.linspace(1,52,52,dtype="int")
steplength=168
try:
    steplength=int(sys.argv[1]) #day, multiply by 7 to get week
except:
    pass
num_timesteps=int(num_years*52*24*7/steplength)
time=np.linspace(1,num_timesteps,num_timesteps,dtype="int")

wind_DE,wind_DE_std=average_array(wind_DE,steplength)
solar_DE,solar_DE_std=average_array(solar_DE,steplength)
wind_NO,wind_NO_std=average_array(wind_NO,steplength)
load_DE,load_DE_std=average_array(load_DE,steplength)
load_NO,load_NO_std=average_array(load_NO,steplength)
load_NO=fill_nan(load_NO)
wind_NO=fill_nan(wind_NO)
wind_DE=fill_nan(wind_DE)
load_DE=fill_nan(load_DE)-15000
solar_DE=fill_nan(solar_DE)
water_NO=weekify(water_data,num_years)
water_NO+=0.02*np.mean(water_NO) #add thermal

logarithm=True
polydeg=1
if logarithm:
    #wind_DE,wind_DE_lambda=boxcox(wind_DE)
    #wind_NO,wind_NO_lambda=boxcox(wind_NO)
    #solar_DE,solar_DE_lambda=boxcox(solar_DE)
    #load_DE,load_DE_lambda=boxcox(load_DE)
    #load_NO,load_NO_lambda=boxcox(load_NO)
    #water_NO,water_NO_lambda=boxcox(water_NO)
    wind_DE,solar_DE,wind_NO,load_NO,load_DE,water_NO=np.log(wind_DE),np.log(solar_DE),np.log(wind_NO),np.log(load_NO),np.log(load_DE),np.log(water_NO)

print("Wind Norway total:%f MWh"%(np.sum(wind_NO)*steplength))
print("Wind Germany total:%f MWh"%(np.sum(wind_DE)*steplength))
print("Water Norway total:%f MWh"%(np.sum(water_NO)*steplength))
print("Solar Germany total:%f MWh"%(np.sum(solar_DE)*steplength))

print("Load Germany total:%f MWh"%(np.sum(load_DE)*steplength))
print("Load Norway total:%f MWh"%(np.sum(load_NO)*steplength))

def fit_seasonal(data_withouttrend,period=52,degree=4):
    data=data_withouttrend
    X = [i%period for i in range(0, len(data))]
    coef = np.polyfit(X, data, degree)
    function=np.poly1d(coef)
    def function_period(t):
        return function(t%period)
    return function_period
def remove_trend(data,time,period=52,degree_trend=1,degree_season=4):
    trend_function=np.poly1d(np.polyfit(time, data, degree_trend))
    season_function=fit_seasonal(data-trend_function(time))
    def total_trend(t):
        return trend_function(t)+season_function(t)
    residue=data-total_trend(time)
    return residue, total_trend
period=int(24*7*52/steplength)
print(len(water_NO),len(wind_DE))
wind_DE_residue,wind_DE_function=remove_trend(wind_DE,time,period=period)
wind_NO_residue,wind_NO_function=remove_trend(wind_NO,time,period=period)
load_DE_residue,load_DE_function=remove_trend(load_DE,time,period=period)
load_NO_residue,load_NO_function=remove_trend(load_NO,time,degree_season=6,period=period)
solar_DE_residue,solar_DE_function=remove_trend(solar_DE,time,degree_season=8,period=period)
water_NO_residue,water_NO_function=remove_trend(water_NO,time,degree_season=4,period=period)

plt.plot(time,wind_NO,label="wind NO",color="cyan")
plt.plot(time,wind_NO_function(time),"--",color="cyan")
plt.plot(time,wind_DE,label="wind DE",color="black")
plt.plot(time,wind_DE_function(time),"--",color="black")
plt.plot(time,load_NO,label="load NO",color="green")
plt.plot(time,load_NO_function(time),"--",color="green")
plt.plot(time,load_DE,label="load DE",color="red")
plt.plot(time,load_DE_function(time),"--",color="red")
plt.plot(time,solar_DE,label="solar DE",color="orange")
plt.plot(time,solar_DE_function(time),"--",color="orange")
plt.plot(time,water_NO,label="water NO",color="blue")
plt.plot(time,water_NO_function(time),"--",color="blue")

plt.xlabel("week")
if logarithm:
    plt.ylabel("log(MW)")
else:
    plt.ylabel("MW")
plt.legend()
plt.savefig("../graphs/time_series_electricity_data.pdf")
plt.show()


fig, axs = plt.subplots(2, 2,figsize=(10,10))
axs[0, 0].plot(wind_NO_residue,label="NO")
axs[0, 0].set_title('Wind')
axs[0, 0].plot(wind_DE_residue,label="DE")
axs[0, 0].legend()
axs[0, 1].plot(load_NO_residue,label="NO")
axs[0, 1].plot(load_DE_residue,label="DE")
axs[0, 1].set_title('Load')
axs[0, 1].legend()

axs[1, 0].plot(water_NO_residue,label="water NO")
axs[1, 0].plot(solar_DE_residue,label="solar DE")
axs[1, 0].set_title('Water/solar')
axs[1, 0].legend()
plt.tight_layout()
plt.xlabel("week")
plt.ylabel("Residue")
plt.savefig("../graphs/residue_timeseries_log.pdf")
plt.show()



wind_DE_fit=np.poly1d(np.polyfit(time, wind_DE, 1))
solar_DE_fit=np.poly1d(np.polyfit(time, solar_DE, 1))
load_DE_fit=np.poly1d(np.polyfit(time, load_DE, 1))
wind_NO_fit=np.poly1d(np.polyfit(time, wind_NO, 1))
load_NO_fit=np.poly1d(np.polyfit(time, load_NO, 1))
Wind_NO_trend,Wind_NO_seasonal,Wind_NO_resid=decompose(wind_NO-wind_NO_fit(time),period=52)
Wind_DE_trend,Wind_DE_seasonal,Wind_DE_resid=decompose(wind_DE-wind_DE_fit(time),period=52)
load_NO_trend,load_NO_seasonal,load_NO_resid=decompose(load_NO-load_NO_fit(time),period=52)
load_DE_trend,load_DE_seasonal,load_DE_resid=decompose(load_DE-load_DE_fit(time),period=52)
water_NO_trend,water_NO_seasonal,water_NO_resid=decompose(water_NO-water_NO_fit(time),period=52)
solar_DE_trend,solar_DE_seasonal,solar_DE_resid=decompose(solar_DE-solar_DE_fit(time),period=52)

def plot_data(type):
    if type=="trend":
        plt.plot(Wind_NO_trend,label="Wind NO")
        plt.plot(Wind_DE_trend,label="Wind DE")
        plt.plot(load_NO_trend,label="load NO")
        plt.plot(load_DE_trend,label="load DE")
        plt.plot(solar_DE_trend,label="solar DE")
        plt.plot(water_NO_trend,label="water NO")
    elif type=="seasonal":
        plt.plot(Wind_NO_seasonal,label="Wind NO")
        plt.plot(Wind_DE_seasonal,label="Wind DE")
        plt.plot(load_NO_seasonal,label="load NO")
        plt.plot(load_DE_seasonal,label="load DE")
        plt.plot(solar_DE_seasonal,label="solar DE")
        plt.plot(water_NO_seasonal,label="water NO")
    elif type=="resid":
        plt.plot(Wind_NO_resid,label="Wind NO")
        plt.plot(Wind_DE_resid,label="Wind DE")
        #plt.plot(load_NO_resid,label="load NO")
        #plt.plot(load_DE_resid,label="load DE")
        plt.plot(solar_DE_resid,label="solar DE")
        plt.plot(water_NO_resid,label="water NO")
    plt.legend()
    plt.show()
plot_data("resid")

adf_test(Wind_NO_resid)
adf_test(Wind_DE_resid)#
adf_test(load_NO_resid)#-load_NO_fit(time))
adf_test(load_DE_resid)#-load_DE_fit(time))
adf_test(water_NO_resid)#-water_NO_fit(time))
adf_test(solar_DE_resid)#-solar_DE_fit(time))
kpss_test(Wind_NO_resid)
kpss_test(Wind_DE_resid)#
kpss_test(load_NO_resid)#-load_NO_fit(time))
kpss_test(load_DE_resid)#-load_DE_fit(time))
kpss_test(water_NO_resid)#-water_NO_fit(time))
kpss_test(solar_DE_resid)#-solar_DE_fit(time))

#solar_DE=np.exp(solar_DE)
actual=wind_DE-wind_DE_fit(time)
arma_mod20=ARIMA(actual,order=(3,0,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod20.plot_predict()
plt.legend()
plt.show()
from statsmodels.tsa.ar_model import ar_select_order

mod=ar_select_order(actual,maxlag=13)
print(mod.ar_lags)

import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(actual, lags=40)
sm.graphics.tsa.plot_acf(actual, lags=40)
plt.show()
