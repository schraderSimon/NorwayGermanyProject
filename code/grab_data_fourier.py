import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import ccf
from SamplerSystem import *
from helper_functions import *
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 12, 'legend.labelspacing':0.2})

def average_array(arr,length):
    """Takes an array and returns a new array of length len(arr)/length,
    where means are calculated along "length"
    """
    mean=np.mean(arr.reshape(-1,length),axis=1)
    std=np.std(arr.reshape(-1,length),axis=1)
    return mean,std
def fill_nan(array):
    """Interpolates an array"""
    array_interpolated=pd.DataFrame(array).interpolate().values.ravel()
    return array_interpolated
def adf_test(timeseries,output=False):
    """
    Augmented Dickey Fuller test. Set "output=False" if no detailed output is desired.
    Prints "series may not be stationary" if p>0.05
    """
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    if dfoutput["p-value"]>0.05:
        print("series ADF failed,p=%.2f"%dfoutput["p-value"])
    if output:
        print ('Results of Dickey-Fuller Test:')
        for key,value in dftest[4].items():
           dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)
def kpss_test(timeseries,output=False):
    """
    KPSS test. Set "output=False" if no detailed output is desired.
    Prints "series may not be stationary" if p<0.05
    """
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    if(kpss_output["p-value"]<0.05):
        print("series KPSS failed,p=%.4f"%kpss_output["p-value"])
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
def weekify(water_data,num_years):
    """Turns monthly data into week data"""
    water=water_data["Elektrisk kraft"].to_numpy()
    water_peryear=np.split(water,num_years)
    num_days=[31,28,31,30,31,30,31,31,30,31,30,31]
    num_days=np.roll(num_days,-(begin_month-1)) #Make the array begin at the right number of days
    water_perweek=[]
    for year in range(num_years):
        num_days_copy=np.copy(num_days)
        counter=0
        for fourweek in range(52):
            if num_days_copy[counter]>=7:

                water_perweek.append(water_peryear[year][counter]*7/num_days[counter])
                num_days_copy[counter]-=7
            else:
                days_left=num_days_copy[counter]
                appenderino=0
                appenderino+=water_peryear[year][counter]*(7/num_days[counter])*(days_left/7)
                num_days_copy[counter]-=days_left
                counter+=1
                appenderino+=water_peryear[year][counter]*(7/num_days[counter])*((7-days_left)/7)
                num_days_copy[counter]-=(7-days_left)
                water_perweek.append(appenderino)
    water_perweek=np.array(water_perweek)/(24*7)
    return water_perweek
def four_weekify(water_data,num_years):
    """Turns monthly data into 4-week data"""
    water=water_data["Elektrisk kraft"].to_numpy()
    water_peryear=np.split(water,num_years)
    num_days=[31,28,31,30,31,30,31,31,30,31,30,31]
    num_days=np.roll(num_days,-(begin_month-1)) #Make the array begin at the right number of days
    water_perfourweek=[]
    for year in range(num_years):
        num_days_copy=np.copy(num_days)
        counter=0
        for fourweek in range(13):
            if num_days_copy[counter]>=28:

                water_perfourweek.append(water_peryear[year][counter]*28/num_days[counter])
                num_days_copy[counter]-=28
            else:
                days_left=num_days_copy[counter]
                appenderino=0
                appenderino+=water_peryear[year][counter]*(28/num_days[counter])*(days_left/28)
                num_days_copy[counter]-=days_left
                counter+=1
                appenderino+=water_peryear[year][counter]*(28/num_days[counter])*((28-days_left)/28)
                num_days_copy[counter]-=(28-days_left)
                water_perfourweek.append(appenderino)
    water_perfourweek=np.array(water_perfourweek)/(24*28)
    return water_perfourweek
def constfunc(data):
    """Returns a function f(t) that simply returns the mean of input array 'data' """
    def f(t):
        return np.mean(data)
    return f
def fit_seasonal(data_withouttrend,period=52,degree=6):
    """
    Fits a polynomial best fit of degree "degree" over "period" datapoints.
    """
    data=data_withouttrend
    X = [i%period for i in range(0, len(data))]
    coef = np.polyfit(X, data, degree)
    function=np.poly1d(coef)
    def function_period(t):
        return function((t-1)%period) #-1 because my time starts at one.
    return function_period, coef
def remove_trend_fourier(data,time,period=52,degree_trend=1,degree_season=6,trend_function=0,num_cos=0):
    trend_coef=0
    if trend_function == 0:
        trend_coef=np.polyfit(time, data, degree_trend)
        trend_function=np.poly1d(trend_coef)
    num_sin=degree_season-num_cos
    popt, pcov = curve_fit(make_fourier(num_cos,num_sin,period/2), time, data-trend_function(time), [1.0] * (degree_season+1))
    #plt.plot(time,make_fourier(10,10,period/2)(time,*popt),label="fit")

    def total_trend(t):
        return trend_function(t)+make_fourier(num_cos,num_sin,period/2)(t,*popt)
    residual=data-total_trend(time)
    assert abs(np.mean(residual))<1e-5
    return residual, total_trend, trend_coef,popt
def remove_trend(data,time,period=52,degree_trend=1,degree_season=6,trend_function=0):
    """
    Detrends data.
    Input:
    data (array): The time series
    time (array): The corresponding time-values of the time series
    period (int): The periodicity of the seasonal trend
    degree_trend (int): The degree of the polynomial to fit the trend function
    degree_season (int): The degree of the polynomial to fit the season
    trend_function (function) (optional): The trend function if already determined

    Returns:
    array - the Residuals (the "error" of the fit)
    function (callable) - the function f(t) describing the trend
    trend_coef : The polynomial coefficients of the trend function
    season_coef: The polynomial coefficients of the seasonal function
    """
    trend_coef=0
    if trend_function == 0:
        trend_coef=np.polyfit(time, data, degree_trend)
        trend_function=np.poly1d(trend_coef)
    season_function,season_coef=fit_seasonal(data-trend_function(time),degree=degree_season,period=period)
    def total_trend(t):
        return trend_function(t)+season_function(t)
    residual=data-total_trend(time)
    assert abs(np.mean(residual))<1e-5
    return residual, total_trend, trend_coef,season_coef

"""Load data"""
data=pd.read_csv("../data/time_series_60min_singleindex_2020.csv")

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
time=time[onyear_index_start:onyear_index_end]
wind_DE=data["DE_wind_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() # Innstead of "DE_wind_capacity" which does not matter so much
load_DE=data["DE_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()#-12000
solar_DE=data["DE_solar_generation_actual"][onyear_index_start:onyear_index_end].to_numpy()
load_NO=data["NO_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()
wind_NO=data["NO_wind_onshore_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() #Assuming there is no offshore wind?

"""See that data is approximately correct"""
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

water_data["Elektrisk kraft"]+=np.mean(water_data["Elektrisk kraft"])*0.02

steplength=168 #a week.

num_timesteps=int(num_years*52*24*7/steplength) #Number of time steps in total
time=np.linspace(1,num_timesteps,num_timesteps,dtype="int")

wind_DE,wind_DE_std=average_array(wind_DE,steplength)
solar_DE,solar_DE_std=average_array(solar_DE,steplength)
wind_NO,wind_NO_std=average_array(wind_NO,steplength)
load_DE,load_DE_std=average_array(load_DE,steplength)
load_NO,load_NO_std=average_array(load_NO,steplength)
load_NO=fill_nan(load_NO)
wind_NO=fill_nan(wind_NO)
wind_DE=fill_nan(wind_DE)
load_DE=fill_nan(load_DE)-15000 #Remove 15000 from the German data as part of the deterministic trend
solar_DE=fill_nan(solar_DE)
water_NO=weekify(water_data,num_years)
#water_NO+=0.02*np.mean(water_NO) #add thermal
time_month=np.linspace(1,int(13*num_years),int(13*num_years))
water_NO_4week=four_weekify(water_data,num_years)

print("Wind Norway total:%f MWh"%(np.sum(wind_NO)*steplength))
print("Wind Germany total:%f MWh"%(np.sum(wind_DE)*steplength))
print("Solar Germany total:%f MWh"%(np.sum(solar_DE)*steplength))
print("Load Germany total:%f MWh"%(np.sum(load_DE)*steplength))
print("Load Norway total:%f MWh"%(np.sum(load_NO)*steplength))
print("Water Norway total (weekified):%f MWh"%(np.sum(water_NO_4week)*steplength*4))
print("Water Norway total (total):%f Mwh"%(np.sum(water_data["Elektrisk kraft"].to_numpy())))

logarithm=True
polydeg=1
if logarithm: #take logarithm of the data
    wind_DE,solar_DE,wind_NO,load_NO,load_DE,water_NO,water_NO_4week=np.log(wind_DE),np.log(solar_DE),np.log(wind_NO),np.log(load_NO),np.log(load_DE),np.log(water_NO),np.log(water_NO_4week)



period=int(24*7*52/steplength)
trend_coefs=[0,0,0,0,0,0]
season_coefs=[0,0,0,0,0,0]
deg_fourier=6
num_cos=0
wind_DE_residue,wind_DE_function,trend_coefs[1],season_coefs[1]=remove_trend_fourier(wind_DE,time,period=period,degree_season=deg_fourier,num_cos=num_cos)
wind_NO_residue,wind_NO_function,trend_coefs[0],season_coefs[0]=remove_trend_fourier(wind_NO,time,period=period,degree_season=deg_fourier,num_cos=num_cos)
load_DE_residue,load_DE_function,trend_coefs[3],season_coefs[3]=remove_trend_fourier(load_DE,time,period=period,degree_season=deg_fourier,num_cos=num_cos)
load_NO_residue,load_NO_function,trend_coefs[2],season_coefs[2]=remove_trend_fourier(load_NO,time,period=period,degree_season=deg_fourier,num_cos=num_cos)
solar_DE_residue,solar_DE_function,trend_coefs[5],season_coefs[5]=remove_trend_fourier(solar_DE,time,period=period,degree_season=deg_fourier,num_cos=num_cos)
water_NO_residue,water_NO_function,water_NO_trend_coef,water_NO_season_coef=remove_trend_fourier(water_NO,time,period=period,degree_season=deg_fourier,num_cos=num_cos)


water_NO4_trend  = np.poly1d(np.polyfit(time_month, water_NO_4week,1))
trend_coefs[4]=[0,np.mean(water_NO_4week)]
water_NO4_season_coef, pcov = curve_fit(make_fourier(num_cos,deg_fourier-num_cos,13/2), time_month, water_NO_4week-water_NO4_trend(time_month), [1.0] * (deg_fourier+1))
season_coefs[4]=water_NO4_season_coef
water_NO4_season=lambda t:make_fourier(num_cos,deg_fourier-num_cos,13/2)(t,*water_NO4_season_coef)
water_NO4_residue= water_NO_4week-water_NO4_season(time_month)-water_NO4_trend(time_month)
water_NO4_function=lambda t: water_NO4_season(t/4)+constfunc(water_NO_4week)(t/4)
print(time)
for i in range(len(time)):
    print(time[i],water_NO4_function(i))


water_NO_trend  = np.poly1d(np.polyfit(time, water_NO,1))
water_NO_season,water_NO_season_coef = fit_seasonal(water_NO-water_NO_trend(time),period=52,degree=4)
water_NO_residue= water_NO-water_NO_season(time)-water_NO_trend(time)
stigning=np.polyfit(time,load_NO,1)[0]#couple wind production to  energy production
stigning=0  #make wind production constant in time
water_NO_function=lambda t: water_NO_season(t)+np.poly1d([stigning,np.mean(water_NO)])(t)


colors=["cyan","black","green","red","blue","orange"]
def plot_timeseries():
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
    plt.plot(time_month*4,water_NO_4week,label="water NO",color="blue")
    plt.plot(time,water_NO4_function(time),"--",color="blue")

    plt.xlabel("week")
    if logarithm:
        plt.ylabel("log(MW)")
    else:
        plt.ylabel("MW")
    plt.legend(loc="upper left")
    plt.tight_layout()
    if logarithm:
        plt.savefig("../graphs/time_series_electricity_data_logfourier.pdf")
        plt.show()
        plt.plot(time,np.exp(wind_NO),label="wind NO",color="cyan")
        plt.plot(time,np.exp(wind_NO_function(time)),"--",color="cyan")
        plt.plot(time,np.exp(wind_DE),label="wind DE",color="black")
        plt.plot(time,np.exp(wind_DE_function(time)),"--",color="black")
        plt.plot(time,np.exp(load_NO),label="load NO",color="green")
        plt.plot(time,np.exp(load_NO_function(time)),"--",color="green")
        plt.plot(time,np.exp(load_DE),label="load DE",color="red")
        plt.plot(time,np.exp(load_DE_function(time)),"--",color="red")
        plt.plot(time,np.exp(solar_DE),label="solar DE",color="orange")
        plt.plot(time,np.exp(solar_DE_function(time)),"--",color="orange")
        plt.plot(time_month*4,np.exp(water_NO_4week),label="water NO",color="blue")
        plt.plot(time,np.exp(water_NO4_function(time)),"--",color="blue")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.ylabel("MW")
        plt.xlabel("week")
        plt.savefig("../graphs/time_series_electricity_datafourier.pdf")
    else:
        plt.savefig("../graphs/time_series_electricity_datafourier.pdf")
    plt.show()
plot_timeseries()

def plot_residues():
    fig, axs = plt.subplots(3, 1,figsize=(10,10))
    for i in range(3):
            axs[i].set_xlabel("week")
            axs[i].set_ylabel("residual")

    axs[0].plot(time,wind_NO_residue,label="NO")
    axs[0].set_title('Wind')
    axs[0].plot(time,wind_DE_residue,label="DE")
    axs[0].legend()

    axs[1].plot(time,load_NO_residue,label="NO")
    axs[1].plot(time,load_DE_residue,label="DE")
    axs[1].set_title('Load')
    axs[1].legend()

    axs[2].plot(time_month*4,water_NO4_residue,"o",label="water NO")
    axs[2].plot(time,solar_DE_residue,label="solar DE")
    axs[2].set_title('Water/solar')
    axs[2].legend()

    plt.tight_layout()
    if logarithm:
        plt.savefig("../graphs/residue_timeseries_logfourier.pdf")
    else:
        plt.savefig("../graphs/residue_timeseriesfourier.pdf")
    plt.show()
plot_residues()


print("Wind NO:")
adf_test(wind_NO_residue)
kpss_test(wind_NO_residue)
print("Wind DE:")
adf_test(wind_DE_residue)
kpss_test(wind_DE_residue)
print("Load NO")
adf_test(load_NO_residue)
kpss_test(load_NO_residue)
print("Load DE")
adf_test(load_DE_residue)
kpss_test(load_DE_residue)
"""
for i in range(2,20):
    for j in range(i+1):
        load_NO_residue,load_NO_function,trend_coefs[2],season_coefs[2]=remove_trend_fourier(load_NO,time,period=period,degree_season=i,num_cos=j)
        print("%d %d"%(i,j))
        adf_test(load_NO_residue)
        kpss_test(load_NO_residue)
sys.exit(1)
"""
print("Water NO")

adf_test(water_NO4_residue)
kpss_test(water_NO4_residue)
print("Solar DE")
adf_test(solar_DE_residue)
kpss_test(solar_DE_residue)

order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
original_series=[wind_NO,wind_DE,load_NO,load_DE,water_NO_4week,solar_DE]
deterministic_functions=[wind_NO_function,wind_DE_function,load_NO_function,load_DE_function,water_NO4_function,solar_DE_function]
times=[time,time,time,time,time_month*4,time]
residues=[wind_NO_residue,wind_DE_residue,load_NO_residue,load_DE_residue,water_NO4_residue,solar_DE_residue]
periods=[52]*4;periods.append(13); periods.append(52)

def test_series():
    """Test wether the decomposition is correct"""
    for i in [1,2,3,0,5]: #We do not expect water to be correct, as we directly modify the trend. The trend is expected to be linear.
        assert np.all(np.abs(original_series[i]-deterministic_functions[i](times[i])-residues[i])<1e-5)
test_series()

future_years=2
time_future=np.linspace(1,(num_years+future_years)*52,(num_years+future_years)*52)
time_future_month=np.linspace(1,(num_years+future_years)*13,(num_years+future_years)*13)
times_future=[time_future,time_future,time_future,time_future,time_future_month,time_future]
plot_times=times; plot_times[4]*=4
plot_times_future=times_future;plot_times_future[4]*=4




arma_models=[]
for i,residue in enumerate(residues):
    print(len(residue))
    maxlag=3
    try:
        arma_model_degree=ar_select_order(residue,maxlag=maxlag).ar_lags[-1]

    except IndexError:
        arma_model_degree=0
    print("%s is a %d process"%(order[i],arma_model_degree))
    arma_models.append(ARIMA(residue,order=(arma_model_degree,0,0)).fit())
    print(len(residue),len(arma_models[-1].resid))
print("Trend coefficients")
for i in range(len(order)):
    print("\\item %s: $T(t)=%ft+%f$ \\\\"%(order[i],trend_coefs[i][0],trend_coefs[i][1]))
print("Seasonal coefficients")
print("Time series &",end="")
for j in range(6,0,-1):
        print(" $a_%d$& "%j,end="")
print(" $a_0$ \\\\ \hline")

for i in range(len(order)):
    print("%s & "%order[i],end="")
    number_coefs=len(season_coefs[i])
    for j in range(0,number_coefs-1):
        exponent=int(np.log10(np.abs(season_coefs[i][j])))-1
        #print("$%.3E$ &"%season_coefs[i][j],end="")
        print("$%.2f\cdot10^{%d}$  &"%(season_coefs[i][j]/10**(exponent),exponent),end="")
    exponent=int(np.log10(np.abs(season_coefs[i][-1])))-1
    #print("$%.3E$ &"%season_coefs[i][j],end="")
    print("$%.2f\cdot10^{%d}$  \\\\ \\hline"%(season_coefs[i][-1]/10**(exponent),exponent))
    #print("$%.3E$ \\\\ \\hline"%season_coefs[i][-1])
print("stationray coeff")
for i,arma_model in enumerate(arma_models):
    #print(arma_model.summary())
    #print(arma_model.params)
    #print(arma_model.bse)
    sigma=arma_model.params[-1]
    sigmaerr=arma_model.bse[-1]
    phi1,phi2,phi3=0,0,0
    ephi1,ephi2,ephi3=0,0,0 #errors
    if len(arma_model.params)>2:
        phi1=arma_model.params[1]
        ephi1=arma_model.bse[1]
    if len(arma_model.params)>3:
        phi2=arma_model.params[2]
        ephi2=arma_model.bse[2]
    if len(arma_model.params)>4:
        phi3=arma_model.params[3]
        ephi3=arma_model.bse[3]
    print("%s & %f$\pm$%f & %f$\pm$%f & %f$\pm$%f & %f$\pm$ %f \\\\ \\hline"%(order[i],phi1,ephi1,phi2,ephi2,phi3,ephi3,sigma,sigmaerr))
new_data=[]
for i in range(len(arma_models)):
    deterministic_y=deterministic_functions[i](times_future[i])
    time_series_y=arma_models[i].simulate(len(times_future[i]))
    new_data.append(np.exp(deterministic_y+time_series_y))

def plot_example():
    """Plots an example simulation"""
    for i in range(len(arma_models)):
        if i != 4:
            plt.plot(plot_times_future[i],new_data[i],label=order[i],color=colors[i])
        else:
            plt.plot(plot_times_future[i],new_data[i],"o",markersize=2,label=order[i],color=colors[i])
    plt.axvline(52*num_years,linestyle="--",color="grey",label="future line")
    plt.title("Example of a system simulated 2017-2022")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.xlabel("week")
    plt.ylabel("MW")
    plt.tight_layout()
    plt.savefig("../graphs/testing_predictionsfourier.pdf")
    plt.show()
plot_example()
def plot_correlations():
    """Plots the correlations of the fitted data"""
    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(3, 2,figsize=(10,10))
    for i in range(3):
        for j in range(2):
            axs[i,j].set_xlabel("week")
            axs[i,j].set_ylabel("residual")
            #axs[i,j].legend()
    axs[2,1].set_xlabel("month (4 weeks)")

    axs[0,0].set_title("NO wind, DE wind")
    axs[0,0].xcorr(arma_models[0].resid,arma_models[1].resid,maxlags=3,color="orange")
    axs[0,0].axhline(+1.96/np.sqrt(len(wind_DE_residue)))
    axs[0,0].axhline(-1.96/np.sqrt(len(wind_DE_residue)))

    axs[0,1].set_title("NO load, DE load")
    axs[0,1].xcorr(arma_models[2].resid,arma_models[3].resid,maxlags=3,color="orange")
    axs[0,1].axhline(+1.96/np.sqrt(len(wind_DE_residue)))
    axs[0,1].axhline(-1.96/np.sqrt(len(wind_DE_residue)))

    axs[1,0].set_title("DE wind, DE sun")
    axs[1,0].xcorr(arma_models[1].resid,arma_models[5].resid,maxlags=3,color="orange")
    axs[1,0].axhline(+1.96/np.sqrt(len(wind_DE_residue)))
    axs[1,0].axhline(-1.96/np.sqrt(len(wind_DE_residue)))

    axs[1,1].set_title("NO wind, DE sun")
    axs[1,1].xcorr(arma_models[0].resid,arma_models[5].resid,maxlags=3,color="orange")
    axs[1,1].axhline(+1.96/np.sqrt(len(wind_DE_residue)))
    axs[1,1].axhline(-1.96/np.sqrt(len(wind_DE_residue)))

    axs[2,0].set_title("DE wind, DE load")
    axs[2,0].xcorr(arma_models[1].resid,arma_models[3].resid,maxlags=3,color="orange")
    axs[2,0].axhline(+1.96/np.sqrt(len(wind_DE_residue)))
    axs[2,0].axhline(-1.96/np.sqrt(len(wind_DE_residue)))

    axs[2,1].set_title("NO load, NO water")
    axs[2,1].xcorr(np.mean(arma_models[2].resid.reshape(-1,4),axis=1),arma_models[4].resid,maxlags=3,color="orange")
    axs[2,1].axhline(+1.96/np.sqrt(len(water_NO4_residue)))
    axs[2,1].axhline(-1.96/np.sqrt(len(water_NO4_residue)))
    plt.tight_layout()
    plt.savefig("../graphs/residual_correlationsfourier.pdf")
    plt.show()
    plt.rcParams.update({'font.size': 12})
plot_correlations()

wind_and_sun=np.array([wind_NO_residue,wind_DE_residue,solar_DE_residue]).T
load=np.array([load_NO_residue,load_DE_residue]).T
monthly_load_and_water=np.array(([np.mean(load_NO_residue.reshape(-1,4),axis=1),water_NO4_residue])).T
df_load = pd.DataFrame(load, columns = ['Load NO',"Load DE"],index=time)

df_sunwind=pd.DataFrame(wind_and_sun, columns = ["Wind NO",'Wind DE',"Sun DE"],index=time)
df_NOloadwater=pd.DataFrame(monthly_load_and_water, columns = ['Load NO',"Water NO"],index=time_month)
model_windsun = VAR(df_sunwind).fit(maxlags=1,ic="aic")
model_load = VAR(df_load).fit(maxlags=3,ic="aic")
model_NOloadwater = VAR(df_NOloadwater).fit(maxlags=2,ic="aic")
def plot_predict_loads():
    matrix_1=model_load.coefs[0]
    matrix_2=model_load.coefs[1]
    matrix_3=model_load.coefs[2]
    true_vals=load
    print(true_vals[0])
    predict_vals=np.zeros(load.shape)
    for i in range(3,len(time)):
        predict_vals[i]=matrix_1@true_vals[i-1]+matrix_2@true_vals[i-2]+matrix_3@true_vals[i-3]
    plt.plot((predict_vals[:,0]),label="Pred. NO load")
    plt.plot((true_vals[:,0]),label="True NO load")
    #plt.plot((predict_vals[:,0]-true_vals[:,0]),label="difference")
    plt.legend()
    plt.show()
#plot_predict_loads()
def print_latex_tables():
    print("Wind,Wind,Sun")
    print(model_windsun.summary())
    print(model_windsun.sigma_u)
    print(model_windsun.coefs)
    import array_to_latex as a2l
    latex_code = a2l.to_ltx(model_windsun.coefs[0], frmt = '{:6.6f}', arraytype = 'pmatrix')
    latex_code = a2l.to_ltx(model_windsun.sigma_u.to_numpy(), frmt = '{:6.6f}', arraytype = 'pmatrix')
    print("Load,Load")
    print(model_load.summary())
    print(model_load.sigma_u)
    print(model_load.coefs)

    import array_to_latex as a2l
    latex_code = a2l.to_ltx(model_load.coefs[0], frmt = '{:6.6f}', arraytype = 'pmatrix')
    latex_code = a2l.to_ltx(model_load.coefs[1], frmt = '{:6.6f}', arraytype = 'pmatrix')
    latex_code = a2l.to_ltx(model_load.coefs[2], frmt = '{:6.6f}', arraytype = 'pmatrix')
    latex_code = a2l.to_ltx(model_load.sigma_u.to_numpy(), frmt = '{:6.6f}', arraytype = 'pmatrix')
    print("Load,Water")
    print(model_NOloadwater.summary())
    print(model_NOloadwater.sigma_u)
    print(model_NOloadwater.sigma_u)
    latex_code = a2l.to_ltx(model_NOloadwater.sigma_u.to_numpy(), frmt = '{:6.6f}', arraytype = 'pmatrix')
    latex_code = a2l.to_ltx(model_NOloadwater.sigma_u.to_numpy()/4, frmt = '{:6.6f}', arraytype = 'pmatrix')
print_latex_tables()
seed=np.random.randint(0,100000)

wind_sun=VARSampler(3,1,model_windsun.coefs,model_windsun.sigma_u.to_numpy(),seed=seed+1)
propagationSteps=wind_sun.sample_series((num_years+2)*52)
time=np.linspace(1,(num_years+2)*52,(num_years+2)*52)
wind_NO_test=np.exp(propagationSteps[0]+wind_NO_function(time))
wind_DE_test=np.exp(propagationSteps[1]+wind_DE_function(time))
sun_DE_test=np.exp(propagationSteps[2]+solar_DE_function(time))
loadings=VARSampler(2,3,model_load.coefs,model_load.sigma_u.to_numpy(),seed=seed+2)
propagationSteps,loadingstrandom=loadings.sample_series((num_years+2)*52,returnRandom=True)
load_NO_test=np.exp(propagationSteps[0]+load_NO_function(time))
load_DE_test=np.exp(propagationSteps[1]+load_DE_function(time))
water_test_withoutfunc=water_sampler(loadingstrandom[:,0],model_NOloadwater.sigma_u.to_numpy()/4,seed=seed+3)
water_NO_test=np.exp(water_test_withoutfunc+water_NO_function(time))
plt.plot(wind_NO_test,label="wind NO",color=colors[0])
plt.plot(wind_DE_test,label="wind DE",color=colors[1])
plt.plot(sun_DE_test,label="sun DE",color=colors[5])
plt.plot(load_NO_test,label="load NO",color=colors[2])
plt.plot(load_DE_test,label="load DE",color=colors[3])
plt.plot(water_NO_test,label="water NO",color=colors[4])
plt.axvline(52*num_years,linestyle="--",color="grey",label="future line")
plt.title("Example of a system simulated 2017-2022")
plt.legend(loc="upper left")
plt.tight_layout()
plt.xlabel("week")
plt.ylabel("MW")
plt.savefig("../graphs/testing_predictions_enhancedfourier.pdf")
plt.show()

trend_dict={}
season_dict={}
max_season=len(max(season_coefs, key=len))
for i in range(6):
    season_dict_element=np.zeros(max_season)
    season_dict_element[max_season-len(season_coefs[i]):]=season_coefs[i]
    trend_dict[order[i]]=trend_coefs[i]
    season_dict[order[i]]=season_dict_element

trend_pd=pd.DataFrame(trend_dict)
season_pd=pd.DataFrame(season_dict)
#wind_sun_pd=pd.DataFrame({"coefs_0":model_windsun.coefs[0]})
#loads=pd.DataFrame({"coefs":model_load.coefs,"sigma":model_load.sigma_u})
#water=pd.DataFrame({"sigma:":model_NOloadwater.sigma_u/4})
import scipy.io
mdict={"load_coefs":model_load.coefs,"windsun_coefs":model_windsun.coefs,"load_sigma":model_load.sigma_u.to_numpy(),"windsun_sigma":model_windsun.sigma_u.to_numpy(),"water_sigma":model_NOloadwater.sigma_u.to_numpy()/4}
scipy.io.savemat("../data/timeseriesfourier.mat", mdict=mdict, oned_as='row')
trend_pd.to_csv("../data/trendsfourier.csv")
season_pd.to_csv("../data/seasonfourier.csv")
