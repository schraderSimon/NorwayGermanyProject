import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
def average_array(arr,length):
    mean=np.mean(arr.reshape(-1,length),axis=1)
    std=np.std(arr.reshape(-1,length),axis=1)
    return mean,std
def fill_nan(array):
    array_interpolated=pd.DataFrame(array).interpolate().values.ravel()
    return array_interpolated
data=pd.read_csv("../data/time_series_60min_singleindex_2020.csv")
print(data.tail())
time=data["cet_cest_timestamp"]
begin_year=2015; begin_month=6;begin_day=1
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
steplength=int(sys.argv[1]) #day, multiply by 7 to get week
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
load_DE=fill_nan(load_DE)
solar_DE=fill_nan(solar_DE)
#wind_DE,solar_DE,wind_NO,load_NO,load_DE=np.log(wind_DE),np.log(solar_DE),np.log(wind_NO),np.log(load_NO),np.log(load_DE)
wind_DE_fit=np.poly1d(np.polyfit(time, wind_DE, 1))
solar_DE_fit=np.poly1d(np.polyfit(time, solar_DE, 1))
load_DE_fit=np.poly1d(np.polyfit(time, load_DE, 1))
wind_NO_fit=np.poly1d(np.polyfit(time, wind_NO, 1))
load_NO_fit=np.poly1d(np.polyfit(time, load_NO, 1))
plt.plot(time,wind_NO,label="wind NO",color="blue")
plt.plot(time,wind_NO_fit(time),"--",color="blue")
#plt.errorbar(time,wind_DE,wind_DE_std,label="wind DE")
plt.plot(time,wind_DE,label="wind DE",color="black")
plt.plot(time,wind_DE_fit(time),"--",color="black")
plt.plot(time,load_NO,label="load NO",color="green")
plt.plot(time,load_NO_fit(time),"--",color="green")
#plt.errorbar(time,load_DE,load_DE_std,label="load DE")
plt.plot(time,load_DE,label="load_DE",color="red")
plt.plot(time,load_DE_fit(time),"--",color="red")
#plt.errorbar(time,solar_DE,solar_DE_std,label="solar DE")
plt.plot(time,solar_DE,label="solar DE",color="orange")
plt.plot(time,solar_DE_fit(time),"--",color="orange")

#plt.plot(time,wind_DE+solar_DE,label="Wind and solar DE")

plt.xlabel("week")
plt.ylabel("MW")
plt.legend()
plt.savefig("../graphs/time_series_electricity_data.pdf")
plt.show()

print("Wind Norway total:%f"%(np.sum(wind_NO)))
print("Wind Germany total:%f"%(np.sum(wind_DE)))
print("Load Germany total:%f"%(np.sum(load_DE)))
print("Load Norway total:%f"%(np.sum(load_NO)))


plt.plot(time,wind_NO-wind_NO_fit(time),label="wind NO")
plt.plot(time,wind_DE-wind_DE_fit(time),label="wind DE")
plt.plot(time,load_NO-load_NO_fit(time),label="load NO")
plt.plot(time,load_DE-load_DE_fit(time),label="load_DE")
plt.plot(time,solar_DE-solar_DE_fit(time),label="solar DE")

plt.xlabel("week")
plt.ylabel("MW")
plt.legend()
plt.show()
