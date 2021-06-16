import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def average_array(arr,length):
    return np.sum(arr.reshape(-1,length),axis=1)
def fill_nan(array):
    array_interpolated=pd.DataFrame(array).interpolate().values.ravel()
    return array_interpolated
data=pd.read_csv("time_series_60min_singleindex.csv")
print(data.tail())
time=data["cet_cest_timestamp"]
onyear_index_start=np.where(data=="2019-01-01T00:00:00+0100")[0][0]
onyear_index_end=np.where(data=="2020-01-01T00:00:00+0100")[0][0]-24 #drop the last day of the year for simplicity reasons I guess?

print("Index at: ",onyear_index_start)
time=time[onyear_index_start:onyear_index_end]
print(len(time)/24)
wind_DE=data["DE_wind_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() # Innstead of "DE_wind_capacity" which does not matter so much
load_DE=data["DE_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()
solar_DE=data["DE_solar_generation_actual"][onyear_index_start:onyear_index_end].to_numpy()
load_NO=data["NO_load_actual_entsoe_transparency"][onyear_index_start:onyear_index_end].to_numpy()
wind_NO=data["NO_wind_onshore_generation_actual"][onyear_index_start:onyear_index_end].to_numpy() #Assuming there is no offshore wind?
#time=np.linspace(1,52,52,dtype="int")
steplength=24 #day, multiply by 7 to get week
time=np.linspace(1,364,364,dtype="int")

wind_DE=average_array(wind_DE,steplength)/1e6
solar_DE=average_array(solar_DE,steplength)/1e6
wind_NO=average_array(wind_NO,steplength)/1e6
load_DE=average_array(load_DE,steplength)/1e6
load_NO=average_array(load_NO,steplength)/1e6
load_NO=fill_nan(load_NO)
plt.plot(time,wind_NO,label="wind NO")
plt.plot(time,wind_DE,label="wind DE")
plt.plot(time,load_NO,label="load NO")
plt.plot(time,load_DE,label="load DE")
plt.plot(time,solar_DE,label="solar DE")
plt.xlabel("week")
plt.ylabel("TW")
plt.legend()
plt.show()

print("Wind Norway total:%f"%(np.sum(wind_NO)))
print("Wind Germany total:%f"%(np.sum(wind_DE)))
print("Load Germany total:%f"%(np.sum(load_DE)))
print("Load Norway total:%f"%(np.sum(load_NO)))
