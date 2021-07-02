import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from cases import *
from helper_functions import *
from sklearn.neighbors import KernelDensity
order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
periods=[52,52,52,52,13,52]
colors=["cyan","black","green","red","blue","orange"]
coefs=scipy.io.loadmat("../data/timeseries.mat")
trend_coefs=pd.read_csv("../data/trends.csv")
#trend_coefs["water NO"][0]=trend_coefs["load NO"][0]*2 #make water raise as high as production
#trend_coefs["wind NO"][0]*=0.5 #make water raise as high as production
#trend_coefs["load NO"][0]*=0.5 #make water raise as high as production
trend_coefs["wind DE"][0]*=0.8 #make water raise as high as production

season_coefs=pd.read_csv("../data/season.csv")

functions=[]
for i in range(len(order)):
    trend=trend_coefs[order[i]]
    season=season_coefs[order[i]]
    functions.append(coefs_to_function(trend,season,period=periods[i]))
try:
    start_year=int(sys.argv[1])
    num_years=int(sys.argv[2])
    num_simulations=int(sys.argv[3])
except:
    start_year=2020
    num_years=1
    num_simulations=10000

if start_year==2022:
    mean_wind = 2.048
elif start_year==2020:
    mean_wind=0.41
else:
    #If the mean is not been precalculated, estimate it from a different, independent simulation
    case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=24958234534253245,start_year=start_year,num_years=num_years)
    case0_simulator.simulate_n_years(n=1000)
    mean_wind=np.mean(case0_simulator.wind_toNorway/1e6)
seed=1234567
case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years)
case0_simulator.simulate_n_years(n=num_simulations)
CO2_case0=case0_simulator.get_CO2()/1e9
case2_simulator=case2(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0,mean_wind=mean_wind)
case2_simulator.simulate_n_years(n=num_simulations)
CO2_case2=case2_simulator.get_CO2()/1e9
exp_balance_case2=case2_simulator.import_export_balance
nor_balance_case2=case2_simulator.norwegian_balance
nor_balance_case0=case0_simulator.norwegian_balance
