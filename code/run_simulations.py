import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from cases import *
from helper_functions import *
from sklearn.neighbors import KernelDensity
periods=[52,52,52,52,13,52]
order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
coefs=scipy.io.loadmat("../data/timeseries.mat")
trend_coefs=pd.read_csv("../data/trends.csv")
#trend_coefs["water NO"][0]=trend_coefs["load NO"][0]*4 #make water raise as high as production
season_coefs=pd.read_csv("../data/season.csv")
mean_winddict={2022:2.048,2020:0.41}
functions=[]
for i in range(len(order)):
    trend=trend_coefs[order[i]]
    season=season_coefs[order[i]]
    functions.append(coefs_to_function(trend,season,period=periods[i]))
try:
    start_year=int(sys.argv[1])
    num_years=int(sys.argv[2])
    num_simulations=int(sys.argv[3])
    seed=int(sys.argv[4])
except:
    start_year=2020
    num_years=1
    num_simulations=10000
    seed=0

filename_case0="../data/case0_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case1="../data/case1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case1_delay1="../data/case1_%d_%d_%d_%d_delay1.csv"%(start_year,num_years,num_simulations,seed)

filename_case2="../data/case2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)

try:
    case0_data=pd.read_csv(filename_case0)
except FileNotFoundError:
    print("Case 0 not found, simulating")
    case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years)
    case0_simulator.simulate_n_years(n=num_simulations)
    nor_balance_0=case0_simulator.norwegian_balance/1e6
    CO2_hist_case0=case0_simulator.get_CO2()/1e9
    german_wind_surplus=case0_simulator.wind_surplus/1e6
    german_wind_toNorway=case0_simulator.wind_toNorway/1e6
    case0_results=pd.DataFrame({"CO2":CO2_hist_case0,"Norwegian Balance":nor_balance_0,"German wind surplus":german_wind_surplus,"German wind to Norway":german_wind_toNorway})
    case0_results.to_csv(filename_case0)
try:
    case1_data=pd.read_csv(filename_case1)
except FileNotFoundError:
    print("Case 1 not found, simulating")
    case1_simulator=case1(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
    case1_simulator.simulate_n_years(n=num_simulations)
    nor_balance_1=case1_simulator.norwegian_balance/1e6
    CO2_hist_case1=case1_simulator.get_CO2()/1e9
    exp_balance_case1=-case1_simulator.import_export_balance/1e6
    case1_results=pd.DataFrame({"CO2":CO2_hist_case1,"Norwegian Balance":nor_balance_1,"Norwegian export":exp_balance_case1})
    case1_results.to_csv(filename_case1)
try:
    case1_delay1_data=pd.read_csv(filename_case1_delay1)
except FileNotFoundError:
    print("Case 1 delay1 not found, simulating")
    case1_simulator=case1(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=1,delay_NOtoDE=1)
    case1_simulator.simulate_n_years(n=num_simulations)
    nor_balance_1=case1_simulator.norwegian_balance/1e6
    CO2_hist_case1=case1_simulator.get_CO2()/1e9
    exp_balance_case1=-case1_simulator.import_export_balance/1e6
    case1_results=pd.DataFrame({"CO2":CO2_hist_case1,"Norwegian Balance":nor_balance_1,"Norwegian export":exp_balance_case1})
    case1_results.to_csv(filename_case1_delay1)
try:
    case2_data=pd.read_csv(filename_case2)
except FileNotFoundError:
    print("Case 2 not found, simulating")
    try: #Check if mean_wind for this year is already defined. If not...
        mean_wind=mean_winddict[start_year]
    except KeyError:
        case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=24958234534253245,start_year=start_year,num_years=num_years)
        case0_simulator.simulate_n_years(n=1000)
        mean_wind=np.mean(case0_simulator.wind_toNorway/1e6)
    case2_simulator=case2(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0,mean_wind=mean_wind)
    case2_simulator.simulate_n_years(n=num_simulations)
    CO2_hist_case2=case2_simulator.get_CO2()/1e9
    exp_balance_case2=-case2_simulator.import_export_balance/1e6
    nor_balance_case2=case2_simulator.norwegian_balance/1e6
    case2_results=pd.DataFrame({"CO2":CO2_hist_case2,"Norwegian Balance":nor_balance_case2,"Norwegian export":exp_balance_case2})
    case2_results.to_csv(filename_case2)