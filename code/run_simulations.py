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
#trend_coefs["water NO"][0]=trend_coefs["load NO"][0]*2 #make water raise as fast  as production
season_coefs=pd.read_csv("../data/season.csv")
#mean_winddict={2022:2.048,2020:0.41}
mean_winddict={}
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
filename_case3_1="../data/case3_1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_2="../data/case3_2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_3="../data/case3_3_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)

filename_case4="../data/case4_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)

try:
    case0_data=pd.read_csv(filename_case0)
except FileNotFoundError:
    print("Case 0 not found, simulating")
    case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years)
    case0_simulator.simulate_n_years(n=num_simulations)
    nor_balance_0=case0_simulator.norwegian_balance/1e6
    nor_balance_0_nowind=case0_simulator.norwegian_balance_nowind/1e6
    CO2_hist_case0=case0_simulator.get_CO2()/1e9
    CO2_hist_case0_nowind=case0_simulator.CO2_nowind/1e9
    german_wind_surplus=case0_simulator.wind_surplus/1e6
    german_wind_toNorway=case0_simulator.wind_toNorway/1e6
    results={"CO2":CO2_hist_case0,"CO2 nowind":CO2_hist_case0_nowind,"Norwegian Balance nowind":nor_balance_0_nowind,"Norwegian Balance":nor_balance_0,"German wind surplus":german_wind_surplus,"German wind to Norway":german_wind_toNorway}
    results["wind NO"]=case0_simulator.profiles[:,0]/1e6
    results["wind DE"]=case0_simulator.profiles[:,1]/1e6
    results["load NO"]=case0_simulator.profiles[:,2]/1e6
    results["load DE"]=case0_simulator.profiles[:,3]/1e6
    results["water NO"]=case0_simulator.profiles[:,4]/1e6
    results["solar DE"]=case0_simulator.profiles[:,5]/1e6
    case0_results=pd.DataFrame(results)
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
    DEtoNO_case1=case1_simulator.num_days_DEtoNO
    NOtoDE_case1=case1_simulator.num_days_NOtoDE
    case1_results=pd.DataFrame({"Days NO to DE":NOtoDE_case1, "Days DE to NO":DEtoNO_case1, "CO2":CO2_hist_case1,"Norwegian Balance":nor_balance_1,"Norwegian export":exp_balance_case1})
    case1_results.to_csv(filename_case1)
try:
    case1_delay1_data=pd.read_csv(filename_case1_delay1)
except FileNotFoundError:
    print("Case 1 delay1 not found, simulating")
    case1_simulator=case1_delay1(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=1,delay_NOtoDE=1)
    case1_simulator.simulate_n_years(n=num_simulations)
    nor_balance_1=case1_simulator.norwegian_balance/1e6
    CO2_hist_case1=case1_simulator.get_CO2()/1e9
    exp_balance_case1=-case1_simulator.import_export_balance/1e6
    DEtoNO_case1=case1_simulator.num_days_DEtoNO
    NOtoDE_case1=case1_simulator.num_days_NOtoDE
    case1_results=pd.DataFrame({"Days NO to DE":NOtoDE_case1, "Days DE to NO":DEtoNO_case1,"CO2":CO2_hist_case1,"Norwegian Balance":nor_balance_1,"Norwegian export":exp_balance_case1})
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
try:
    case3_1_data=pd.read_csv(filename_case3_1)
except FileNotFoundError:
    print("Case 3_1 not found, simulating")
    case3_1_simulator=case3_1(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
    case3_1_simulator.simulate_n_years(n=num_simulations)
    nor_balance_case3_1=case3_1_simulator.norwegian_balance/1e6
    CO2_hist_case3_1=case3_1_simulator.get_CO2()/1e9
    CO2_bad_hist_case3_1=case3_1_simulator.CO2_bad/1e9
    exp_balance_case3_1=-case3_1_simulator.import_export_balance/1e6
    case3_1_results=pd.DataFrame({"CO2":CO2_hist_case3_1,"CO2 bad":CO2_bad_hist_case3_1,"Norwegian Balance":nor_balance_case3_1})
    case3_1_results.to_csv(filename_case3_1)
try:
    case3_3_data=pd.read_csv(filename_case3_3)
except FileNotFoundError:
    print("Case 3_3 not found, simulating")
    case3_3_simulator=case3_3(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
    case3_3_simulator.simulate_n_years(n=num_simulations)
    nor_balance_case3_3=case3_3_simulator.norwegian_balance/1e6
    CO2_hist_case3_3=case3_3_simulator.get_CO2()/1e9
    CO2_bad_hist_case3_3=case3_3_simulator.CO2_bad/1e9
    exp_balance_case3_3=-case3_3_simulator.import_export_balance/1e6
    case3_3_results=pd.DataFrame({"CO2":CO2_hist_case3_3,"CO2 bad":CO2_bad_hist_case3_3,"Norwegian Balance":nor_balance_case3_3,"Norwegian export":exp_balance_case3_3})
    case3_3_results.to_csv(filename_case3_3)
try:
    case3_2_data=pd.read_csv(filename_case3_2)
except FileNotFoundError:
    print("Case 3_2 not found, simulating")
    try: #Check if mean_wind for this year is already defined. If not...
        mean_wind=mean_winddict[start_year]
    except KeyError:
        case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=24958234534253245,start_year=start_year,num_years=num_years)
        case0_simulator.simulate_n_years(n=1000)
        mean_wind=np.mean(case0_simulator.wind_toNorway/1e6)
    case3_2_simulator=case3_2(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0,mean_wind=mean_wind)
    case3_2_simulator.simulate_n_years(n=num_simulations)
    nor_balance_case3_2=case3_2_simulator.norwegian_balance/1e6
    CO2_hist_case3_2=case3_2_simulator.get_CO2()/1e9
    CO2_bad_hist_case3_2=case3_2_simulator.CO2_bad/1e9
    exp_balance_case3_2=-case3_2_simulator.import_export_balance/1e6
    case3_2_results=pd.DataFrame({"CO2":CO2_hist_case3_2,"CO2 bad":CO2_bad_hist_case3_2,"Norwegian Balance":nor_balance_case3_2})
    case3_2_results.to_csv(filename_case3_2)
try:
    case4_data=pd.read_csv(filename_case4)
except FileNotFoundError:
    print("Case 4 not found, simulating")
    case4_simulator=case4(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
    case4_simulator.simulate_n_years(n=num_simulations)
    nor_balance_case4=case4_simulator.norwegian_balance/1e6
    CO2_hist_case4=case4_simulator.get_CO2()/1e9
    CO2_bad_hist_case4=case4_simulator.CO2_bad/1e9
    exp_balance_case4=-case4_simulator.import_export_balance/1e6
    case4_results=pd.DataFrame({"CO2":CO2_hist_case4,"Norwegian Balance":nor_balance_case4,"Norwegian export":exp_balance_case4, "CO2 bad":CO2_bad_hist_case4})
    case4_results.to_csv(filename_case4)
