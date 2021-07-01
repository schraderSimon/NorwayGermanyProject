import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from cases import *
def coefs_to_function(trend_coef,season_coef,period=52):
    trendfunc=np.poly1d(trend_coef)
    seasonfunc=np.poly1d(season_coef)
    def seasonfunc_periodic(t):
        return seasonfunc((t-1)%period)
    return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))

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
    start_year=2020
    num_years=1

def compare_cases():
    num_years=1
    start_year=2020
    num_simulations=100
    seed=0
    case1_simulator=case1(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
    case1_simulator.sendable_max=0
    case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=seed,start_year=start_year,num_years=num_years)
    case0_simulator.simulate_n_years(n=num_simulations)
    case1_simulator.simulate_n_years(n=num_simulations)
    assert np.all(np.abs(case0_simulator.get_CO2()-case1_simulator.get_CO2())/1e9<1e-12)
compare_cases()

num_simulations=10000
seed=0
case0_simulator=case0(coefs,trend_coefs,season_coefs,seed=0,start_year=start_year,num_years=num_years)
case0_simulator.simulate_n_years(n=num_simulations)
german_wind_surplus=case0_simulator.wind_surplus/1e6
german_wind_toNorway=case0_simulator.wind_toNorway/1e6
print(np.min(german_wind_surplus))
print(np.min(german_wind_toNorway))

plt.title("Wind,n=%d, year=%d,years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(german_wind_surplus,label="Total wind overproduction")
sns.kdeplot(german_wind_toNorway,label="Exportable wind overproduction")
plt.xlabel(r"TWh ")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("../graphs/wind.pdf")
plt.show()
#case1_simulator.plotlast()
case1_simulator_delay0=case1(coefs,trend_coefs,season_coefs,seed=0,start_year=start_year,num_years=num_years,delay_DEtoNO=0,delay_NOtoDE=0)
case1_simulator_delay1=case1(coefs,trend_coefs,season_coefs,seed=0,start_year=start_year,num_years=num_years,delay_DEtoNO=1,delay_NOtoDE=1)
case1_simulator_delay0.simulate_n_years(n=num_simulations)
case1_simulator_delay1.simulate_n_years(n=num_simulations)

CO2_hist_case0=case0_simulator.get_CO2()/1e9
CO2_hist_case1_delay0=case1_simulator_delay0.get_CO2()/1e9
CO2_hist_case1_delay1=case1_simulator_delay1.get_CO2()/1e9

exp_balance_case1_delay0=case1_simulator_delay0.import_export_balance
exp_balance_case1_delay1=case1_simulator_delay1.import_export_balance
nor_balance_case1_delay0=case1_simulator_delay0.norwegian_balance
nor_balance_case1_delay1=case1_simulator_delay1.norwegian_balance
nor_balance_0=case0_simulator.norwegian_balance


sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Case 0")
sns.kdeplot(CO2_hist_case1_delay0,label="Case 1, delay 0")
sns.kdeplot(CO2_hist_case1_delay1,label="Case 1, delay 1")
print("CO2:")
print("Case 0: %.2f+-%.2f"%(np.mean(CO2_hist_case0),np.std(CO2_hist_case0)))
print("Case 1 delay 0: %.2f+-%.2f"%(np.mean(CO2_hist_case1_delay0),np.std(CO2_hist_case1_delay0)))
print("Case 1 delay 1: %.2f+-%.2f"%(np.mean(CO2_hist_case1_delay1),np.std(CO2_hist_case1_delay1)))
print("Reduction beetween Case 1 and Case 0: %.2f+-%.2f"%(np.mean(CO2_hist_case1_delay1-CO2_hist_case0),np.std(CO2_hist_case1_delay1-CO2_hist_case0)))
plt.xlabel(r"Million Tons CO$_2$")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("../graphs/Emissions_case0_case1.pdf")
plt.show()
plt.title("Electricity, n=%d, year=%d,years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(nor_balance_0/1e6,label="NO el. surplus, case 0")
sns.kdeplot(-exp_balance_case1_delay0/1e6,label="Norwegian el. Export, case 1, delay 0")
sns.kdeplot(nor_balance_case1_delay0/1e6,label="NO el. surplus, case 1, delay 0")
print("Norwegian surplus case 0: %.4f+-%.4f"%(np.mean(nor_balance_0/1e6),np.std(nor_balance_0/1e6)))
print("Norwegian surplus case 1: %.4f+-%.4f"%(np.mean(nor_balance_case1_delay1/1e6),np.std(nor_balance_case1_delay1/1e6)))
print("Norwegian export case 1: %.4f+-%.4f"%(np.mean(-exp_balance_case1_delay1/1e6),np.std(-exp_balance_case1_delay1/1e6)))
print("Difference in surplus beetween Case 1 and Case 0: %.2f+-%.2f"%(np.mean((nor_balance_case1_delay1-nor_balance_0)/1e6),np.std((nor_balance_case1_delay1-nor_balance_0)/1e6)))
plt.xlabel("TWh")
plt.ylabel("Probability")
plt.legend(loc="upper left")
plt.tight_layout()

plt.savefig("../graphs/Balance_case0_case1.pdf")
plt.show()
