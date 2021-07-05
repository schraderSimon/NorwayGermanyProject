import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from cases import *
from helper_functions import *
from sklearn.neighbors import KernelDensity
from scipy import stats
plt.rcParams.update({'font.size': 14})

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
    num_simulations=int(sys.argv[3])
    if sys.argv[4]=="True":
        savefile=True
    else:
        savefile=False
except IndexError:
    start_year=2020
    num_years=1
    num_simulations=10000
    savefile=False
seed=0

filename_case0="../data/case0_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case1="../data/case1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case2="../data/case2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case1_delay1="../data/case1_%d_%d_%d_%d_delay1.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_1="../data/case3_1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_2="../data/case3_2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)

case0_data=pd.read_csv(filename_case0)
case1_data=pd.read_csv(filename_case1)
case1_delay1_data=pd.read_csv(filename_case1_delay1)
case2_data=pd.read_csv(filename_case2)
case3_1_data=pd.read_csv(filename_case3_1)
case3_2_data=pd.read_csv(filename_case3_2)
german_wind_surplus=case0_data["German wind surplus"].to_numpy()
german_wind_toNorway=case0_data["German wind to Norway"].to_numpy()

over_10_rate=len(german_wind_surplus[np.where(german_wind_surplus>5)])/num_simulations
print("Wind overproduction: %.3f±%.3f"%(np.mean(german_wind_surplus),np.std(german_wind_surplus)))
print("Wind send to Norway: %.3f±%.3f"%(np.mean(german_wind_toNorway),np.std(german_wind_toNorway)))
print("Outlayer rate (above 5 TWh): %f"%(over_10_rate))
plt.title("Wind overproduction, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
kde = KernelDensity(kernel='exponential', bandwidth=0.05).fit(german_wind_surplus.reshape(-1,1))
dens=np.exp(kde.score_samples(np.linspace(0,5,1000).reshape(-1,1)))
plt.plot(np.linspace(0,5,1000),dens,color="red",alpha=0.5)
plt.hist(german_wind_surplus,bins=200,density=True,range=(0,10),label="Total wind overproduction",color="red",alpha=0.5)
bins=int(200/10*np.max(german_wind_toNorway))
plt.hist(german_wind_toNorway,bins=bins,density=True,label="Exportable wind overproduction",color="blue",alpha=0.5)
sns.kdeplot(german_wind_toNorway,color="blue",bw_adjust=0.4,alpha=0.5)
plt.xlabel(r"TWh ")
plt.xlim(0,5)
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
if savefile:
    plt.savefig("../graphs/wind_%d.pdf"%(start_year))
plt.show()

CO2_hist_case0=case0_data["CO2"].to_numpy()
CO2_hist_case1=case1_data["CO2"].to_numpy()
CO2_hist_case1_delay1=case1_delay1_data["CO2"].to_numpy()
CO2_hist_case2=case2_data["CO2"].to_numpy()
CO2_hist_case3_1=case3_1_data["CO2"].to_numpy()
CO2_bad_hist_case3_1=case3_1_data["CO2 bad"].to_numpy()
CO2_hist_case3_2=case3_2_data["CO2"].to_numpy()
CO2_bad_hist_case3_2=case3_2_data["CO2 bad"].to_numpy()

exp_balance_case1=case1_data["Norwegian export"].to_numpy()
exp_balance_case1_delay1=case1_delay1_data["Norwegian export"].to_numpy()
exp_balance_case2=case2_data["Norwegian export"].to_numpy()

nor_balance_case0=case0_data["Norwegian Balance"].to_numpy()
nor_balance_case1=case1_data["Norwegian Balance"].to_numpy()
nor_balance_case1_delay1=case1_delay1_data["Norwegian Balance"].to_numpy()
nor_balance_case2=case2_data["Norwegian Balance"].to_numpy()
nor_balance_case3_1=case3_1_data["Norwegian Balance"].to_numpy()
nor_balance_case3_2=case3_2_data["Norwegian Balance"].to_numpy()

num_weeks_NOtoDE=case1_data["Days NO to DE"].to_numpy()
num_weeks_DEtoNO=case1_data["Days DE to NO"].to_numpy()

plt.hist(num_weeks_NOtoDE,density=True,alpha=0.1,bins=30,color="red")
plt.hist(num_weeks_DEtoNO,density=True,alpha=0.1,bins=20,color="blue")
sns.kdeplot(num_weeks_NOtoDE,color="red",bw_adjust=2,label="NO to DE")
sns.kdeplot(num_weeks_DEtoNO,color="blue",bw_adjust=2,label="DE to NO")
plt.xlabel("Number of weeks")
plt.ylabel("Probability")
plt.legend()
plt.title(r"Number of weeks exporting, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

plt.tight_layout()
if savefile:
    plt.savefig("../graphs/Num_weeks_%d.pdf"%(start_year))
plt.show()


fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))

ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="red")#,ax=ax2)
ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")#,ax=ax2)
ax2.hist(CO2_hist_case1_delay1,density=True,alpha=0.1,bins=20,color="green")#,ax=ax2)

sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Case 0",color="red",ax=ax2)
sns.kdeplot(CO2_hist_case1,label="Case 1, delay 0",color="blue",ax=ax2)
sns.kdeplot(CO2_hist_case1_delay1,label="Case 1, delay 1",color="green",ax=ax2)
print("CO2:")
print("Case 0: %.2f±%.2f"%(np.mean(CO2_hist_case0),np.std(CO2_hist_case0)))
print("Case 1 delay 0: %.2f±%.2f"%(np.mean(CO2_hist_case1),np.std(CO2_hist_case1)))
print("Case 1 delay 1: %.2f±%.2f"%(np.mean(CO2_hist_case1_delay1),np.std(CO2_hist_case1_delay1)))
print("Reduction beetween Case 1 and Case 0: %.2f±%.2f"%(np.mean(CO2_hist_case1-CO2_hist_case0),np.std(CO2_hist_case1-CO2_hist_case0)))
ax2.set_xlabel(r"Million Tons CO$_2$")
ax2.set_ylabel("Probability")
ax2.legend()
ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

ax1.set_title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(nor_balance_case0,label="NO el. surplus, case 0",color="red",ax=ax1)
sns.kdeplot(exp_balance_case1,label="Norwegian el. Export, case 1",color="blue",ax=ax1)
sns.kdeplot(nor_balance_case1,label="NO el. surplus, case 1",color="cyan",ax=ax1)
ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")#,ax=ax1)
ax1.hist(exp_balance_case1,bins=20,density=True,alpha=0.1,color="blue")#,ax=ax1)
ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="cyan")#,ax=ax1)

print("Norwegian surplus case 0: %.4f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
print("Norwegian surplus case 1: %.4f±%.4f"%(np.mean(nor_balance_case1),np.std(nor_balance_case1)))
print("Norwegian export case 1: %.4f±%.4f"%(np.mean(exp_balance_case1),np.std(exp_balance_case1)))
print("Difference in surplus beetween Case 1 and Case 0: %.2f±%.2f"%(np.mean((nor_balance_case1-nor_balance_case0)),np.std((nor_balance_case1-nor_balance_case0))))
ax1.set_xlabel("TWh")
ax1.set_ylabel("Probability")
ax1.legend(loc="upper left")
ax1.set_xlim(-5,25)
plt.tight_layout()
if savefile:
    plt.savefig("../graphs/case0_case1_%d.pdf"%(start_year))
plt.show()



fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))

ax2.hist(CO2_hist_case2,density=True,alpha=0.1,bins=20,color="green")
sns.kdeplot(CO2_hist_case2,x=r"Million tons CO$_2$",label="Case 2",color="green",ax=ax2)
ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="blue")
sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Case 0",color="blue",ax=ax2)

print("Reduction beetween Case 2 and Case 0: %.4f±%.4f"%(np.mean(CO2_hist_case0-CO2_hist_case2),np.std(CO2_hist_case0-CO2_hist_case2)))
ax2.set_xlabel(r"Million Tons CO$_2$")
ax2.set_ylabel("Probability")
ax2.legend()
ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))



ax1.set_title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(nor_balance_case0,label="NO el. surplus, case 0",color="red",ax=ax1)
sns.kdeplot(exp_balance_case2,label="Norwegian el. Export, case 2",color="blue",ax=ax1)
sns.kdeplot(nor_balance_case2,label="NO el. surplus, case 2",color="green",ax=ax1)
ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")
ax1.hist(exp_balance_case2,bins=20,density=True,alpha=0.1,color="blue")
ax1.hist(nor_balance_case2,bins=20,density=True,alpha=0.1,color="green")
print("Norwegian surplus case 0: %.10f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
print("Norwegian surplus case 2: %.10f±%.4f"%(np.mean(nor_balance_case2),np.std(nor_balance_case2)))
statistic,pval=stats.ks_2samp(nor_balance_case0, nor_balance_case2)
if pval>0.05:
    print("Null hypothesis cannot be rejected, p=%.2f, nor_balance_case0 and nor_balance_case2 might have the same distribution"%pval)
else:
    print("Null hypothesis can be rejected, p=%.2f, nor_balance_case0 and nor_balance_case2 probably do not have the same distribution"%pval)

print("Norwegian export case 2: %.4f±%.4f"%(np.mean(exp_balance_case2),np.std(exp_balance_case2)))
print("Difference in surplus beetween Case 1 and Case 0: %.2f±%.2f"%(np.mean((nor_balance_case2-nor_balance_case0)),np.std((nor_balance_case2-nor_balance_case0))))

ax1.set_xlabel("TWh")
ax1.set_ylabel("Probability")
ax1.legend(loc="upper right")
plt.tight_layout()
if savefile:
    plt.savefig("../graphs/case0_case2_%d.pdf"%(start_year))
plt.show()


fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")

sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3_1 (480)",color="red",ax=ax2)
sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3_1 (333)",color="orange",ax=ax2)
sns.kdeplot(CO2_hist_case1,label="Case 1",color="blue",ax=ax2)
ax2.set_xlabel(r"Million Tons CO$_2$")
ax2.set_ylabel("Probability")
ax2.legend()
ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, case 3-1",color="red",ax=ax1)
sns.kdeplot(nor_balance_case1,label="NO el. surplus, case 1",color="green",ax=ax1)
ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="green")
ax1.set_xlabel("TWh")
ax1.set_ylabel("Probability")
ax1.legend(loc="upper left")
ax1.set_xlim(-5,20)
plt.tight_layout()
if savefile:
    plt.savefig("../graphs/case1_case3_%d.pdf"%(start_year))
print("CO2 Reduction with Case 3-1 (bad) compared to Case 1: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case1)),np.std((CO2_bad_hist_case3_1-CO2_hist_case1))))
print("CO2 Reduction with Case 3-1 (good) compared to Case 1:  %.2f±%.2f"%(np.mean((CO2_hist_case3_1-CO2_hist_case1)),np.std((CO2_hist_case3_1-CO2_hist_case1))))
print("Load Reduction with Case 3-1 compared to Case 1:  %.2f±%.2f"%(np.mean((nor_balance_case3_1-nor_balance_case1)),np.std((nor_balance_case3_1-nor_balance_case1))))

plt.show()



fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
ax2.hist(CO2_hist_case3_2,density=True,alpha=0.1,bins=20,color="blue")
ax2.hist(CO2_bad_hist_case3_2,density=True,alpha=0.1,bins=20,color="green")

sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3_1 (480)",color="red",ax=ax2)
sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3_1 (333)",color="orange",ax=ax2)
sns.kdeplot(CO2_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3_2 (480)",color="blue",ax=ax2)
sns.kdeplot(CO2_bad_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3_2 (333)",color="green",ax=ax2)
ax2.set_xlabel(r"Million Tons CO$_2$")
ax2.set_ylabel("Probability")
ax2.legend()
ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, case 3-1",color="red",ax=ax1)
ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
sns.kdeplot(nor_balance_case3_2,label="NO el. surplus, case 3-2",color="blue",ax=ax1)
ax1.hist(nor_balance_case3_2,density=True,bins=20,alpha=0.1,color="blue")

ax1.set_xlabel("TWh")
ax1.set_ylabel("Probability")
ax1.legend(loc="upper left")
ax1.set_xlim(-5,20)
plt.tight_layout()
if savefile:
    plt.savefig("../graphs/case1_case3-2_%d.pdf"%(start_year))
print("CO2 Reduction with Case 3-2 (bad) compared to Case 3-1 (bad):  %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1)),np.std((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1))))
print("CO2 Reduction with Case 3-2 (good) compared to Case 2 (good):  %.2f±%.2f"%(np.mean((CO2_hist_case3_2-CO2_hist_case3_1)),np.std((CO2_hist_case3_2-CO2_hist_case3_1))))
print("Load Reduction with Case 3-2 compared to Case 3-1: %.2f±%.2f"%(np.mean((nor_balance_case3_2-nor_balance_case3_1)),np.std((nor_balance_case3_2-nor_balance_case3_1))))

plt.show()
