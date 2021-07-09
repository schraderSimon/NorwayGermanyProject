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

plt.rcParams.update({'font.size': 12, 'legend.labelspacing':0.2})
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
filename_case0_2020="../data/case0_2020_%d_%d_%d.csv"%(num_years,num_simulations,seed)
filename_case0_2022="../data/case0_2022_%d_%d_%d.csv"%(num_years,num_simulations,seed)

filename_case1="../data/case1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case2="../data/case2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case1_delay1="../data/case1_%d_%d_%d_%d_delay1.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_1="../data/case3_1_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_2="../data/case3_2_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case3_3="../data/case3_3_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)
filename_case4="../data/case4_%d_%d_%d_%d.csv"%(start_year,num_years,num_simulations,seed)

case0_data=pd.read_csv(filename_case0)
case0_data_2020=pd.read_csv(filename_case0_2020)
case0_data_2022=pd.read_csv(filename_case0_2022)

case1_data=pd.read_csv(filename_case1)
case1_delay1_data=pd.read_csv(filename_case1_delay1)
case2_data=pd.read_csv(filename_case2)
case3_1_data=pd.read_csv(filename_case3_1)
case3_2_data=pd.read_csv(filename_case3_2)
case3_3_data=pd.read_csv(filename_case3_3)
case4_data=pd.read_csv(filename_case4)
german_wind_surplus=case0_data["German wind surplus"].to_numpy()
german_wind_toNorway=case0_data["German wind to Norway"].to_numpy()
def plotwind():
    cutoff=20
    over_10_rate=len(german_wind_surplus[np.where(german_wind_surplus>cutoff)])/num_simulations
    print("Wind overproduction: %.3f±%.3f"%(np.mean(german_wind_surplus),np.std(german_wind_surplus)))
    print("Wind send to Norway: %.3f±%.3f"%(np.mean(german_wind_toNorway),np.std(german_wind_toNorway)))
    print("Outlayer rate (above %.f TWh): %f"%(cutoff,over_10_rate))
    plt.title("Wind, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
    kde = KernelDensity(kernel='exponential', bandwidth=0.05).fit(german_wind_surplus.reshape(-1,1))
    dens=np.exp(kde.score_samples(np.linspace(0,cutoff,1000).reshape(-1,1)))
    #plt.plot(np.linspace(0,cutoff,1000),dens,color="red",alpha=0.5)
    sns.kdeplot(german_wind_surplus,color="red",bw_adjust=0.4,alpha=0.5)

    plt.hist(german_wind_surplus,bins=200,density=True,range=(0,40),label="Total wind overproduction",color="red",alpha=0.5)
    bins=int(200/cutoff*np.max(german_wind_toNorway))
    plt.hist(german_wind_toNorway,bins=bins,density=True,label="Exportable wind overproduction",color="blue",alpha=0.5)
    sns.kdeplot(german_wind_toNorway,color="blue",bw_adjust=0.4,alpha=0.5)
    plt.xlabel(r"TWh ")
    plt.xlim(0,cutoff)
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/wind_%d.pdf"%(start_year))
    plt.show()

plt.rcParams.update({'font.size': 14})
CO2_hist_case0=case0_data["CO2"].to_numpy()
CO2_hist_case0_nowind=case0_data["CO2 nowind"].to_numpy()

CO2_hist_case1=case1_data["CO2"].to_numpy()
CO2_hist_case1_delay1=case1_delay1_data["CO2"].to_numpy()
CO2_hist_case2=case2_data["CO2"].to_numpy()
CO2_hist_case3_1=case3_1_data["CO2"].to_numpy()
CO2_hist_case3_3=case3_3_data["CO2"].to_numpy()
CO2_bad_hist_case3_1=case3_1_data["CO2 bad"].to_numpy()
CO2_hist_case3_2=case3_2_data["CO2"].to_numpy()
CO2_bad_hist_case3_2=case3_2_data["CO2 bad"].to_numpy()
CO2_bad_hist_case3_3=case3_3_data["CO2 bad"].to_numpy()
CO2_hist_case4=case4_data["CO2"].to_numpy()
CO2_bad_hist_case4=case4_data["CO2 bad"].to_numpy()


exp_balance_case1=case1_data["Norwegian export"].to_numpy()
exp_balance_case1_delay1=case1_delay1_data["Norwegian export"].to_numpy()
exp_balance_case2=case2_data["Norwegian export"].to_numpy()
exp_balance_case3_3=case3_3_data["Norwegian export"].to_numpy()


nor_balance_case0=case0_data["Norwegian Balance"].to_numpy()
nor_balance_case0_nowind=case0_data["Norwegian Balance nowind"].to_numpy()

nor_balance_case1=case1_data["Norwegian Balance"].to_numpy()
nor_balance_case1_delay1=case1_delay1_data["Norwegian Balance"].to_numpy()
nor_balance_case2=case2_data["Norwegian Balance"].to_numpy()
nor_balance_case3_1=case3_1_data["Norwegian Balance"].to_numpy()
nor_balance_case3_2=case3_2_data["Norwegian Balance"].to_numpy()
nor_balance_case3_3=case3_3_data["Norwegian Balance"].to_numpy()
nor_balance_case4=case4_data["Norwegian Balance"].to_numpy()

toGermany_case3_3=case3_3_data["Germany"].to_numpy()
toPlatforms_case3_3=case3_3_data["Platforms"].to_numpy()
toGermany_case4=case4_data["Germany"].to_numpy()
toPlatforms_case4=case4_data["Platforms"].to_numpy()


num_weeks_NOtoDE=case1_data["Days NO to DE"].to_numpy()
num_weeks_DEtoNO=case1_data["Days DE to NO"].to_numpy()
num_weeks_NOtoDE_delay1=case1_delay1_data["Days NO to DE"].to_numpy()
num_weeks_DEtoNO_delay1=case1_delay1_data["Days DE to NO"].to_numpy()
def boxwhisker():
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=False, sharey=True,figsize=(12,6))
    data=[]
    for i in range(6):
        data.append(case0_data_2020[order[i]].to_numpy())
    data=np.array(data).T
    #data=np.log(data)
    data_pandas=pd.DataFrame(data)
    data_pandas.columns=order
    print(data_pandas)
    #print(data)
    #plt.boxplot(data,vert=False)

    for i in range(6):
        sns.kdeplot(data[:,i],label=order[i],color=colors[i],ax=ax2)
    #g=sns.pairplot(data_pandas,kind="kde")
    #g.map_lower(sns.kdeplot, levels=4, color=".2")
    ax2.set_title("2020")
    ax2.legend()
    ax2.set_xlabel("TWh")
    data=[]
    for i in range(6):
        data.append(case0_data_2022[order[i]].to_numpy())
    data=np.array(data).T
    #data=np.log(data)
    data_pandas=pd.DataFrame(data)
    data_pandas.columns=order
    print(data_pandas)
    #print(data)
    #plt.boxplot(data,vert=False)

    for i in range(6):
        sns.kdeplot(data[:,i],label=order[i],color=colors[i],ax=ax1)
    #g=sns.pairplot(data_pandas,kind="kde")
    #g.map_lower(sns.kdeplot, levels=4, color=".2")
    ax1.set_title("2022")
    ax1.legend()
    ax1.set_xlabel("TWh")
    plt.tight_layout()
    plt.savefig("../graphs/correlations_production.pdf")
    plt.show()

def plotstuff():
    plt.hist(num_weeks_NOtoDE,density=True,alpha=0.1,bins=30,color="red")
    plt.hist(num_weeks_DEtoNO,density=True,alpha=0.1,bins=20,color="blue")
    sns.kdeplot(num_weeks_NOtoDE,color="red",bw_adjust=2,label="NO to DE, delay 0")
    sns.kdeplot(num_weeks_DEtoNO,color="blue",bw_adjust=2,label="DE to NO, delay 0")
    plt.hist(num_weeks_NOtoDE_delay1,density=True,alpha=0.1,bins=30,color="green")
    plt.hist(num_weeks_DEtoNO_delay1,density=True,alpha=0.1,bins=20,color="yellow")
    sns.kdeplot(num_weeks_NOtoDE_delay1,color="green",bw_adjust=2,label="NO to DE, delay 1")
    sns.kdeplot(num_weeks_DEtoNO_delay1,color="yellow",bw_adjust=2,label="DE to NO, delay 1")
    plt.xlabel("Number of weeks")
    plt.ylabel("Probability")
    plt.legend()
    plt.title(r"Num. weeks exporting, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/Num_weeks_%d.pdf"%(start_year))
    plt.show()
def plot1():
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
    sns.kdeplot(nor_balance_case1_delay1,label="NO el. surplus, case 1 delay 1",color="magenta",ax=ax1)
    ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")#,ax=ax1)
    ax1.hist(exp_balance_case1,bins=20,density=True,alpha=0.1,color="blue")#,ax=ax1)
    ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="cyan")#,ax=ax1)
    ax1.hist(nor_balance_case1_delay1,bins=20,density=True,alpha=0.1,color="magenta")
    print("Norwegian surplus case 0: %.4f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
    print("Norwegian surplus case 1: %.4f±%.4f"%(np.mean(nor_balance_case1),np.std(nor_balance_case1)))
    print("Norwegian export case 1: %.4f±%.4f"%(np.mean(exp_balance_case1),np.std(exp_balance_case1)))
    print("Difference in surplus beetween Case 1 and Case 0: %.2f±%.2f"%(np.mean((nor_balance_case1-nor_balance_case0)),np.std((nor_balance_case1-nor_balance_case0))))
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case0_case1_%d.pdf"%(start_year))
    plt.show()
    maxfit=10000
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=True,figsize=(12,6))
    ax2.hist2d(exp_balance_case1[:maxfit],CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit],bins=50,cmap="PuRd")
    m,b = np.polyfit(exp_balance_case1[:maxfit], CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit], 1)
    print("Slope: %f g/kWh"%(m*1000))
    def f(x):
        return m*x+b
    ax2.plot(exp_balance_case1[:maxfit],f(exp_balance_case1[:maxfit]),color="grey",label="linear fit")
    ax2.set_xlabel("TWh exported (netto)")
    ax2.set_ylabel(r"Million tons CO$_2$ saved")
    ax2.legend()
    maxnum=np.max(num_weeks_DEtoNO[:maxfit])
    minnum=np.min(num_weeks_DEtoNO[:maxfit])
    ax1.hist2d(num_weeks_DEtoNO[:maxfit],CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit],bins=(int(maxnum-minnum),20),cmap="PuRd")
    m,b = np.polyfit(num_weeks_DEtoNO[:maxfit], CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit], 1)
    def f2(x):
        return m*x+b
    ax1.plot(num_weeks_DEtoNO[:maxfit],f(num_weeks_DEtoNO[:maxfit]),color="grey",label="linear fit")
    ax1.set_xlabel(r"Number of weeks DE$\rightarrow$NO")
    plt.tight_layout()

    if savefile:
        plt.savefig("../graphs/case0_case1_confirm_%d.pdf"%(start_year))
    plt.show()


def plot2():
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
    diff=nor_balance_case0-nor_balance_case2
    statistic,pval=stats.ks_2samp(diff,np.random.normal(0,np.std(diff),len(diff)))
    if pval>0.05:
        print("Null hypothesis cannot be rejected, p=%.4f,stat=%.4f, The difference is normally distributed"%(pval,statistic))
    else:
        print("Null hypothesis can be rejected, p=%.4f,stat=%.4f, The difference is not normally distributed"%(pval,statistic))
    print("Mean: %.2f, Median: %.2f"%(np.mean(exp_balance_case2),np.median(exp_balance_case2)))
    statistic,pval=stats.skewtest(exp_balance_case2)
    skewness=stats.skew(exp_balance_case2)
    if pval>0.05:
        print("Null hypothesis cannot be rejected, p=%.4f,stat=%.4f, Same skewness as a normal distribution"%(pval,statistic))
    else:
        print("Null hypothesis can be rejected, p=%.4f,stat=%.4f, different skewness as a normal distribution with skewness %f"%(pval,statistic,skewness))


    print("Norwegian export case 2: %.4f±%.4f"%(np.mean(exp_balance_case2),np.std(exp_balance_case2)))
    print("Difference in surplus beetween Case 2 and Case 0: %.2f±%.2f"%(np.mean((nor_balance_case2-nor_balance_case0)),np.std((nor_balance_case2-nor_balance_case0))))

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right")
    plt.tight_layout()

    if savefile:
        plt.savefig("../graphs/case0_case2_%d.pdf"%(start_year))
    plt.show()
    maxfit=1000
    plt.hist(diff,alpha=0.1,color="red",label="1")
    plt.hist(np.random.normal(0,np.std(diff),len(diff)),alpha=0.1,color="blue",label="2")
    #plt.scatter(nor_balance_case0[:maxfit],nor_balance_case2[:maxfit])
    #m,b = np.polyfit(exp_balance_case2[:maxfit], CO2_hist_case0[:maxfit]-CO2_hist_case2[:maxfit], 1)
    #def f(x):
    #    return m*x+b
    #plt.plot(exp_balance_case2[:maxfit],f(exp_balance_case2[:maxfit]),color="white")
    plt.xlabel("Norwegian overproduction case 0")
    plt.ylabel(r"Norwegian overproduction case 2")
    plt.legend()
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case0_case2_confirm_%d.pdf"%(start_year))
    plt.show()

def plot31():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")
    #ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (g)",color="red",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (b)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case1,label="Case 1",color="blue",ax=ax2)
    #sns.kdeplot(CO2_hist_case0,label="Case 0",color="green",ax=ax2)

    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="lower left")
    ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

    plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
    sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, case 3-1",color="red",ax=ax1)
    sns.kdeplot(nor_balance_case1,label="NO el. surplus, case 1",color="blue",ax=ax1)
    sns.kdeplot(nor_balance_case0,label="NO el. surplus, case 0",color="green",ax=ax1)
    ax1.hist(nor_balance_case0,bins=20,density=True,alpha=0.1,color="green")

    ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
    ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="blue")
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="lower left")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case1_case3_%d.pdf"%(start_year))
    print("CO2 Reduction with Case 3-1 (bad) compared to Case 1: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case1)),np.std((CO2_bad_hist_case3_1-CO2_hist_case1))))
    print("CO2 Reduction with Case 3-1 (good) compared to Case 1:  %.2f±%.2f"%(np.mean((CO2_hist_case3_1-CO2_hist_case1)),np.std((CO2_hist_case3_1-CO2_hist_case1))))
    print("Load Reduction with Case 3-1 compared to Case 1:  %.2f±%.2f"%(np.mean((nor_balance_case3_1-nor_balance_case1)),np.std((nor_balance_case3_1-nor_balance_case1))))

    plt.show()


def plot32():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case3_2,density=True,alpha=0.1,bins=20,color="blue")
    ax2.hist(CO2_bad_hist_case3_2,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (g)",color="red",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (b)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3-2 (g)",color="blue",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3-2 (b)",color="green",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="lower left")
    ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

    plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
    sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, case 3-1",color="red",ax=ax1)
    ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
    sns.kdeplot(nor_balance_case3_2,label="NO el. surplus, case 3-2",color="blue",ax=ax1)
    ax1.hist(nor_balance_case3_2,density=True,bins=20,alpha=0.1,color="blue")

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="lower left")
    #ax1.set_xlim(-5,20)
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case1_case3-2_%d.pdf"%(start_year))
    print("CO2 Reduction with Case 3-2 (bad) compared to Case 3-1 (bad):  %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1)),np.std((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1))))
    print("CO2 Reduction with Case 3-2 (good) compared to Case 3-1 (good):  %.2f±%.2f"%(np.mean((CO2_hist_case3_2-CO2_hist_case3_1)),np.std((CO2_hist_case3_2-CO2_hist_case3_1))))
    print("Load Reduction with Case 3-2 compared to Case 3-1: %.2f±%.2f"%(np.mean((nor_balance_case3_2-nor_balance_case3_1)),np.std((nor_balance_case3_2-nor_balance_case3_1))))

    plt.show()

def plot33():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case3_3,density=True,alpha=0.1,bins=20,color="blue")
    ax2.hist(CO2_bad_hist_case3_3,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (g)",color="red",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Case 3-1 (b)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case3_3,x=r"Million tons CO$_2$",label="Case 3-3 (g)",color="blue",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_3,x=r"Million tons CO$_2$",label="Case 3-3 (b)",color="green",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

    plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
    sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, case 3-1",color="red",ax=ax1)
    ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
    sns.kdeplot(nor_balance_case3_3,label="NO el. surplus, case 3-3",color="blue",ax=ax1)
    ax1.hist(nor_balance_case3_3,density=True,bins=20,alpha=0.1,color="blue")

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left")
    #ax1.set_xlim(-5,20)
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case1_case3-3_%d.pdf"%(start_year))
    print("CO2 Reduction with Case 3-3 (bad) compared to Case 3-1 (bad):  %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_3-CO2_bad_hist_case3_1)),np.std((CO2_bad_hist_case3_3-CO2_bad_hist_case3_1))))
    print("CO2 Reduction with Case 3-3 (good) compared to Case 3-1 (good):  %.2f±%.2f"%(np.mean((CO2_hist_case3_3-CO2_hist_case3_1)),np.std((CO2_hist_case3_3-CO2_hist_case3_1))))
    print("Load Reduction with Case 3-3 compared to Case 3-1: %.2f±%.2f"%(np.mean((nor_balance_case3_3-nor_balance_case3_1)),np.std((nor_balance_case3_3-nor_balance_case3_1))))
    print("Load of Case 3-3: %.2f±%.2f"%(np.mean((nor_balance_case3_3)),np.std((nor_balance_case3_3))))
    plt.show()

    fig, (ax2, ax1,ax3) = plt.subplots(1, 3, sharex=False, sharey=False,figsize=(12,6))
    sns.kdeplot(toGermany_case3_3,x=r"Million tons CO$_2$",label="Sent to Germany",color="red",ax=ax2)
    sns.kdeplot(toPlatforms_case3_3,x=r"Million tons CO$_2$",label="Sent to Platforms",color="orange",ax=ax2)
    ax2.set_xlabel("TWh")
    ax2.legend()
    ax1.hist2d(toGermany_case3_3,toPlatforms_case3_3,bins=50,cmap="PuRd")
    ax1.set_xlabel("Sent to Germany")
    ax1.set_ylabel("Sent to Platforms")
    m,b = np.polyfit(toGermany_case3_3, toPlatforms_case3_3, 1)
    print("Slope: %f"%(m))
    def f(x):
        return m*x+b
    ax1.plot(toGermany_case3_3,f(toGermany_case3_3),color="grey",label="linear fit")
    percentys=toGermany_case3_3/(toPlatforms_case3_3+toGermany_case3_3)*100
    sns.kdeplot(percentys,x=r"Million tons CO$_2$",color="orange",ax=ax3)
    ax3.set_xlim([(100-np.max(percentys)),np.max(percentys)])
    ax3.set_xlabel('% sent to Germany')
    if savefile:
        plt.savefig("../graphs/case33_afteranalysis%d.pdf"%(start_year))
    plt.tight_layout()
    plt.show()


def plot4():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case4,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case4,density=True,alpha=0.1,bins=20,color="green")
    ax2.hist(CO2_hist_case0_nowind,density=True,alpha=0.1,bins=20,color="blue")
    sns.kdeplot(CO2_bad_hist_case4,x=r"Million tons CO$_2 (bad)$",label="Case 4 (bad)",color="red",ax=ax2)
    sns.kdeplot(CO2_hist_case4,x=r"Million tons CO$_2 (good)$",label="Case 4 (good)",color="green",ax=ax2)
    sns.kdeplot(CO2_hist_case0_nowind,label="Case 0 (no wind)",color="blue",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper right")
    ax2.set_title(r"CO$_2$, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))

    plt.title("Electricity, n=%d, year=%d, years=%d"%(num_simulations,start_year,num_years))
    sns.kdeplot(nor_balance_case4,label="NO el. surplus, case 4",color="red",ax=ax1)
    sns.kdeplot(nor_balance_case0_nowind,label="NO el. surplus, case 0 (no wind) ",color="green",ax=ax1)
    ax1.hist(nor_balance_case4,density=True,bins=20,alpha=0.1,color="red")
    ax1.hist(nor_balance_case0_nowind,bins=20,density=True,alpha=0.1,color="green")
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/case4_%d.pdf"%(start_year))
    print("CO2 Reduction with Case 4 (bad) compared to Case 0:  %.2f±%.2f"%(np.mean((CO2_bad_hist_case4-CO2_hist_case0_nowind)),np.std((CO2_bad_hist_case4-CO2_hist_case0_nowind))))
    print("CO2 Reduction with Case 4 (good) compared to Case 0:  %.2f±%.2f"%(np.mean((CO2_hist_case4-CO2_hist_case0_nowind)),np.std((CO2_hist_case4-CO2_hist_case0_nowind))))
    plt.show()

boxwhisker()
plotstuff()
plot1()
plot2()
plot31()
plot32()
plot33()
plot4()
