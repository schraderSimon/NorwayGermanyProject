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
import matplotlib
import pingouin

cmap = matplotlib.cm.get_cmap('PuRd')
background=cmap(0)
plt.rcParams.update({'font.size': 12, 'legend.labelspacing':0.2})
order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
periods=[52,52,52,52,13,52]
colors=["cyan","black","green","red","blue","orange"]
coefs=scipy.io.loadmat("../data/timeseries.mat")
trend_coefs=pd.read_csv("../data/trends.csv")
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
    if sys.argv[4]!="False":
        savefile=True
    else:
        savefile=False
    seed=int(sys.argv[5])
    type=sys.argv[6]
except IndexError:
    start_year=2020
    num_years=1
    num_simulations=10000
    savefile=False
    seed=0
    type="None"
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
    if start_year==2020:
        cutoff=5
    elif start_year==2022:
        cutoff=20
    else:
        cutoff=10
    over_10_rate=len(german_wind_surplus[np.where(german_wind_surplus>cutoff)])/num_simulations
    print("Wind overproduction: %.3f±%.3f"%(np.mean(german_wind_surplus),np.std(german_wind_surplus)))
    print("Wind send to Norway: %.3f±%.3f"%(np.mean(german_wind_toNorway),np.std(german_wind_toNorway)))
    print("Outlayer rate (above %.f TWh): %f"%(cutoff,over_10_rate))
    outfile=open("../tables/%s_wind_overproduction_%d.csv"%(type,start_year),"w")

    outfile.write("Wind overproduction, %.3f±%.3f\n"%(np.mean(german_wind_surplus),np.std(german_wind_surplus)))
    outfile.write("Wind send to Norway: %.3f±%.3f\n"%(np.mean(german_wind_toNorway),np.std(german_wind_toNorway)))
    outfile.write("Outlayer rate (in plot)(above %.f TWh): %f"%(cutoff,over_10_rate))
    outfile.close()
    plt.title("Wind, n=%d, year=%d"%(num_simulations,start_year))
    kde = KernelDensity(kernel='exponential', bandwidth=0.05).fit(german_wind_surplus.reshape(-1,1))
    dens=np.exp(kde.score_samples(np.linspace(0,cutoff,1000).reshape(-1,1)))
    #plt.plot(np.linspace(0,cutoff,1000),dens,color="red",alpha=0.5)
    sns.kdeplot(german_wind_surplus,color="red",bw_adjust=0.4,alpha=0.5)

    plt.hist(german_wind_surplus,bins=200,density=True,range=(0,cutoff*2),label="Total wind overproduction",color="red",alpha=0.5)
    bins=int(200/cutoff*np.max(german_wind_toNorway))
    plt.hist(german_wind_toNorway,bins=bins,density=True,label="Exportable wind overproduction",color="blue",alpha=0.5)
    sns.kdeplot(german_wind_toNorway,color="blue",bw_adjust=0.4,alpha=0.5)
    plt.xlabel(r"TWh ")
    plt.xlim(0,cutoff)
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_wind_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()

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
def plotProductionDistr():
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, sharey=True,figsize=(12,6))
    data=[]
    for i in range(6):
        data.append(case0_data_2020[order[i]].to_numpy())
    data=np.array(data).T
    data_pandas=pd.DataFrame(data)
    data_pandas.columns=order
    print(data_pandas)
    for i in range(6):
        sns.kdeplot(data[:,i],label=order[i],color=colors[i],ax=ax2)
    ax2.set_title("2020")
    ax2.legend()
    ax2.set_xlabel("TWh")
    data=[]
    for i in range(6):
        data.append(case0_data_2022[order[i]].to_numpy())
    data=np.array(data).T
    data_pandas=pd.DataFrame(data)
    data_pandas.columns=order
    print(data_pandas)

    for i in range(6):
        sns.kdeplot(data[:,i],label=order[i],color=colors[i],ax=ax1)
    ax1.set_title("2022")
    ax1.legend()
    ax1.set_xlabel("TWh")
    plt.tight_layout()
    plt.savefig("../graphs/%s_correlations_production.pdf"%type)
    plt.cla()#plt.show()

def plotImportExportDist():
    plt.figure(figsize=(8,6))
    plt.hist(num_weeks_NOtoDE,density=True,alpha=0.1,bins=30,color="red")
    plt.hist(num_weeks_DEtoNO,density=True,alpha=0.1,bins=20,color="blue")
    sns.kdeplot(num_weeks_NOtoDE,color="red",bw_adjust=2,label="NO to DE, no delay")
    sns.kdeplot(num_weeks_DEtoNO,color="blue",bw_adjust=2,label="DE to NO, no delay")
    plt.hist(num_weeks_NOtoDE_delay1,density=True,alpha=0.1,bins=28,color="green")
    plt.hist(num_weeks_DEtoNO_delay1,density=True,alpha=0.1,bins=20,color="orange")
    sns.kdeplot(num_weeks_NOtoDE_delay1,color="green",bw_adjust=2,label="NO to DE, delay")
    sns.kdeplot(num_weeks_DEtoNO_delay1,color="orange",bw_adjust=2,label="DE to NO, delay")
    plt.xlabel("Number of weeks")
    plt.ylabel("Probability")
    plt.legend()
    plt.title(r"weeks exporting, n=%d, year=%d"%(num_simulations,start_year))

    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_Num_weeks_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()
def plotMAIN():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))

    ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="red")#,ax=ax2)
    ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")#,ax=ax2)

    sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="baseline",color="red",ax=ax2)
    sns.kdeplot(CO2_hist_case1,label="main",color="blue",ax=ax2)
    print("CO2:")
    print("Baseline: %.2f±%.2f"%(np.mean(CO2_hist_case0),np.std(CO2_hist_case0)))
    print("Main: %.2f±%.2f"%(np.mean(CO2_hist_case1),np.std(CO2_hist_case1)))
    print("Reduction beetween main and baseline: %.2f±%.2f"%(np.mean(CO2_hist_case1-CO2_hist_case0),np.std(CO2_hist_case1-CO2_hist_case0)))
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    ax1.set_title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case0,label="NO el. surplus, baseline",color="red",ax=ax1)
    sns.kdeplot(exp_balance_case1,label="Norwegian net el. Export, main",color="blue",ax=ax1)
    sns.kdeplot(nor_balance_case1,label="NO el. surplus, main",color="cyan",ax=ax1)
    ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")#,ax=ax1)
    ax1.hist(exp_balance_case1,bins=20,density=True,alpha=0.1,color="blue")#,ax=ax1)
    ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="cyan")#,ax=ax1)
    print("Norwegian surplus baseline: %.4f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
    print("Norwegian surplus main: %.4f±%.4f"%(np.mean(nor_balance_case1),np.std(nor_balance_case1)))
    print("Norwegian export main: %.4f±%.4f"%(np.mean(exp_balance_case1),np.std(exp_balance_case1)))
    print("Difference in surplus beetween baseline and main: %.2f±%.2f"%(np.mean((nor_balance_case1-nor_balance_case0)),np.std((nor_balance_case1-nor_balance_case0))))
    outfile=open("../tables/%s_maincase_%d.csv"%(type,start_year),"w")
    outfile.write("Main case emissions, %.3f±%.3f\n"%(np.mean(CO2_hist_case1),np.std(CO2_hist_case1)))
    outfile.write("Baseline emissions: %.3f±%.3f\n"%(np.mean(CO2_hist_case0),np.std(CO2_hist_case0)))
    outfile.write("Main case emission reduction, %.3f±%.3f\n"%(np.mean(CO2_hist_case1-CO2_hist_case0),np.std(CO2_hist_case1-CO2_hist_case0)))
    outfile.write("Main case net elecricity export, %.3f±%.3f\n"%(np.mean(exp_balance_case1),np.std(exp_balance_case1)))
    outfile.write("Main case electricity surplus, %.3f±%.3f\n"%(np.mean(nor_balance_case1),np.std(nor_balance_case1)))
    outfile.write("Baseline case electricity surplus, %.3f±%.3f\n"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
    outfile.close()
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case0_case1_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()
    maxfit=10000
    xaxis=np.linspace(np.min(nor_balance_case0),np.max(nor_balance_case0),1000)
    fig, (ax3,ax2, ax1) = plt.subplots(1, 3, sharex=False, sharey=False,figsize=(12,6))
    ax3.hist2d(nor_balance_case0,exp_balance_case1,bins=50,cmap="PuRd")
    ax3.set_xlabel("Baseline surplus (TWh)")
    ax3.set_ylabel("Norwegian net exports (TWh)")
    m,b = np.polyfit(nor_balance_case0, exp_balance_case1, 1)
    def f(x):
        return m*x+b
    ax3.plot(xaxis,f(xaxis),color="grey",label="linear fit")
    ax3.legend()
    ax2.hist2d(nor_balance_case0,CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit],bins=50,cmap="PuRd")
    ax2.set_facecolor(background)
    m,b = np.polyfit(nor_balance_case0, CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit], 1)
    print("Slope: %f g/kWh, intercept: %f g"%(m*1000,b*1000))
    def f(x):
        return m*x+b
    ax2.set_xlim([np.min(nor_balance_case0),np.max(nor_balance_case0)])
    ax2.set_ylim([0,np.max(CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit])])

    ax2.plot(xaxis,f(xaxis),color="grey",label="linear fit")
    ax2.set_xlabel("Baseline surplus (TWh)")
    ax2.set_ylabel(r"Million tons CO$_2$ saved")
    ax2.legend()
    maxnum=np.max(num_weeks_DEtoNO[:maxfit])
    minnum=np.min(num_weeks_DEtoNO[:maxfit])
    ax1.hist2d(num_weeks_DEtoNO[:maxfit],CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit],bins=(int(maxnum-minnum),20),cmap="PuRd")
    ax1.set_facecolor(background)

    m,b = np.polyfit(num_weeks_DEtoNO[:maxfit], CO2_hist_case0[:maxfit]-CO2_hist_case1[:maxfit], 1)
    def f2(x):
        return m*x+b
    ax1.plot(num_weeks_DEtoNO[:maxfit],f(num_weeks_DEtoNO[:maxfit]),color="grey",label="linear fit")
    ax1.legend()
    ax1.set_xlabel(r"Number of weeks DE$\rightarrow$NO")
    ax1.set_ylabel(r"Million tons CO$_2$ saved")
    plt.tight_layout()

    if savefile:
        plt.savefig("../graphs/%s_case0_case1_confirm_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()

def plot2():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))

    ax2.hist(CO2_hist_case2,density=True,alpha=0.1,bins=20,color="green")
    sns.kdeplot(CO2_hist_case2,x=r"Million tons CO$_2$",label="Case 2",color="green",ax=ax2)
    ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="blue")
    sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Baseline",color="blue",ax=ax2)

    print("Reduction beetween Case 2 and baseline: %.4f±%.4f"%(np.mean(CO2_hist_case0-CO2_hist_case2),np.std(CO2_hist_case0-CO2_hist_case2)))
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))



    ax1.set_title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case0,label="NO el. surplus, baseline",color="red",ax=ax1)
    sns.kdeplot(exp_balance_case2,label="Norwegian net el. Export, case 2",color="blue",ax=ax1)
    sns.kdeplot(nor_balance_case2,label="NO el. surplus, case 2",color="green",ax=ax1)
    ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")
    ax1.hist(exp_balance_case2,bins=20,density=True,alpha=0.1,color="blue")
    ax1.hist(nor_balance_case2,bins=20,density=True,alpha=0.1,color="green")
    print("Norwegian surplus baseline: %.10f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
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
    print("Difference in surplus beetween Case 2 and baseline: %.2f±%.2f"%(np.mean((nor_balance_case2-nor_balance_case0)),np.std((nor_balance_case2-nor_balance_case0))))

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right")
    plt.tight_layout()

    if savefile:
        plt.savefig("../graphs/%s_case0_case2_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()
    maxfit=1000
    plt.hist(diff,alpha=0.1,color="red",label="1")
    plt.hist(np.random.normal(0,np.std(diff),len(diff)),alpha=0.1,color="blue",label="2")
    #plt.scatter(nor_balance_case0[:maxfit],nor_balance_case2[:maxfit])
    #m,b = np.polyfit(exp_balance_case2[:maxfit], CO2_hist_case0[:maxfit]-CO2_hist_case2[:maxfit], 1)
    #def f(x):
    #    return m*x+b
    #plt.plot(exp_balance_case2[:maxfit],f(exp_balance_case2[:maxfit]),color="white")
    plt.xlabel("Norwegian overproduction baseline")
    plt.ylabel(r"Norwegian overproduction case 2")
    plt.legend()
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case0_case2_confirm_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()

def plotPLATFORM():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")
    #ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Platform (h)",color="red",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Platform (l)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case1,label="Main",color="blue",ax=ax2)
    #sns.kdeplot(CO2_hist_case0,label="",color="green",ax=ax2)

    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="lower left")
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    plt.title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, Platform",color="red",ax=ax1)
    sns.kdeplot(nor_balance_case1,label="NO el. surplus, main",color="blue",ax=ax1)
    sns.kdeplot(nor_balance_case0,label="NO el. surplus, baseline",color="green",ax=ax1)
    ax1.hist(nor_balance_case0,bins=20,density=True,alpha=0.1,color="green")

    ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
    ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="blue")
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="lower left")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case1_case3_%d.pdf"%(type,start_year))
    plt.cla()#plt.show()
    outfile=open("../tables/%s_platformcase_%d.csv"%(type,start_year),"w")
    outfile.write("High platform case emissions, %.3f±%.3f\n"%(np.mean(CO2_hist_case3_1),np.std(CO2_hist_case3_1)))
    outfile.write("Low platform case emissions, %.3f±%.3f\n"%(np.mean(CO2_bad_hist_case3_1),np.std(CO2_bad_hist_case3_1)))
    outfile.write("High platform case emission reduction, %.3f±%.3f\n"%(np.mean((CO2_hist_case3_1-CO2_hist_case0)),np.std((CO2_hist_case3_1-CO2_hist_case0))))
    outfile.write("Low platform case emission reduction, %.3f±%.3f\n"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case0)),np.std((CO2_bad_hist_case3_1-CO2_hist_case0))))
    outfile.write("Platform case electricity surplus in NO, %.3f±%.3f\n"%(np.mean(nor_balance_case3_1),np.std(nor_balance_case3_1)))
    outfile.write("Low Platform emission difference compared to main, %.3f±%.3f\n"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case1)),np.std((CO2_bad_hist_case3_1-CO2_hist_case1))))
    outfile.write("High Platform emission difference compared to main, %.3f±%.3f\n"%(np.mean((CO2_hist_case3_1-CO2_hist_case1)),np.std((CO2_hist_case3_1-CO2_hist_case1))))
    outfile.write("Platform case elctricity surplus difference compared to main, %.3f±%.3f\n"%(np.mean((nor_balance_case3_1-nor_balance_case1)),np.std((nor_balance_case3_1-nor_balance_case1))))
    outfile.close()
    print("CO2 Reduction with Platform (low) compared to Main: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case1)),np.std((CO2_bad_hist_case3_1-CO2_hist_case1))))
    print("CO2 Reduction with Platform (high) compared to Main:  %.2f±%.2f"%(np.mean((CO2_hist_case3_1-CO2_hist_case1)),np.std((CO2_hist_case3_1-CO2_hist_case1))))
    print("Load Reduction with Platform compared to Main:  %.2f±%.2f"%(np.mean((nor_balance_case3_1-nor_balance_case1)),np.std((nor_balance_case3_1-nor_balance_case1))))
    print("CO2 Reduction with Platform (low) compared to Baseline: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_1-CO2_hist_case0)),np.std((CO2_bad_hist_case3_1-CO2_hist_case0))))
    print("CO2 Reduction with Platform (high) compared to Baseline:  %.2f±%.2f"%(np.mean((CO2_hist_case3_1-CO2_hist_case0)),np.std((CO2_hist_case3_1-CO2_hist_case0))))
    print("Load Reduction with Platform compared to Main:  %.2f±%.2f"%(np.mean((nor_balance_case3_1-nor_balance_case1)),np.std((nor_balance_case3_1-nor_balance_case1))))
    print("Pltform low: %.2f±%.2f"%(np.mean(CO2_bad_hist_case3_1),np.std(CO2_bad_hist_case3_1)))
    print("Pltform high: %.2f±%.2f"%(np.mean(CO2_hist_case3_1),np.std(CO2_hist_case3_1)))
    print("Norwegian surplus Platform: %.4f±%.4f"%(np.mean(nor_balance_case3_1),np.std(nor_balance_case3_1)))


    print("Paired sample T-test that the main case has a LOWER mean than the platform scenario")

    print("High reduction scenario")
    test=pingouin.wilcoxon(CO2_hist_case1[:],CO2_hist_case3_1[:],alternative="less")
    #test=pingouin.ttest(CO2_hist_case1,CO2_hist_case3_1,paired=True,alternative="less")
    print("P-val: %e"%(test["p-val"]))
    print("Interval:")
    #print(test["CI95%"])

    print("Low reduction scenario")
    test=pingouin.wilcoxon(CO2_hist_case1[:],CO2_bad_hist_case3_1[:],alternative="less")
    #test=pingouin.ttest(CO2_hist_case1[:100],CO2_bad_hist_case3_1[:100],paired=True,alternative="less")
    #print(test["CI95%"])
    print("P-val: %e"%(test["p-val"]))
    print("Interval:")
    sns.kdeplot(CO2_hist_case1-CO2_bad_hist_case3_1,label="bad")
    sns.kdeplot(CO2_hist_case1-CO2_hist_case3_1,label="good")
    plt.legend()
    plt.cla()#plt.show()


    print("Paired sample T-test that the main case has a HIGHER mean than the platform scenario")

    print("High reduction scenario")
    test=pingouin.wilcoxon(CO2_hist_case1,CO2_hist_case3_1,alternative="greater")
    test=pingouin.ttest(CO2_hist_case1,CO2_hist_case3_1,paired=True,alternative="greater")
    print("P-val: %e"%(test["p-val"]))
    print("Interval:")
    #print(test["CI95%"])

    print("Low reduction scenario")
    test=pingouin.wilcoxon(CO2_hist_case1,CO2_bad_hist_case3_1,alternative="greater")
    test=pingouin.ttest(CO2_hist_case1,CO2_bad_hist_case3_1,paired=True,alternative="greater")
    print("P-val: %e"%(test["p-val"]))
    print("Interval:")
    #print(test["CI95%"])



def plot32():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case3_2,density=True,alpha=0.1,bins=20,color="blue")
    ax2.hist(CO2_bad_hist_case3_2,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case3_1,x=r"Million tons CO$_2$",label="Platform (h)",color="red",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Platform (l)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3-2 (h)",color="blue",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_2,x=r"Million tons CO$_2$",label="Case 3-2 (l)",color="green",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="lower left")
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    plt.title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case3_1,label="NO el. surplus, Platform",color="red",ax=ax1)
    ax1.hist(nor_balance_case3_1,density=True,bins=20,alpha=0.1,color="red")
    sns.kdeplot(nor_balance_case3_2,label="NO el. surplus, case 3-2",color="blue",ax=ax1)
    ax1.hist(nor_balance_case3_2,density=True,bins=20,alpha=0.1,color="blue")

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="lower left")
    #ax1.set_xlim(-5,20)
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case1_case3-2_%d.pdf"%(type,start_year))
    print("CO2 Reduction with Case 3-2 (low) compared to Platform (low):  %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1)),np.std((CO2_bad_hist_case3_2-CO2_bad_hist_case3_1))))
    print("CO2 Reduction with Case 3-2 (high) compared to Platform (high):  %.2f±%.2f"%(np.mean((CO2_hist_case3_2-CO2_hist_case3_1)),np.std((CO2_hist_case3_2-CO2_hist_case3_1))))
    print("Load Reduction with Case 3-2 compared to Platform: %.2f±%.2f"%(np.mean((nor_balance_case3_2-nor_balance_case3_1)),np.std((nor_balance_case3_2-nor_balance_case3_1))))

    plt.cla()#plt.show()

def plotDEPLATFORM():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    #ax2.hist(CO2_hist_case3_1,density=True,alpha=0.1,bins=20,color="red")
    #ax2.hist(CO2_bad_hist_case3_1,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")
    ax2.hist(CO2_hist_case3_3,density=True,alpha=0.1,bins=20,color="orange")
    ax2.hist(CO2_bad_hist_case3_3,density=True,alpha=0.1,bins=20,color="green")

    sns.kdeplot(CO2_hist_case1,x=r"Million tons CO$_2$",label="Main",color="blue",ax=ax2)
    #sns.kdeplot(CO2_bad_hist_case3_1,x=r"Million tons CO$_2$",label="Platform (l)",color="orange",ax=ax2)
    sns.kdeplot(CO2_hist_case3_3,x=r"Million tons CO$_2$",label="Pltf+DE (h)",color="orange",ax=ax2)
    sns.kdeplot(CO2_bad_hist_case3_3,x=r"Million tons CO$_2$",label="Pltf+DE (l)",color="green",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    plt.title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case1,label="NO el. surplus, Main",color="blue",ax=ax1)
    ax1.hist(nor_balance_case1,density=True,bins=20,alpha=0.1,color="blue")
    sns.kdeplot(nor_balance_case3_3,label="NO el. surplus, Pltf+DE",color="red",ax=ax1)
    ax1.hist(nor_balance_case3_3,density=True,bins=20,alpha=0.1,color="red")

    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left")
    #ax1.set_xlim(-5,20)
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case1_case3-3_%d.pdf"%(type,start_year))
    outfile=open("../tables/%s_DE_and_platformcase_%d.csv"%(type,start_year),"w")
    outfile.write("High DE+platform case emissions, %.3f±%.3f\n"%(np.mean(CO2_hist_case3_3),np.std(CO2_hist_case3_3)))
    outfile.write("Low DE+platform case emissions, %.3f±%.3f\n"%(np.mean(CO2_bad_hist_case3_3),np.std(CO2_bad_hist_case3_3)))
    outfile.write("High DE+platform case emission reduction, %.3f±%.3f\n"%(np.mean((CO2_hist_case3_3-CO2_hist_case0)),np.std((CO2_hist_case3_3-CO2_hist_case0))))
    outfile.write("Low DE+platform case emission reduction, %.3f±%.3f\n"%(np.mean((CO2_bad_hist_case3_3-CO2_hist_case0)),np.std((CO2_bad_hist_case3_3-CO2_hist_case0))))
    outfile.write("DE+Platform case electricity surplus in NO, %.3f±%.3f\n"%(np.mean((nor_balance_case3_3)),np.std((nor_balance_case3_3))))
    outfile.write("Low DE+Platform emission difference compared to main, %.3f±%.3f\n"%(np.mean((CO2_bad_hist_case3_3-CO2_hist_case1)),np.std((CO2_bad_hist_case3_3-CO2_hist_case1))))
    outfile.write("High DE+Platform emission difference compared to main, %.3f±%.3f\n"%(np.mean((CO2_hist_case3_3-CO2_hist_case1)),np.std((CO2_hist_case3_3-CO2_hist_case1))))
    outfile.write("Platform case elctricity surplus difference compared to main, %.3f±%.3f\n"%(np.mean((nor_balance_case3_3-nor_balance_case1)),np.std((nor_balance_case3_3-nor_balance_case1))))
    outfile.close()
    print("CO2 Reduction with Platform+DE (low) compared to Main: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_3-CO2_hist_case1)),np.std((CO2_bad_hist_case3_3-CO2_hist_case1))))
    print("CO2 Reduction with Platform+DE (high) compared to Main:  %.2f±%.2f"%(np.mean((CO2_hist_case3_3-CO2_hist_case1)),np.std((CO2_hist_case3_3-CO2_hist_case1))))
    print("Load Reduction with Platform+DE compared to Main:  %.2f±%.2f"%(np.mean((nor_balance_case3_3-nor_balance_case1)),np.std((nor_balance_case3_3-nor_balance_case1))))
    print("CO2 Reduction with Platform+DE (low) compared to Baseline: %.2f±%.2f"%(np.mean((CO2_bad_hist_case3_3-CO2_hist_case0)),np.std((CO2_bad_hist_case3_3-CO2_hist_case0))))
    print("CO2 Reduction with Platform+DE (high) compared to Baseline:  %.2f±%.2f"%(np.mean((CO2_hist_case3_3-CO2_hist_case0)),np.std((CO2_hist_case3_3-CO2_hist_case0))))
    print("Load Reduction with Platform+DE compared to Main:  %.2f±%.2f"%(np.mean((nor_balance_case3_3-nor_balance_case1)),np.std((nor_balance_case3_3-nor_balance_case1))))
    print("Pltform+DE low: %.2f±%.2f"%(np.mean(CO2_bad_hist_case3_3),np.std(CO2_bad_hist_case3_3)))
    print("Pltform+DE high: %.2f±%.2f"%(np.mean(CO2_hist_case3_3),np.std(CO2_hist_case3_3)))
    print("Norwegian surplus Platform+DE: %.4f±%.4f"%(np.mean(nor_balance_case3_3),np.std(nor_balance_case3_3)))
    plt.cla()#plt.show()

    fig, (ax2, ax1,ax3) = plt.subplots(1, 3, sharex=False, sharey=False,figsize=(12,6))
    sns.kdeplot(toGermany_case3_3,x=r"Million tons CO$_2$",label="Sent to Germany",color="red",ax=ax2)
    sns.kdeplot(toPlatforms_case3_3,x=r"Million tons CO$_2$",label="Sent to Platforms",color="orange",ax=ax2)
    ax2.set_xlabel("TWh")
    ax2.legend()
    ax1.hist2d(toGermany_case3_3,toPlatforms_case3_3,bins=50,cmap="PuRd")
    ax1.set_facecolor(background)

    ax1.set_xlabel("Sent to Germany")
    ax1.set_ylabel("Sent to Platforms")
    m,b = np.polyfit(toGermany_case3_3, toPlatforms_case3_3, 1)
    print("Slope: %f, Intercept: %f"%(m,b))
    def f(x):
        return m*x+b
    xaxis=np.linspace(0,np.max(toGermany_case3_3),1000)
    ax1.plot(xaxis,f(xaxis),color="grey",label="linear fit")
    ax1.set_xlim(0,np.max(toGermany_case3_3))
    ax1.set_ylim(0,np.max(toPlatforms_case3_3))
    percentys=toGermany_case3_3/(toPlatforms_case3_3+toGermany_case3_3)*100
    sns.kdeplot(percentys,x=r"Million tons CO$_2$",color="orange",ax=ax3)
    ax3.set_xlim([(100-np.max(percentys)),np.max(percentys)])
    ax3.set_xlabel('% sent to Germany')
    if savefile:
        plt.savefig("../graphs/%s_case33_afteranalysis%d.pdf"%(type,start_year))
    plt.tight_layout()
    plt.cla()#plt.show()


def plot4():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))
    ax2.hist(CO2_hist_case4,density=True,alpha=0.1,bins=20,color="red")
    ax2.hist(CO2_bad_hist_case4,density=True,alpha=0.1,bins=20,color="green")
    ax2.hist(CO2_hist_case0_nowind,density=True,alpha=0.1,bins=20,color="blue")
    sns.kdeplot(CO2_bad_hist_case4,x=r"Million tons CO$_2 (low)$",label="Case 4 (low)",color="red",ax=ax2)
    sns.kdeplot(CO2_hist_case4,x=r"Million tons CO$_2 (high)$",label="Case 4 (high)",color="green",ax=ax2)
    sns.kdeplot(CO2_hist_case0_nowind,label="Baseline (no wind)",color="blue",ax=ax2)
    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper right")
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    plt.title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case4,label="NO el. surplus, case 4",color="red",ax=ax1)
    sns.kdeplot(nor_balance_case0_nowind,label="NO el. surplus, baseline (no wind) ",color="green",ax=ax1)
    ax1.hist(nor_balance_case4,density=True,bins=20,alpha=0.1,color="red")
    ax1.hist(nor_balance_case0_nowind,bins=20,density=True,alpha=0.1,color="green")
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case4_%d.pdf"%(type,start_year))
    print("CO2 Reduction with Case 4 (low) compared to baseline:  %.2f±%.2f"%(np.mean((CO2_bad_hist_case4-CO2_hist_case0_nowind)),np.std((CO2_bad_hist_case4-CO2_hist_case0_nowind))))
    print("CO2 Reduction with Case 4 (high) compared to baseline:  %.2f±%.2f"%(np.mean((CO2_hist_case4-CO2_hist_case0_nowind)),np.std((CO2_hist_case4-CO2_hist_case0_nowind))))
    plt.cla()#plt.show()
def plot1_delay1():
    fig, (ax2, ax1) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(12,6))

    ax2.hist(CO2_hist_case0,density=True,alpha=0.1,bins=20,color="red")#,ax=ax2)
    ax2.hist(CO2_hist_case1,density=True,alpha=0.1,bins=20,color="blue")#,ax=ax2)
    ax2.hist(CO2_hist_case1_delay1,density=True,alpha=0.1,bins=20,color="green")#,ax=ax2)

    sns.kdeplot(CO2_hist_case0,x=r"Million tons CO$_2$",label="Baseline",color="red",ax=ax2)
    sns.kdeplot(CO2_hist_case1,label="main (no delay)",color="blue",ax=ax2)
    sns.kdeplot(CO2_hist_case1_delay1,label="main (delay)",color="green",ax=ax2)
    print("CO2:")
    print("baseline: %.2f±%.2f"%(np.mean(CO2_hist_case0),np.std(CO2_hist_case0)))
    print("Main delay: %.2f±%.2f"%(np.mean(CO2_hist_case1_delay1),np.std(CO2_hist_case1_delay1)))
    print("Reduction beetween main delay and baseline: %.2f±%.2f"%(np.mean(CO2_hist_case1_delay1-CO2_hist_case0),np.std(CO2_hist_case1_delay1-CO2_hist_case0)))
    print("Reduction beetween main delay and baseline: %.2f±%.2f"%(np.mean(CO2_hist_case1_delay1-CO2_hist_case1),np.std(CO2_hist_case1_delay1-CO2_hist_case1)))

    ax2.set_xlabel(r"Million Tons CO$_2$")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.set_title(r"CO$_2$, n=%d, year=%d"%(num_simulations,start_year))

    ax1.set_title("Electricity, n=%d, year=%d"%(num_simulations,start_year))
    sns.kdeplot(nor_balance_case0,label="NO el. surplus, baseline",color="red",ax=ax1)
    sns.kdeplot(exp_balance_case1,label="Norwegian net el. Export, main",color="blue",ax=ax1)
    sns.kdeplot(nor_balance_case1,label="NO el. surplus, main",color="cyan",ax=ax1)
    sns.kdeplot(nor_balance_case1_delay1,label="NO el. surplus, main (delay)",color="magenta",ax=ax1)
    ax1.hist(nor_balance_case0,density=True,bins=20,alpha=0.1,color="red")#,ax=ax1)
    ax1.hist(exp_balance_case1,bins=20,density=True,alpha=0.1,color="blue")#,ax=ax1)
    ax1.hist(nor_balance_case1,bins=20,density=True,alpha=0.1,color="cyan")#,ax=ax1)
    ax1.hist(nor_balance_case1_delay1,bins=20,density=True,alpha=0.1,color="magenta")
    print("Norwegian baseline surplus: %.4f±%.4f"%(np.mean(nor_balance_case0),np.std(nor_balance_case0)))
    print("Norwegian surplus Main delay: %.4f±%.4f"%(np.mean(nor_balance_case1_delay1),np.std(nor_balance_case1_delay1)))
    print("Norwegian export Main: %.4f±%.4f"%(np.mean(exp_balance_case1),np.std(exp_balance_case1)))
    print("Difference in surplus beetween main (delay) and baseline: %.2f±%.2f"%(np.mean((nor_balance_case1_delay1-nor_balance_case0)),np.std((nor_balance_case1_delay1-nor_balance_case0))))
    ax1.set_xlabel("TWh")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    if savefile:
        plt.savefig("../graphs/%s_case0_case1_delay%d.pdf"%(type,start_year))
    plt.cla()#plt.show()
plotwind()
plotProductionDistr()
plotImportExportDist()
plotMAIN()
#plot2() Not used in article
plotPLATFORM()
#plot32() Not used in article
plotDEPLATFORM()
#plot4() Not used in article
