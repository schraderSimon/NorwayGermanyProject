from cases import *
import pandas as pd
import scipy.io
import sys
def compare_cases():
    coefs=scipy.io.loadmat("../data/timeseries.mat")
    trend_coefs=pd.read_csv("../data/trends.csv")
    season_coefs=pd.read_csv("../data/season.csv")
    num_years=1
    start_year=2020
    num_simulations=100
    seed=0
    params=[coefs,trend_coefs,season_coefs,num_years,start_year,seed,"independent",0,0]
    cases=[case0(*params[:-2]),case1(*params),case2(*params),case3_1(*params),case3_2(*params),case3_3(*params)]
    for case in cases:
        case.sendable_max=0
        try:
            case.platform_restriction=0
        except:
            pass
        case.simulate_n_years(n=num_simulations)
        assert np.all(np.abs(case.get_CO2()-cases[0].get_CO2())/1e9<1e-12)
compare_cases()
def simulate_5_steps_test(): #Test wether the simulation set up works as intended by more explicitely calculating the results than my for loops
    def coefs_to_function(trend_coef,season_coef,period):
        trendfunc=np.poly1d(trend_coef)
        seasonfunc=np.poly1d(season_coef)
        def seasonfunc_periodic(t):
            return seasonfunc((t-1)%period)
        return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))
    coefs=scipy.io.loadmat("../data/timeseries.mat")
    trend_coefs=pd.read_csv("../data/trends.csv")
    season_coefs=pd.read_csv("../data/season.csv")
    sigma_windsun=coefs["windsun_sigma"]
    #print(sigma_windsun)
    num_years=1
    start_year=2020
    num_simulations=100
    seed=0
    params=[coefs,trend_coefs,season_coefs,num_years,start_year,seed,0,0]
    testcase=case0(*params[:-2])
    testcase.simulate_n_years(n=1) #Simulate for one year
    simulation_results=testcase.simulation_results[:,:10]/(24*7)
    rng_wind=np.random.default_rng(0)
    rng_load=np.random.default_rng(1)
    rng_water=np.random.default_rng(2)
    order=["wind NO","wind DE","load NO","load DE","water NO","solar DE"]
    periods=[52,52,52,52,13,52]
    functions=[]
    for i in range(6):
        trend=trend_coefs[order[i]]
        season=season_coefs[order[i]]
        functions.append(coefs_to_function(trend,season,period=periods[i]))

    sigma_windsun=coefs["windsun_sigma"]
    matrix_windsun=coefs["windsun_coefs"][0]
    #print(matrix_windsun) #test correctness
    #print(sigma_windsun) #test correctness
    matrix_load_ar1=coefs["load_coefs"][0]
    matrix_load_ar2=coefs["load_coefs"][1]
    matrix_load_ar3=coefs["load_coefs"][2]
    sigma_water=coefs["water_sigma"]
    sigma_load=coefs["load_sigma"]
    #print(matrix_load_ar1);print(matrix_load_ar2);print(matrix_load_ar3);print(sigma_load) #test correctness
    #print(sigma_water)
    random_numbers_windsun=np.zeros((10,3))
    random_numbers_load=np.zeros((10,2))
    for i in range(10):
        random_numbers_windsun[i]=rng_wind.multivariate_normal(np.zeros(3),sigma_windsun)
        random_numbers_load[i]=rng_load.multivariate_normal(np.zeros(2),sigma_load)
    #print(random_numbers_windsun)
    random_numbers_water=np.zeros(10)
    sigmawater=np.sqrt(sigma_water[1,1]) #standard deviation for water
    sigmaload=np.sqrt(sigma_water[0,0]) #Standard deviation for load
    p=sigma_water[1,0]/(sigmawater*sigmaload) #rho in the covariance matrix
    for i in range(10):
        random_numbers_water[i]=rng_water.normal(sigmawater/sigmaload*p*random_numbers_load[i,0],np.sqrt((1-p**2))*sigmawater)

    results_windsun=np.zeros((10,3))
    results_loads=np.zeros((10,2))
    results_water=np.zeros(10)
    for i in range(10):
        results_windsun[i]=matrix_windsun@results_windsun[i-1]+random_numbers_windsun[i]
        results_loads[i]=matrix_load_ar1@results_loads[i-1]+matrix_load_ar2@results_loads[i-2]+matrix_load_ar3@results_loads[i-3]+random_numbers_load[i]
    results=np.zeros((10,6))
    for i in range(0,4):
        results_water[i]=random_numbers_water[i]
    for i in range(4,8):
        mean=np.mean(results_water[0:4])
        results_water[i]=random_numbers_water[i]+mean*1.189324
    for i in range(8,10):
        mean=np.mean(results_water[4:8])
        mean_prev=np.mean(results_water[0:4])
        results_water[i]=random_numbers_water[i]+mean*1.189324+mean_prev*(-0.484997)
    for i in range(10):
        time=i+(start_year-2017)*52
        results[i,0]=np.exp(functions[0](time)+results_windsun[i,0]) #Norwegian wind
        results[i,2]=np.exp(functions[2](time)+results_loads[i,0]) #Norwegian load
        results[i,1]=np.exp(functions[1](time)+results_windsun[i,1]) #German wind
        results[i,3]=np.exp(functions[3](time)+results_loads[i,1]) #German load
        results[i,4]=np.exp(functions[4](time)+results_water[i]) #Water NO
        results[i,5]=np.exp(functions[5](time)+results_windsun[i,2]) #German sun
    assert np.all(np.abs(results.T-simulation_results)<0.01)
simulate_5_steps_test()
