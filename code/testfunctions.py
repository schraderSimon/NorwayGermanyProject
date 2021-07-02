from cases import *
import pandas as pd
import scipy.io
def compare_cases():
    coefs=scipy.io.loadmat("../data/timeseries.mat")
    trend_coefs=pd.read_csv("../data/trends.csv")
    season_coefs=pd.read_csv("../data/season.csv")
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
