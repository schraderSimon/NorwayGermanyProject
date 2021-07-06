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
    params=[coefs,trend_coefs,season_coefs,num_years,start_year,seed,0,0]
    cases=[case1(*params),case2(*params),case3_1(*params),case3_2(*params),case3_3(*params)]
    for case in cases:
        case.sendable_max=0
        try:
            case.platform_restriction=0
        except:
            pass
        case.simulate_n_years(n=num_simulations)
        assert np.all(np.abs(case.get_CO2()-cases[0].get_CO2())/1e9<1e-12)
compare_cases()
