import numpy as np
def coefs_to_function(trend_coef,season_coef,period=52):
    trendfunc=np.poly1d(trend_coef)
    seasonfunc=np.poly1d(season_coef)
    def seasonfunc_periodic(t):
        return seasonfunc((t-1)%period)
    return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))
def histogram_to_arrays(input):
    number_histograms=int(len(input)/10)
    count,pos=np.histogram(input,bins=number_histograms,density=True)
    return count, pos
