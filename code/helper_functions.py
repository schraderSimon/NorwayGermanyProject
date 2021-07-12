import numpy as np
def coefs_to_function(trend_coef,season_coef,period=52):
    trendfunc=np.poly1d(trend_coef)
    seasonfunc=np.poly1d(season_coef)
    def seasonfunc_periodic(t):
        return seasonfunc((t-1)%period)
    return (lambda t:seasonfunc_periodic(t*period/52)+trendfunc(t*period/52))
def fourier_coefs_to_function(trend_coef,fourier_coefs,na,nb,period=52):
    trendfunc=np.poly1d(trend_coef)
    seasonfunc=make_fourier(na,nb,period/2)
    return (lambda t:seasonfunc(t*period/52,*fourier_coefs)+trendfunc(t*period/52))
def histogram_to_arrays(input):
    number_histograms=int(len(input)/10)
    count,pos=np.histogram(input,bins=number_histograms,density=True)
    return count, pos
def make_fourier(na, nb,p):
    def fourier(x, *a):
        ret = 0.0
        for deg in range(0, na):
            ret += a[deg] * np.cos((deg+1) * np.pi / p * (x-a[-1]))
        for deg in range(na, na+nb):
            ret += a[deg] * np.sin((deg+1) * np.pi / p * (x-a[-1]))
        return ret
    return fourier
