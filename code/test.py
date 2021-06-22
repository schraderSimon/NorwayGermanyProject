from random import gauss
from random import seed
from pandas import Series

from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)
series.plot()
plt.show()
