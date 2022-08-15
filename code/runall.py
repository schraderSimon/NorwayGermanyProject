import subprocess
import sys
from os import remove
import glob
def delcases(directory, num_MC, years):
    for year in years:
        files=glob.glob("%s/*_%d_*_%d_*.csv"%(directory,year,num_MC))
        for file in files:
            remove(file)
num_MC=10000 #The number of MC steps n
rngseed=0 #Change to get different data. 0 as starting seed is used in all simulations
years=[2020,2022] #Running different years is possible, but changes need to be made in the plot file or the file naming convention
#Step 1: Run "grab_data" files. They create the mathematical VAR model based on the data and also plot some graphs that (not all...) are in the report.
filenames=["grab_data.py","grab_data_fourier.py"]
for filename in filenames:
    bashCommand="python3 %s %d"%(filename,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
# Step 2: Create data using the approach from the main text.
delcases("../data",num_MC,years)
for year in years:
    bashCommand="python3 run_simulations.py %d 1 %d %d 0 dependent"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
for year in years:
    bashCommand="python3 plot_simulations_small.py %d 1 %d True %d default"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
# Step 3: Create data using the sinusoidal deseasoning (A.2)
delcases("../data",num_MC,years)
for year in years:
    bashCommand="python3 run_simulations.py %d 1 %d %d 1 dependent"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
for year in years:
    bashCommand="python3 plot_simulations_small.py %d 1 %d True %d A2"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
# Step 4: Create data using different water sampling (A.1)
delcases("../data",num_MC,years)
for year in years:
    bashCommand="python3 run_simulations.py %d 1 %d %d 0 independent"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
for year in years:
    bashCommand="python3 plot_simulations_small.py %d 1 %d True %d A1"%(year,num_MC,rngseed)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,shell=False)
    output, error = process.communicate()
