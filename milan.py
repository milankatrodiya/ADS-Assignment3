# -*- coding: utf-8 -*-
"""
Created on Thu Jan 5 17:03:36 2023

@author: prach
"""


"""
CLUSTERING PART STARTS
"""

# import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet


# function to read file
def readFile(x):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        -------
        ghg_grow : variable for storing csv file

    '''
    ghg_grow = pd.read_csv("ghg-emissions.csv");
    ghg_grow = pd.read_csv(x)
    ghg_grow = ghg_grow.fillna(0.0)
    return ghg_grow
 
# calling readFile function to display dataframe 
ghg_grow = readFile("ghg-emissions.csv")

print("\nGHG-emission Growth: \n", ghg_grow)


# dropping particular columns which are not required to clean data
ghg_grow = ghg_grow.drop(['unit', '1989'], axis=1)

print("\nGHG-emission Growth after dropping columns: \n", ghg_grow)


# transpose dataframe
ghg_grow = pd.DataFrame.transpose(ghg_grow)

print("\nTransposed GHG-emission Growth: \n",ghg_grow)


# populating header with header information
header1 = ghg_grow.iloc[0].values.tolist()

ghg_grow.columns = header1

print("\nGHG-emission Growth Header: \n", ghg_grow)


# remove first two rows from dataframe
ghg_grow = ghg_grow.iloc[2:]

print("\nGHG-emission Growth after selecting particular rows: \n", ghg_grow)


# creating a dataframe for two columns to store original values
pop_ex = ghg_grow[["India","Australia"]].copy()


# extracting maximum and minmum value from new dataframe
max_val = pop_ex.max()

min_val = pop_ex.min()

pop_ex = (pop_ex - min_val) / (max_val - min_val) # operation of min and max

print("\nMin and Max operation on GHG-emission Growth: \n", pop_ex)


# set up clusterer and number of clusters
ncluster = 6

kmeans = cluster.KMeans(n_clusters=ncluster)


# fitting the data where the results are stored in kmeans object
kmeans.fit(pop_ex)

labels = kmeans.labels_ # labels is number of associated clusters


# extracting estimated cluster centres
cen = kmeans.cluster_centers_

print("\nCluster Centres: \n", cen)


# calculate the silhoutte score
print("\nSilhoutte Score: \n",skmet.silhouette_score(pop_ex, labels))


# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:red", "tab:pink", "tab:green", "tab:blue", "tab:brown", \
       "tab:cyan", "tab:orange", "tab:black", "tab:olive", "tab:gray"]

    
# loop over the different labels    
for l in range(ncluster): 
    plt.plot(pop_ex[labels==l]["India"], pop_ex[labels==l]["Australia"], 
             marker="o", markersize=3, color=col[l])    

    
# display cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("India")

plt.ylabel("Australia")

plt.show()    


print("\nCentres: \n", cen)


df_cen = pd.DataFrame(cen, columns=["India", "Australia"])

print(df_cen)

df_cen = df_cen * (max_val - min_val) + max_val

pop_ex = pop_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print("\nDataframe Centre: \n", df_cen)


# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

for l in range(ncluster): # loop over the different labels
    plt.plot(pop_ex[labels==l]["India"], pop_ex[labels==l]["Australia"], "o", markersize=3, color=col[l])
    

# show cluster centres
plt.plot(df_cen["India"], df_cen["Australia"], "dk", markersize=10)

plt.xlabel("India")

plt.ylabel("Australia")

plt.title("GHG-emission Growth(%)")

plt.show()

print("\nCentres: \n", cen)    




# In[ ]:
    
"""
CURVE FIT PART STARTS
"""


# import necessary modules
# import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import errors as err


# function to read file
def readFile(y):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        -------
        air_pop : variable for storing csv file

    '''
    air_pop = pd.read_csv("Air_pollution.csv");
    air_pop = pd.read_csv(y)
    air_pop = air_pop.fillna(0.0)
    return air_pop


# calling readFile function to display dataframe 
air_pop = readFile("Air_pollution.csv")

print("\nAir_pollution: \n", air_pop)


# converting to dataframe
air_pop = pd.DataFrame(air_pop)


# transpose dataframe
air_pop = air_pop.transpose()

print("\nTransposed Air_pollution: \n", air_pop)


# populating header with header information
header2 = air_pop.iloc[0].values.tolist()

air_pop.columns = header2

print("\nAir_pollution Header: \n", air_pop)


# select particular column
air_pop = air_pop["Bahamas, The"]

print("\nAir_pollution after selecting particular column: \n", air_pop)


# rename column
air_pop.columns = ["AIR_P"]

print("\nRenamed Air_pollution: \n", air_pop)


# extracting particular rows
air_pop = air_pop.iloc[5:]

air_pop = air_pop.iloc[:-1]

print("\nAIR_P after selecting particular rows: \n", air_pop)


# resetn index of dataframe
air_pop = air_pop.reset_index()

print("\nAir_pollution reset index: \n", air_pop)


# rename columns
air_pop = air_pop.rename(columns={"index": "Year", "Bahamas, The": "AIR_P"} )

print("\nAir_pollution after renamed columns: \n", air_pop)

print(air_pop.columns)


# plot line graph
air_pop.plot("Year", "AIR_P",     )

plt.legend()

plt.title("Air_pollution")

plt.show()


# curve fit with exponential function
def exponential(s, q0, h):
    '''
        Calculates exponential function with scale factor n0 and growth rate g.
    '''
    s = s - 1960.0
    x = q0 * np.exp(h*s)
    return x


# performing best fit in curve fit
print(type(air_pop["Year"].iloc[1]))

air_pop["Year"] = pd.to_numeric(air_pop["Year"])

print("\nAir_pollution Type: \n", type(air_pop["Year"].iloc[1]))

param, covar = opt.curve_fit(exponential, air_pop["Year"], air_pop["AIR_P"],
p0=(4.978423, 0.03))


# plotting best fit
air_pop["fit"] = exponential(air_pop["Year"], *param)

air_pop.plot("Year", ["AIR_P", "fit"], label=["New AIR_P", "New Fit"])

plt.legend()

plt.title("Air_pollution")

plt.show()


# predict fit for future years
year = np.arange(1960, 2031)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()

plt.plot(air_pop["Year"], air_pop["AIR_P"], label="AIR_P")

plt.plot(year, forecast, label="Forecast")

plt.xlabel("Year")

plt.ylabel("AIR_P")

plt.title("Air_pollution")

plt.legend()    

plt.show()


# err_ranges function
def err_ranges(x, exponential, param, sigma):
    '''
        Calculates the upper and lower limits for the function, parameters and
        sigmas for single value or array x. Functions values are calculated for 
        all combinations of +/- sigma and the minimum and maximum is determined.
        Can be used for all number of parameters and sigmas >=1.
    
        This routine can be used in assignment programs.
    '''
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper




