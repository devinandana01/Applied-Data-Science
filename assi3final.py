# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
import scipy.optimize as opt
import errors as err

# Reading the file to the dataframe 'dtf'
dtf = pd.read_csv("datasetassi3.csv")
dtf
    
# Taking indicators from data set
a = dtf[dtf['Indicator Name'] == 'Urban population growth (annual %)']
b = dtf[dtf['Indicator Name'] == 'Electric power consumption (kWh per capita)']
c = dtf[dtf['Indicator Name'] == 
'Electricity production from renewable sources,excluding hydroelectric (% of total)']
d = dtf[dtf['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)']

# Cleaning the indicator data 'a'
a = a.drop(["Indicator Code","Indicator Name","Country Code","2021","2020"]
           ,axis=1)
a= a.replace(np.NaN, 0)
x_country = ["Australia", "France", "Germany", "Sweden", "Ireland"]
dt1 = a["Country Name"].isin(x_country)
a= a[dt1]
a

# Transposing data set of indicator 'a'
data_trans = np.transpose(a)
data_trans= data_trans.reset_index()
data_trans= data_trans.rename(columns= data_trans.iloc[0])
data_trans= data_trans.drop(0,axis=0)
data_trans= data_trans.rename(columns={"Country Name":"Year"})
data_trans

# Cleaning missing values using dropna function
data_trans["Year"] = pd.to_numeric(data_trans["Year"])
data_trans["Australia"] = pd.to_numeric(data_trans["Australia"])
data_trans["France"] = pd.to_numeric(data_trans["France"])
data_trans["Germany"] = pd.to_numeric(data_trans["Germany"])
data_trans["Sweden"] = pd.to_numeric(data_trans["Sweden"])
data_trans["Ireland"] = pd.to_numeric(data_trans["Ireland"])
data_trans = data_trans.dropna()

# Cleaning data set of indicator 'b'
b = b.drop(["Indicator Code","Indicator Name","Country Code","2021","2020"]
           ,axis=1)
b= b.replace(np.NaN, 0)
dt2 = b["Country Name"].isin(x_country)
b= b[dt2]
b

# Transposing data set of indicator 'a'
dt_trans = np.transpose(b)
dt_trans = dt_trans.reset_index()
dt_trans = dt_trans.rename(columns= dt_trans.iloc[0])
dt_trans = dt_trans.drop(0,axis=0)
dt_trans = dt_trans.rename(columns={"Country Name":"Year"})
dt_trans

# Cleaning missing values using dropna function
dt_trans["Year"] = pd.to_numeric(dt_trans["Year"])
dt_trans["Australia"] = pd.to_numeric(dt_trans["Australia"])
dt_trans["France"] = pd.to_numeric(dt_trans["France"])
dt_trans["Germany"] = pd.to_numeric(dt_trans["Germany"])
dt_trans["Sweden"] = pd.to_numeric(dt_trans["Sweden"])
dt_trans["Ireland"] = pd.to_numeric(dt_trans["Ireland"])
dt_trans = dt_trans.dropna()

# Taking country for plotting matrix scatter plot
Sw = pd.DataFrame()
Sw["Year"] = data_trans["Year"]
Sw["urban_population_growth"] = data_trans["Sweden"]
Sw["electric_power_consumption"] = dt_trans["Sweden"]
Sw = Sw.iloc[1:60,:]

def data_file(country):
    """
    This function will plot scatter matrix graph of two indicators Urban 
    population growth (annual %) 
     of Sweden from 1960-2021. Graphs are also saved as images to local 
     directory.

    ----------
    country : This is a DataFrame used to plot matrix scatter graph.

    Returns
    -------
    None.

    """
    
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()
    
data_file(Sw)

kmean = cluster.KMeans(n_clusters=2,max_iter=30)
pt = np.array(Sw["urban_population_growth"]).reshape(-1,1)
co = np.array(Sw["electric_power_consumption"]).reshape(-1,1)

# fitting he model
clst = np.concatenate((co,pt),axis=1)
nc=2
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(clst)

# assignining the category
label = kmeans.labels_

# finding the centers for cluster
k_center = kmeans.cluster_centers_
col = ['electric_power_consumption','urban_population_growth']
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(clst,columns=col)

# concat result and labels
rs = pd.concat((result,labels),axis=1)

# plotting the cluster
plt.figure(figsize=(7.0,7.0))
plt.title("Sweden's electric power consumption vs Urban population growth")
plt.scatter(rs["electric_power_consumption"],rs["urban_population_growth"]
            ,c=label,cmap="rainbow")
plt.xlabel("Electric power consumption")
plt.ylabel("Urban population growth")

# plotting centers of clusters
plt.scatter(k_center[:,0],k_center[:,1], marker="*",c="black",s=150)
plt.show()

# Taking country for plotting matrix scatter plot
Gr = pd.DataFrame()
Gr["Year"] = data_trans["Year"]
Gr["urban_population_growth"] = data_trans["Germany"]
Gr["electric_power_consumption"] = dt_trans["Germany"]
Gr = Gr.iloc[1:60,:]

def data_f(country):
    """
    This function will plot scatter matrix graph of two indicators Urban 
    population growth (annual %) 
     of Germany from 1960-2021. Graphs are also saved as images to local
     directory.

    Parameters
    ----------
    country : This is a DataFrame used to plot matrix scatter graph.

    Returns
    -------
    None.

    """
    
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()
    
data_f(Gr)
plt.savefig('scattermatrix2.png', dpi=300)

kmean = cluster.KMeans(n_clusters=2,max_iter=30)
pt = np.array(Gr["urban_population_growth"]).reshape(-1,1)
co = np.array(Gr["electric_power_consumption"]).reshape(-1,1)

# fitting the model
cl = np.concatenate((co,pt),axis=1)
nc=2
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)

# assignining the category
label = kmeans.labels_

# finding the centers for cluster
k_center = kmeans.cluster_centers_
col = ['electric_power_consumption','urban_population_growth']
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)

# concat result and labels
rs = pd.concat((result,labels),axis=1)

# plotting the cluster
plt.figure(figsize=(7.0,7.0))
plt.title("Germany's electric power consumption vs Urban population growth")
plt.scatter(rs["electric_power_consumption"],rs["urban_population_growth"]
            ,c=label,cmap="rainbow")
plt.xlabel("Electric power consumption")
plt.ylabel("Urban population growth")

# plotting centers of clusters
plt.scatter(k_center[:,0],k_center[:,1], marker="*",c="black",s=150)
plt.show()


def curve_one(t, scale, growth):
    """
    This function will plot curve fitting graph of Urban population growth
    (annual %) 
    of France and Australia from 1960-2021. Graphs are also saved as images 
    to local directory.

    Parameters
    ----------
    t : TYPE
        List of values
    scale : TYPE
        Scale of curve
    growth : TYPE
        Growth of the curve

    Returns
    -------
    e : TYPE
        Result

    """
    e = scale * np.exp(growth * (t-1975))
    return e


# Doing curve fit for Urban population growth in France
param, cov = opt.curve_fit(curve_one,data_trans["Year"],data_trans["France"]
                           ,p0=[4e8,0.1])
sigma = np.sqrt(np.diag(cov))

# Error
low,up = err.err_ranges(data_trans["Year"],curve_one,param,sigma)
data_trans["fit_value"] = curve_one(data_trans["Year"], * param)
print(data_trans["fit_value"])

#4.Plotting the Urban population growth  of France
plt.figure(figsize=(8, 5))
plt.title("Urban population growth (annual %) - France")
plt.plot(data_trans["Year"],data_trans["France"],label="data")
plt.plot(data_trans["Year"],data_trans["fit_value"],c="yellow",label="fit")
plt.fill_between(data_trans["Year"],low,up,alpha=0.5)
plt.legend()
plt.xlim(1960,2021)
plt.xlabel("Year")
plt.ylabel("Urban population growth")
plt.savefig("urb_pop1.png", dpi = 300, bbox_inches='tight')
plt.show()


# Doing curve fit for Urban population growth  in Australia
param, cov = opt.curve_fit(curve_one,data_trans["Year"],data_trans["Australia"]
                           ,p0=[4e8,0.1])
sigma = np.sqrt(np.diag(cov))

# Error
low,up = err.err_ranges(data_trans["Year"],curve_one,param,sigma)
data_trans["fit_value"] = curve_one(data_trans["Year"], * param)
print(data_trans["fit_value"])

# Plotting the Urban population growth of Australia
plt.figure(figsize=(8, 5))
plt.title("Urban population growth (annual %) - Australia")
plt.plot(data_trans["Year"],data_trans["Australia"],label="data")
plt.plot(data_trans["Year"],data_trans["fit_value"],c="yellow",label="fit")
plt.fill_between(data_trans["Year"],low,up,alpha=0.5)
plt.legend()
plt.xlim(1960,2021)
plt.xlabel("Year")
plt.ylabel("Urban population growth")
plt.savefig("urb_pop2.png", dpi = 300, bbox_inches='tight')
plt.show()



