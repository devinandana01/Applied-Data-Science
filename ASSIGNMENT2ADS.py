"""
This code plots 4 different visualisation methods (bar plot,
line plot, and heat maps) from a data set based on 4 different indexes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

def data_f(a):
    """
    
    Parameters
    ----------
    x(str) : a string that represents the name of the EXCEL file to be read

    Returns :     
        datafile (pd.DataFrame): the pandas DataFrame that contains the data from the EXCEL file
        transpose (pd.DataFrame): the transposed pandas DataFrame
    -------

    """
    
    # create a file path using the input string 
    file_name = r'C:\Users\Devi Nandana S\Downloads' + a
    
    # read the excel file into a pandas DataFrame using the file path
    datafile = pd.read_excel(file_name)
    
    # transpose the DataFrame
    transpose = datafile.transpose()
    
    return(datafile,transpose)

data_f('\API_19_DS2_en_excel_v2_5351881.xls')

datafile = pd.read_excel(r'C:\Users\Devi Nandana S\Downloads\API_19_DS2_en_excel_v2_5351881.xls')


def clean_f(b):
    """
    Cleans the input DataFrame by filling any missing values with 0.

    Parameters
    ----------
    x (pandas DataFrame) : the DataFrame to be cleaned

    Returns :
        cleaned data without any NaN values
    -------

    """
    
    # count the number of missing values in each column of the DataFrame
    b.isnull().sum()
    
    # fill any missing values with 0 and update the DataFrame in place
    b.fillna(0,inplace=True)
    
    return

clean_f(datafile)

def stat_f(c):
    """
    This function takes a pandas DataFrame `c` and performs several statistical calculations on the columns.
    It prints the summary statistics, correlation matrix, skewness, and kurtosis
    for the selected columns.
    ----------
    c : Input the dataframe to perform different statistical functions

    Returns
    -------
    None.

    """
    
    # extract the columns from the 5th column onward and assign to variable "stats"
    stats = c.iloc[:,4:]
    
    # calculate the skewness,kurtosis and Covariance  
    print(skew(stats, axis=0, bias=True),kurtosis(stats, axis=0, bias=True),stats.describe(),stats.cov())
    
stat_f(datafile)

def elec_cons(d):
    """
    
        This function takes a pandas DataFrame `d` containing data on worldbank climate change data and creates a bar
    plot of the percentage change in Electric power consumption for a selection of countries.
    ----------
    d : This is a DataFrame used to plot bar graph.

    Returns:
        This function plots the bar plot of electric power consumption
    -------
    None.

    """
    
    
    # Select rows where the "Indicator Name" column is "Electric power consumption (kWh per capita)"
    data1 = d[d['Indicator Name']=='Electric power consumption (kWh per capita)']
    
    # Select rows where the "Country Name" column is one of a list of countries
    elec = data1[data1['Country Name'].isin(['Australia','Canada','Germany','France','Ireland','Japan','Poland','Sweden'])]
    
    # Define the width of each bar
    bar_width = 0.1
    
    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(elec))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(10, 8))
    plt.bar(r1, elec['1990'], color='lime', width=bar_width, edgecolor='black', label='1990')
    plt.bar(r2, elec['1995'], color='olive', width=bar_width, edgecolor='black', label='1995')
    plt.bar(r3, elec['2000'], color='mediumaquamarine', width=bar_width, edgecolor='black', label='2000')
    plt.bar(r4, elec['2005'], color='darkgreen', width=bar_width, edgecolor='black', label='2005')
    plt.bar(r5, elec['2010'], color='yellowgreen', width=bar_width, edgecolor='black', label='2010')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(elec))], elec['Country Name'], fontsize=15)
    
    #Set the y-tick labels
    plt.yticks(fontsize=15)
    
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold')
    plt.ylabel('Electrcity Power Consumption', fontweight='bold')
    plt.title('Electric power consumption (kWh per capita)', fontweight='bold')
    plt.savefig("barplot1.png")
    plt.legend()
    plt.show()
       

def elec_prod(e):
    """
    
    This function takes a pandas DataFrame `e` containing data on worldbank climate change data and creates a bar
    plot of the percentage change in Electricity production from renewable sources for a selection of countries.
    ----------
    e : This is a DataFrame used to plot bar graph.

    Returns:
        This function plots the bar plot of electricity production from 
        renewable sources
    -------
    None.

    """
    

    # Select rows where the "Indicator Name" column is "Electricity production from renewable sources, excluding hydroelectric (% of total)"
    dt1 = e[e['Indicator Name']=='Electricity production from renewable sources, excluding hydroelectric (% of total)']
    
    # Select rows where the "Country Name" column is one of a list of countries
    el = dt1[dt1['Country Name'].isin(['Australia','Canada','Germany','France','Ireland','Japan','Poland','Sweden'])]
    
    # Define the width of each bar
    bar_width = 0.1
    
    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(el))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(10, 8))
    plt.bar(r1, el['1990'], color='peru', width=bar_width, edgecolor='black', label='1990')
    plt.bar(r2, el['1995'], color='saddlebrown', width=bar_width, edgecolor='black', label='1995')
    plt.bar(r3, el['2000'], color='coral', width=bar_width, edgecolor='black', label='2000')
    plt.bar(r4, el['2005'], color='indianred', width=bar_width, edgecolor='black', label='2005')
    plt.bar(r5, el['2010'], color='maroon', width=bar_width, edgecolor='black', label='2010')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(el))], el['Country Name'], fontsize=15)
    
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold')
    plt.ylabel('Electricity Production', fontweight='bold')
    plt.title('Electricity production from renewable sources, excluding hydroelectric (% of total)', fontweight='bold')
    plt.savefig("barplot2.png")
    plt.legend()
    plt.show()
    
elec_cons(datafile)
elec_prod(datafile)


def dt_lineplot(f):
    """
    

    Plots a line graph of Urban population growth (annual %) of selected countries from the given dataframe
    
    Args:
    f : A Pandas dataframe containing the datafile
    ----------

    Returns:
        This function plots the line plot of urban population growth
    -------
    None.

    """
    
    
    # filtering out the data related to urban population growth annually in % for selected countries
    con = f[f['Indicator Name']=='Urban population growth (annual %)']
    data2 = con[con['Country Name'].isin(['Australia','Canada','Germany','France','Ireland','Japan','Poland','Sweden'])]
    
    # creating transpose of the filtered data
    nand = data2.transpose()
    nand.rename(columns=nand.iloc[0], inplace = True)
    data_transpose = nand.iloc[4:]

    #Replacing the null values by zeros
    data_transpose.fillna(0, inplace=True)
    
    # plotting the line graph
    plt.figure(figsize=(10,8))
    plt.plot(data_transpose.index, data_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(data_transpose.index, data_transpose['Canada'], linestyle='dashed', label='Canada')
    plt.plot(data_transpose.index, data_transpose['Germany'], linestyle='dashed', label='Germany')
    plt.plot(data_transpose.index, data_transpose['France'], linestyle='dashed', label='France')
    plt.plot(data_transpose.index, data_transpose['Ireland'], linestyle='dashed', label='Ireland')
    plt.plot(data_transpose.index, data_transpose['Japan'], linestyle='dashed', label='Japan')
    plt.plot(data_transpose.index, data_transpose['Poland'], linestyle='dashed', label='Poland')
    plt.plot(data_transpose.index, data_transpose['Sweden'], linestyle='dashed', label='Sweden')
    
    # Setting x limit
    plt.xlim('1995','2010')
    
    #set the x-tick labels
    plt.xticks(fontsize=10)
    
    #set the y-tick labels
    plt.yticks(fontsize=10)
    
    # Adding labels to the axis
    plt.xlabel('Year', fontsize=15, fontweight='bold')
    plt.ylabel('Urban population growth', fontsize=15, fontweight='bold')
    plt.title('Urban population growth (annual %)', fontsize=15, fontweight='bold')
    plt.savefig("lineplot1.png")
    plt.legend()
    plt.show()
    

def elec_lineplot(g):
    """
    

        This function takes a pandas dataframe 'g' as input and plots a line graph showing the Renewable electricity output (% of total electricity output) for 8 different countries.
    ----------
    g (pd.DataFrame): pandas dataframe

    Returns:
        This function plots the line plot of renwable electricity output
    -------
    None.

    """
    
    
    # filter the rows which have indicator name as "Renewable electricity output (% of total electricity output)"
    data3 = g[g['Indicator Name']=='Renewable electricity output (% of total electricity output)']
    
    # filter the rows for the 8 selected countries
    renew_elec = data3[data3['Country Name'].isin(['Australia','Canada','Germany','France','Ireland','Japan','Poland','Sweden'])]
    
    # creating transpose of the filtered data
    Tr = renew_elec.transpose()
    Tr.rename(columns=Tr.iloc[0], inplace = True)
    data3_transpose = Tr.iloc[4:]
    
    #Replacing the null values by zeros
    data3_transpose.fillna(0, inplace=True)
    
    #plotting the line graph
    plt.figure(figsize=(10,8))
    plt.plot(data3_transpose.index, data3_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(data3_transpose.index, data3_transpose['Canada'], linestyle='dashed', label='Canada')
    plt.plot(data3_transpose.index, data3_transpose['Germany'], linestyle='dashed', label='Germany')
    plt.plot(data3_transpose.index, data3_transpose['France'], linestyle='dashed', label='France')
    plt.plot(data3_transpose.index, data3_transpose['Ireland'], linestyle='dashed', label='Ireland')
    plt.plot(data3_transpose.index, data3_transpose['Japan'], linestyle='dashed', label='Japan')
    plt.plot(data3_transpose.index, data3_transpose['Poland'], linestyle='dashed', label='Poland')
    plt.plot(data3_transpose.index, data3_transpose['Sweden'], linestyle='dashed', label='Sweden')
    
    #setting x limit
    plt.xlim('1995','2010')
    
    #set the x-tick labels
    plt.xticks(fontsize=10)
    
    #set the y-tick labels
    plt.yticks(fontsize=10)
    
    #adding labels to the axis
    plt.xlabel('Year', fontsize=15, fontweight='bold')
    plt.ylabel('Renewable electricity output', fontsize=15, fontweight='bold')
    plt.title('Renewable electricity output (% of total electricity output)', fontsize=15, fontweight='bold')
    plt.savefig("lineplot2.png")
    plt.legend()
    plt.show()
    
    
dt_lineplot(datafile)     
elec_lineplot(datafile)

def heatmap_ireland(h):
    """
    

    A function that creates a heatmap of the correlation matrix between different indicators for Ireland.

    Args:
    h (pandas.DataFrame): A DataFrame containing data on different indicators for various countries.

    Returns:
    This function plots the heatmap of Ireland
    ----------

    """
    
    #Specify the indicators to be used in the heatmap
    indicator=['Electric power consumption (kWh per capita)',
               'Electricity production from renewable sources, excluding hydroelectric (% of total)',
               'Urban population growth (annual %)',
               'Renewable electricity output (% of total electricity output)',
               'Renewable energy consumption (% of total final energy consumption)']
    
    # Filter the data to keep only Ireland's data and the specified indicators
    ire = h.loc[h['Country Name']=='Ireland']
    ireland = ire[ire['Indicator Name'].isin(indicator)]
    
    # Pivot the data to create a DataFrame with each indicator as a column
    ireland_df = ireland.pivot_table(ireland, columns= h['Indicator Name'])
    
    # Compute the correlation matrix for the DataFrame
    ireland_df.corr()
    
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12,8))
    sns.heatmap(ireland_df.corr(), fmt='.2g', annot=True, cmap='Greens', linecolor='black')
    plt.title('Ireland', fontsize=15, fontweight='bold')
    plt.show()


def heatmap_canada(i):
    """
    
        A function that creates a heatmap of the correlation matrix between different indicators for Canada.

    Args:
    x (pandas.DataFrame): A DataFrame containing data on different indicators for various countries.

    Returns:
    This function plots the heatmap of Canada
    ---------

    """
    
    
    #Specify the indicators to be used in the heatmap
    indicator = ['Electric power consumption (kWh per capita)',
               'Electricity production from renewable sources, excluding hydroelectric (% of total)',
               'Urban population growth (annual %)',
               'Renewable electricity output (% of total electricity output)',
               'Renewable energy consumption (% of total final energy consumption)']
    
    #Filter the data to keep only Canada's data and the specified indicators
    can = i.loc[i['Country Name']=='Canada']
    Canada = can[can['Indicator Name'].isin(indicator)]
    
    # Pivot the data to create a DataFrame with each indicator as a column
    Canada_df = Canada.pivot_table(Canada,columns= i['Indicator Name'])
    
    # Compute the correlation matrix for the DataFrame
    Canada_df.corr()
    
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12,8))
    sns.heatmap(Canada_df.corr(), fmt='.2g', annot=True ,cmap='BuPu', linecolor='black')
    plt.title('Canada', fontsize=15, fontweight='bold')
    plt.show()


heatmap_ireland(datafile)
heatmap_canada(datafile)