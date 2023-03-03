# -*- coding: utf-8 -*-
"""
This code plots 3 different visualisation methods (line plot,
stacked bar plot, and pie chart) from 2 different data sets.
"""

#Importing required packages
import matplotlib.pyplot as plt
import pandas as pd

#Function to plot line graph
def line_plot(lp):
    """
    

    This function will plot line graph of Mobile Cellulalar Subscription (per 
    100 people) of four countries from 2000-2015. Graphs are also saved
    as images to local directory.
    ----------
    lp : This is a DataFrame used to plot line graph.
       The mobile cellular subscribtions rate (per 100 persons) for four nations
       is shown in this line graph from 2000 to 2015.

    Returns
    -------
    None.

    """
    
    plt.figure()
    plt.plot(lp["Year"], lp["Albania"], label="Albania")
    plt.plot(lp["Year"], lp["Belgium"], label="Belgium")
    plt.plot(lp["Year"], lp["Malaysia"], label="Malaysia")
    plt.plot(lp["Year"], lp["Hungary"], label="Hungary")

    #set title and labels
    plt.xlabel("Year")
    plt.ylabel("Mobile Subscription(weighted Average)")
    plt.title("MOBILE CELLULAR SUBSCRIPTION")
    
    #Removing white space from left and right
    plt.xlim(2000, 2015)
    plt.legend()
    plt.savefig("lineplot.png")
    plt.show()
    
#Reading the file to the dataframe 'data'
data = pd.read_excel("dsfinal.xlsx")
line_plot(data)

#Function to plot bar plot
def bar_plot(bp):
    """
    

    This function creates bar diagram of Mobile Cellullar Subscription per 
    100 people of two countries from 2000-2015. Graphs are also saved
    as images to local directory.
    ----------
    bp : This is a DataFrame used to plot bar graph.
       The mobile cellular subscriber rates (per 100 persons) for two nations
       are shown in this  stacked bar plot from 2000 to 2015. 

    Returns
    -------
    None.

    """
    
    plt.figure()
    plt.bar(bp["Year"], bp["Greece"], label="Greece")
    plt.bar(bp["Year"], bp["Kuwait"], label="Kuwait", alpha=0.6)

    #Set title and labels
    plt.xlabel("Year")
    plt.ylabel("Mobile Subscription(Weighted Average)")
    plt.title("MOBILE CELLULAR SUBSCRIPTION")
    plt.savefig("barplot.png")
    plt.legend()
    plt.show()

#calling function to create plot
bar_plot(data)


#Function to plot pie chart
def pie_plot(pp):
    """
    

    This function will represent agriculture, forestry and fishing between
    the years of 1996 and 2018. Graphs are also saved as images to
    local directory.
    ----------
    pp : This is a DataFrame used to plot pie chart.
       This pie plot illustrates comparison of
       agriculture, forestry, and fishing between five nations in the years
       1996 and 2018. 

    Returns
    -------
    None.

    """
    
    plt.pie(pp["AFF(% of GDP)-1996"], labels=pp["Country"],
        autopct="%1.f%%")
    
    #Set title
    plt.title("Agriculture,forestry and fishing(1996)")
    plt.show()
    plt.pie(pp["AFF(% of GDP)-2018"], labels=pp["Country"],
        autopct="%1.f%%")
    
    #Set title
    plt.title("Agriculture,forestry and fishing(2018)")
    plt.savefig("piechart.png")
    plt.show()

#Reading the file into dataframe'data1'
data1 = pd.read_excel("pie96-18.xlsx")
pie_plot(data1)






