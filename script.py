#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:43:33 2023

@author: Musawir
"""


import os
from sklearn.cluster import KMeans
import itertools as iter
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt


def err_ranges(x_data, func, param, sigma):
    """
    This function gets x-axis data, function, param and sigma parameters
    and returns the upper and lower error boundaries
    """
    minimum = func(x_data, *param)
    maximum = minimum
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x_data, *p)
        minimum = np.minimum(minimum, y)
        maximum = np.maximum(maximum, y)
        
    return minimum, maximum   


def linfunc(x, a, b):
    """
    Calculates Linear Function value
    """
    y = a*x + b
    return y


def logistics(t, n0, g, t0):
    """
    Calculates logistics function value
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def goods_export_plot(df):
    """
    Parameters
    ----------
    df : DataFrame
        Having data to plot for 

    Returns
    -------
    None.

    """
    yearly_indexed_df = df.copy().dropna()
    yearly_indexed_df = yearly_indexed_df.set_index("Country Name")
    print(yearly_indexed_df.index.tolist())
    yearly_indexed_df = yearly_indexed_df.drop(
        ['Indicator Name', 'Indicator Code', 'Country Code'], axis=1)

    yearly_indexed_df = yearly_indexed_df.T
    af_data = yearly_indexed_df[["Bangladesh"]]
    print("AFFDDFD")
    print(af_data)
    af_data['year'] = pd.to_numeric(af_data.index)
    param, covar = opt.curve_fit(logistics, af_data["year"], 
                                  af_data["Bangladesh"], 
                                  p0=(20, 0.1, 1990), maxfev=5000)
    af_data["fit"] = logistics(af_data["year"], *param)
    af_data.plot("year", ["Bangladesh", "fit"])
    plt.plot(af_data["year"], af_data['Bangladesh'])
    plt.ylabel("Exports of goods and services (% of GDP)")
    plt.xlabel("Year")
    plt.title("Exports of goods and services (% of GDP)")
    

def co2_gas_solid_rel(df):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame object holding the data for clustering.
    Returns
    -------
    None.

    """
    x_column = 'CO2 From Gas'
    y_column = 'CO2 From Solid'
    x_data = df[x_column]
    y_data = df[y_column]
    x_data = x_data.dropna()
    y_data = y_data.dropna()
    
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['clusters'] = kmeans.fit_predict(df[[x_column, y_column]])

    # Get centers for data clusters
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    ## add to df
    df['cen_x'] = df.clusters.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    df['cen_y'] = df.clusters.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
    
    # define and map colors
    c = ['r', 'b', 'y']
    df['c'] = df.clusters.map({0:c[0], 1:c[1], 2:c[2]})

    fig, ax = plt.subplots()

    ax.scatter(x_data, y_data, 
                c=df.c, alpha = 1, s=4)
    plt.xlabel("CO2 Emission from Gas Fuel Sources")
    plt.ylabel("CO2 Emission from Solid Fuel Sources")
    plt.title("Relationship between CO2 Gas and Solid Sources")
    plt.scatter(df['cen_x'], df['cen_y'], 10, "blue", marker="d",)
    
    
def co2_relation(df):
    """
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    None.

    """
    df = df.dropna()
    x_column = 'CO2 From Gas'
    y_column = 'CO2 From Liquid'
    z_column = 'CO2 From Solid'
    x = df[x_column]
    y = df[y_column]
    z = df[z_column]
    colors = ['#DF2020', '#81DF20', '#2095DF']
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[[x_column, y_column, z_column]])
    df['color'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    fig = plt.figure(figsize=(26,6))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(x, y, z, c=df['color'], s=16)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel(z_column)
    


current_dir = os.path.dirname(__file__)
co2_gas = pd.read_csv(os.path.join(current_dir, 'CO2-from-gaseous-fuel-consumption.csv'))
co2_liquid = pd.read_csv(os.path.join(current_dir, 'CO2-from-liquid-fuel-consumption.csv'))
co2_solid = pd.read_csv(os.path.join(current_dir, 'CO2-from-solid-fuel-consumption.csv'))
goods_export = pd.read_csv(os.path.join(current_dir, 'Exports-of-goods.csv'))

goods_export_plot(goods_export.copy())

years = [str(i) for i in range(2007, 2015)]
co2_liquid_sub = co2_liquid[years]
pd.plotting.scatter_matrix(co2_liquid_sub, figsize=(8, 5), s=5, alpha=0.8)

co2_gas = co2_gas[co2_gas['2016'].notna()]
co2_liquid = co2_liquid[co2_liquid['2016'].notna()]
co2_solid = co2_solid[co2_solid['2016'].notna()]
goods_export = goods_export[goods_export['2016'].notna()]

co2_gas =  co2_gas[["Country Name", '2016']].copy()
co2_liquid =  co2_liquid[["Country Name", '2016']].copy()
co2_solid =  co2_solid[["Country Name", '2016']].copy()
goods_export =  goods_export[["Country Name", '2016']].copy()

co2_whole = pd.merge(co2_gas, co2_liquid, on = "Country Name", how = "outer")
co2_whole = pd.merge(co2_whole, co2_solid, on = "Country Name", how = "outer")

co2_whole = co2_whole.rename(columns = {
    "2016_x":"CO2 From Gas", "2016_y":"CO2 From Liquid",
    "2016": "CO2 From Solid"})

co2_whole = co2_whole.set_index("Country Name")

co2_whole = co2_whole.dropna()
co2_gas_solid_rel(co2_whole.copy())

co2_relation(co2_whole.copy())


plt.show()

co2_whole.T['United States'].plot.pie(
    autopct="%.2f%%", title="CO2 Distribution by Sources in United States")












