import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.stats import norm
import statistics
import numpy as np
from scipy.signal import find_peaks as fp



def read_data(plk_file_path): 
    
    # Reading in the data
    mean_wind_2018 = pd.read_pickle(plk_file_path)
    
    mean_wind_2018.Timestamp = pd.to_datetime(mean_wind_2018.Timestamp,format='%Y-%m-%d %H:%M:%S.%f')

    # Calculating the rolling mean and std--- window of 1 min
    mean_wind_2018['roll_mean'] = mean_wind_2018.anc_gust_wind_speed.rolling(window=60).median()
    mean_wind_2018['roll_std'] = mean_wind_2018.anc_gust_wind_speed.rolling(window=60).std()

    # x = mean +/- 5 sigma
    mean_wind_2018['upper_bound'] = mean_wind_2018['roll_mean'] + 5*mean_wind_2018['roll_std']
    mean_wind_2018['lower_bound'] = mean_wind_2018['roll_mean'] - 5*mean_wind_2018['roll_std']

    # just to make it shorter
    df2 = mean_wind_2018

    #Searching for possible spikes
    spikes_roll_upper = df2.iloc[np.where(df2.anc_gust_wind_speed > df2.upper_bound)]

    spikes_roll_lower = df2.iloc[np.where(df2.anc_gust_wind_speed < df2.lower_bound)]
    
    
    spikes_roll_upper['spikes_mean'] = spikes_roll_upper.anc_gust_wind_speed.rolling(window=60).median()
    spikes_roll_lower['spikes_mean'] = spikes_roll_lower.anc_gust_wind_speed.rolling(window=60).median()


    spikes_roll_upper['spikes_std'] = spikes_roll_upper.anc_gust_wind_speed.rolling(window=60).std()
    spikes_roll_lower['spikes_std'] = spikes_roll_lower.anc_gust_wind_speed.rolling(window=60).std()

    spikes_roll_upper['spikes_height'] = spikes_roll_upper['anc_gust_wind_speed'] - spikes_roll_upper['roll_mean']
    spikes_roll_lower['spikes_height'] = spikes_roll_lower['roll_mean'] - spikes_roll_lower['anc_gust_wind_speed']

    spikes_roll_upper['spikes_significance'] = spikes_roll_upper['spikes_height']/spikes_roll_upper['spikes_std']
    spikes_roll_lower['spikes_significance'] = spikes_roll_lower['spikes_height']/spikes_roll_lower['spikes_std']
    
    
    
    height_average_upper = np.average(spikes_roll_upper.spikes_height)
    height_average_lower = np.average(spikes_roll_lower.spikes_height)

    upper_selected_spikes  = spikes_roll_upper.iloc[np.where(spikes_roll_upper.spikes_height > height_average_upper )]
    lower_selected_spikes  = spikes_roll_lower.iloc[np.where(spikes_roll_lower.spikes_height < height_average_lower)]
    
    
    upper_selected_spikes  = upper_selected_spikes .iloc[np.where(upper_selected_spikes.anc_gust_wind_speed < 45 )]






    
    return df2,upper_selected_spikes ,lower_selected_spikes


def spikes_ave_min(spikes_roll_upper_new):
    from datetime import datetime

    spikes_roll_upper_new['Timestamp'] = pd.to_datetime(spikes_roll_upper_new['Timestamp'])
    spikes_roll_upper_new.set_index(['Timestamp'],inplace=True)
    avarege_spike_in_a_min_upper = spikes_roll_upper_new.resample('60S').mean()
    avarege_spike_in_a_min_upper = avarege_spike_in_a_min_upper.reset_index()
    
    array = np.array(avarege_spike_in_a_min_upper.anc_gust_wind_speed)
    valid_index = []
    for i in range(len(avarege_spike_in_a_min_upper)):
        if avarege_spike_in_a_min_upper.anc_gust_wind_speed.isnull()[i]== False:
            valid_index.append(i)

    bar_spikes = avarege_spike_in_a_min_upper.iloc[valid_index]
    
    
     #bar_spikes = spikes_ave_min(spikes_roll_upper)
    
    #gust_stow = 16.9
    #window_stow = 11.1
 

    fig, (ax3) = plt.subplots(1, 1, figsize=(15,10))

    
  
    ax3.plot(bar_spikes['Timestamp'],bar_spikes['anc_gust_wind_speed'],'o-')
    xmin, xmax = ax3.get_xlim()
    ymin, ymax = ax3.get_ylim()
    #ax1.hlines(window_stow, xmin, xmax, linestyles='dotted', colors='r')
    #ax3.hlines(gust_stow, xmin, xmax, linestyles='dotted', colors='r')
    #ax1.text(xmin, window_stow+0.5, '  window limit')
    #ax3.text(xmin, gust_stow+0.5, '  gust limit')
    ax3.set_xlabel('Timestamp' )
    ax3.set_ylabel('Wind speed m/s')

        
    #plt.savefig(str(save_name))
    plt.show()
    
    
    return bar_spikes
    
    

def spikes_monthly_info(spikes_roll_upper,year,save_name):
    
    Number_of_spike_each_month = {'month': [],'spike_sum' :[] , 'spikes_mean' :[] , 'No. of spikes' :[],'spikes_std' :[],'Maximum_wind_speed' :[],'Minimum_wind_speed' :[] }

    year = str(year)

    for i in np.arange(0,13):

        if i==0:
            print('skip')

        elif i<9:
            start_date = year+'-0'+str(i)+'-01 00:00:00'
            end_date = year+'-0'+str(i+1)+'-01 00:00:00'

            mask = (spikes_roll_upper['Timestamp'] > start_date) & (spikes_roll_upper['Timestamp'] <= end_date)
            df2_new = spikes_roll_upper[mask]

            spikes_month = year+'-0'+str(i)
            spike_sum = df2_new.anc_gust_wind_speed.sum() 
            no_of_spikes = len(df2_new)
            spike_mean = df2_new.anc_gust_wind_speed.mean() 
            spike_std = df2_new.anc_gust_wind_speed.std()
            spike_max = df2_new.anc_gust_wind_speed.max()
            spike_min = df2_new.anc_gust_wind_speed.min()

            Number_of_spike_each_month['month'].append(spikes_month)
            Number_of_spike_each_month['spike_sum'].append(spike_sum)
            Number_of_spike_each_month['spikes_mean'].append(spike_mean)
            Number_of_spike_each_month['No. of spikes' ].append(no_of_spikes)
            Number_of_spike_each_month['spikes_std'].append(spike_std)
            Number_of_spike_each_month['Maximum_wind_speed'].append(spike_max)
            Number_of_spike_each_month['Minimum_wind_speed'].append(spike_min)



        elif i == 9:
            start_date = year+'-0'+str(i)+'-01 00:00:00'
            end_date = year+'-'+str(i+1)+'-01 00:00:00'

            mask = (spikes_roll_upper['Timestamp'] > start_date) & (spikes_roll_upper['Timestamp'] <= end_date)
            df2_new = spikes_roll_upper[mask]

            spikes_month = year+'-0'+str(i)
            spike_sum = df2_new.anc_gust_wind_speed.sum() 
            no_of_spikes = len(df2_new)
            spike_mean = df2_new.anc_gust_wind_speed.mean() 
            spike_std = df2_new.anc_gust_wind_speed.std() 

            Number_of_spike_each_month['month'].append(spikes_month)
            Number_of_spike_each_month['spike_sum'].append(spike_sum)
            Number_of_spike_each_month['spikes_mean'].append(spike_mean)
            Number_of_spike_each_month['No. of spikes' ].append(no_of_spikes)
            Number_of_spike_each_month['spikes_std'].append(spike_std)

        elif i == 12:
            start_date = year +'-'+str(i)+'-01 00:00:00'
            end_date = year +'-'+str(i)+'-31 00:00:00'
            mask = (spikes_roll_upper['Timestamp'] > start_date) & (spikes_roll_upper['Timestamp'] <= end_date)
            df2_new = spikes_roll_upper[mask]

            spikes_month = year +'-'+str(i)
            spike_sum = df2_new.anc_gust_wind_speed.sum() 
            no_of_spikes = len(df2_new)
            spike_mean = df2_new.anc_gust_wind_speed.mean() 
            spike_std = df2_new.anc_gust_wind_speed.std() 

            Number_of_spike_each_month['month'].append(spikes_month)
            Number_of_spike_each_month['spike_sum'].append(spike_sum)
            Number_of_spike_each_month['spikes_mean'].append(spike_mean)
            Number_of_spike_each_month['No. of spikes' ].append(no_of_spikes)
            Number_of_spike_each_month['spikes_std'].append(spike_std)


        else:
            start_date = year+'-'+str(i)+'-01 00:00:00'
            end_date = year +'-'+str(i+1)+'-01 00:00:00'
            mask = (spikes_roll_upper['Timestamp'] > start_date) & (spikes_roll_upper['Timestamp'] <= end_date)
            df2_new = spikes_roll_upper[mask]

            spikes_month = year +'-'+str(i)
            spike_sum = df2_new.anc_gust_wind_speed.sum() 
            no_of_spikes = len(df2_new)
            spike_mean = df2_new.anc_gust_wind_speed.mean() 
            spike_std = df2_new.anc_gust_wind_speed.std() 

            Number_of_spike_each_month['month'].append(spikes_month)
            Number_of_spike_each_month['spike_sum'].append(spike_sum)
            Number_of_spike_each_month['spikes_mean'].append(spike_mean)
            Number_of_spike_each_month['No. of spikes' ].append(no_of_spikes)
            Number_of_spike_each_month['spikes_std'].append(spike_std)
            

    x  = Number_of_spike_each_month['month']  
    y1 = Number_of_spike_each_month['Maximum_wind_speed']  

    y2 = Number_of_spike_each_month['spikes_mean']  

    y3 = Number_of_spike_each_month['spikes_std']  


    colors = Number_of_spike_each_month['No. of spikes' ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,8))

    z1_plot = ax1.scatter(x, y1, c=y3,s=colors, alpha=0.5, cmap='nipy_spectral',vmin=0.0)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Maximum_wind_speed m/s')
    ax1.set_title('The sum of possible spikes each moth')

    plt.colorbar(z1_plot,ax=ax1,orientation='horizontal',label='Standard deviation')


    ax2.plot(x, y3)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Wind_speed std')


    plt.savefig('')

    plt.show()

 
    
    return Number_of_spike_each_month

            
def specific_dates_dataframe(dataframe,spikes_roll_upper,spikes_roll_lower,start_date,end_date):

    df2 = dataframe
    mask = (df2['Timestamp'] > start_date) & (df2['Timestamp'] <= end_date)
    df2_new = df2.loc[mask]

    mask2 = (spikes_roll_lower['Timestamp'] > start_date) & (spikes_roll_lower['Timestamp'] <= end_date)
    spikes_roll_lower_new = spikes_roll_lower.loc[mask2]

    mask3 = (spikes_roll_upper['Timestamp'] > start_date) & (spikes_roll_upper['Timestamp'] <= end_date)
    spikes_roll_upper_new = spikes_roll_upper.loc[mask3]
    
    return df2_new,spikes_roll_upper_new,spikes_roll_lower_new


def spikes_ave_min2(spikes_roll_upper_new):
    from datetime import datetime

    spikes_roll_upper_new['Timestamp'] = pd.to_datetime(spikes_roll_upper_new['Timestamp'])
    spikes_roll_upper_new.set_index(['Timestamp'],inplace=True)
    avarege_spike_in_a_min_upper = spikes_roll_upper_new.resample('600S').mean()
    avarege_spike_in_a_min_upper = avarege_spike_in_a_min_upper.reset_index()
    
    array = np.array(avarege_spike_in_a_min_upper.anc_gust_wind_speed)
    valid_index = []
    for i in range(len(avarege_spike_in_a_min_upper)):
        if avarege_spike_in_a_min_upper.anc_gust_wind_speed.isnull()[i]== False:
            valid_index.append(i)

    bar_spikes = avarege_spike_in_a_min_upper.iloc[valid_index]
    
    gust_stow = 16.9
    window_stow = 11.1

    fig, (ax1) = plt.subplots(1, 1, figsize=(15,5))
    
    ax1.plot(bar_spikes['Timestamp'],bar_spikes['anc_gust_wind_speed'],'o-')
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    #ax1.hlines(window_stow, xmin, xmax, linestyles='dotted', colors='r')
    ax1.hlines(gust_stow, xmin, xmax, linestyles='dotted', colors='r')
    #ax1.text(xmin, window_stow+0.5, '  window limit')
    ax1.text(xmin, gust_stow+0.5, '  gust limit')
    ax1.set_xlabel('Timestamp' )
    ax1.set_ylabel('Wind speed m/s')
    plt.show() 
    
    return bar_spikes

