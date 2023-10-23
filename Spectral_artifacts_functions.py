# My functions

# Defined functions

import katdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

import scipy.stats as stats
from matplotlib.colors import LogNorm


def channel_folding_residual(spectrum, chan_range, fold_interval, segment_width=2048, poly_deg=7):
    """ Compute channel-folding residual after removing the fitted bandpass response.
    
    Normalises the provided spectrum by a polynomial fit to remove the bandpass response.
    The normalised spectrum is then folded to provide the mean deviation from the bandpass 
    response across the specified folding interval(s).
    
    Parameters
    ----------
    spectrum : 1-dimensional numpy array.
        The mean spectrum of a single correlation product across one or more scans.
    chan_range : Numpy slice object. 
        Specifies the range of channels over which to fold the spectrum.
    fold_interval : int or list of integers
        Integer value(s) indicating the interval across which to fold the spectrum.
    segment_width : int, optional
        The specified channel range gets split into segments of specified width and 
        each segment gets analysed separately. Results from individual segments are
        averaged at the end. This lowers the fitting error when analysing spectra
        with undulating bandpass responses.
    pol_deg : int, optional
        The degree of the polynomial used to fit the bandpass response.
        
    Returns
    -------
    data_res : Numpy array, or list of Numpy arrays per fold interval
        A numpy array of the midpoint of each gain solution's dump interval.
        
    """
    # Compute analysis indices
    start_ind = chan_range.start
    stop_ind = chan_range.stop
    ch_rng_calc = np.arange(int(np.ceil(1.0*start_ind/segment_width)*segment_width), 
                            int(np.floor(1.0*stop_ind/segment_width)*segment_width))
    
    
    Nsect = ch_rng_calc.shape[0]//segment_width 
    #print('Nsect is : ')
    #print(' ')
    #print(Nsect)
    
    # Convert scalar fold interval to list,.... 
    
    no_list = type(fold_interval) is not list
    
    if no_list: #if the fold interval is not yet a list make is a list
        fold_interval = [fold_interval]
        
    data_res = []
    
    # Compute folding residual for each specified folding interval
    for ii_fold in range(len(fold_interval)):  #. from 0, ....,63 for this example
        
        # Compute residual for list of folding widths
        data_res_ii = np.zeros(fold_interval[ii_fold])
        
        for ii_sect in range(Nsect):
            # Compute channel range
            ch_rng_sect = ch_rng_calc[ii_sect*segment_width:(ii_sect+1)*segment_width]
            # Get spectrum over the desired channel range
            data_sect = spectrum[ch_rng_sect]
            # Compute the smoothed bandpass shape by fitting a low-order polynomial fit to the data 
            data_coeff = np.polyfit(np.arange(segment_width), data_sect, deg=poly_deg)
            data_poly = np.poly1d(data_coeff)
            data_smooth = data_poly(np.arange(segment_width))
            # Compute error between the spectrum and the smoothed bandpass shape
            data_norm = data_sect/data_smooth - 1.0
            # Reshape according to the desired folding width
            data_norm_rs = np.reshape(data_norm, (segment_width//fold_interval[ii_fold], fold_interval[ii_fold]))
            # Add folded data for this segment
            data_res_ii += np.sum(data_norm_rs, axis=0)
        # Compute average error across segments for this folding width 
        data_res.append(data_res_ii/(1.0*Nsect*segment_width//fold_interval[ii_fold]))
    # Recast to scalar if required
    if no_list:
        data_res = data_res[0]
    return data_res



# This function calculates the residuals between baselines

def mean_spectra_folded_residuals(vis, cp_loaded, d,channel_min, channel_max) :
    
    show_fold_selection = True
    ch_fold = np.s_[channel_min:channel_max]
    fold_factor_list = 64
    dp_start = 2

    # Compute mean spectruma and maximum error in 64-chan folding residual
    max_res = []
    
    cp_tup_all = []
    cp_mean_all = []
    max_ind_all = []
    channels_all = []
    
    for ii_cp in range(cp_loaded.shape[0]):
        cp_tup = cp_loaded[ii_cp,:]
        cp_mean = np.abs(np.mean(vis[dp_start:,:,ii_cp], axis=0))
        
        # Compute folding residual sequences
        data_res = channel_folding_residual(cp_mean, ch_fold, fold_factor_list)
        max_ind = np.argmax(np.abs(data_res))
        
        
        
        max_res.append(data_res[max_ind])
        
        
        cp_tup_all.append(cp_tup)
        
       
        cp_mean_all.append(cp_mean[max_ind])
        
        max_ind_all.append(max_ind)

     
        
        #channels_all[ii_cp] = d.channels
        #print(channels_all)
        # Show region selected for analysis
        if ii_cp==0 and show_fold_selection:
            fig,ax = plt.subplots(figsize=(10,5))
            
            ax.plot(d.channels,cp_mean, label=str(cp_tup)) 
            ax.set_xlabel('Channels')
            ax.set_ylabel('Power [Counts]')
            ax.fill_between(np.arange(ch_fold.start, ch_fold.stop), 0, 1.05*np.max(cp_mean), 
                     facecolor="gray", color='gray', alpha=0.4, label='Analysis Region')
            
            ax2=ax.twiny()
            ax2.plot( d.channel_freqs/1e6,cp_mean)
            ax2.set_xlabel('Frequencies (MHz)')
            ax2.set_ylabel('Power [Counts]')
        
            plt.title('Channels selected for channel-folding analysis: {}'.format(d.experiment_id),fontsize=15)
            plt.tight_layout()

            plt.show()
            
            
    
            
    #return cp_tup_all, cp_mean_all,max_ind_all, max_res 
    return max_res 



# Check artefacts in selected file


    
def artifacts(cp_loaded,max_res ) :

    # Plot maximum residuals assuming only cross were selected

    ind_h, data_h,ind_v, data_v = [], [], [], []
    
    H_pol_attennas = []
    V_pol_attennas = []
    
    V_pol_attenna_index_x = []
    V_pol_attenna_index_y = []

    H_pol_attenna_index_x = []
    H_pol_attenna_index_y = []

    for ii_cp in range(cp_loaded.shape[0]):
        
        cp_rec_ind = int(cp_loaded[ii_cp,:][0][1:-1])
        
        attenna_ind_x = int(cp_loaded[ii_cp,:][0][1:-1])
        attenna_ind_y = int(cp_loaded[ii_cp,:][1][1:-1])
        
        
        cp_tup = cp_loaded[ii_cp,:]

        attenna_id1 = cp_loaded[ii_cp,:][0][1:4]   
        attenna_id2 = cp_loaded[ii_cp,:][1][1:4]

        pol0 = cp_loaded[ii_cp,:][0][-1]
        pol1 = cp_loaded[ii_cp,:][1][-1]
        
        
        
        if attenna_id1==attenna_id2 and pol0== pol1=='h': # auto correlation
            H_pol_attennas.append(cp_tup)
            ind_h.append(cp_rec_ind)
            data_h.append(abs(max_res[ii_cp]))
            
            H_pol_attenna_index_x.append(attenna_ind_x)
            H_pol_attenna_index_y.append(attenna_ind_y)

        if attenna_id1==attenna_id2 and pol0== pol1=='v': # auto correlation
            V_pol_attennas.append(cp_tup)
            ind_v.append(cp_rec_ind)
            data_v.append(abs(max_res[ii_cp]))
            
            V_pol_attenna_index_x.append(attenna_ind_x)
            V_pol_attenna_index_y.append(attenna_ind_y)


        if attenna_id1 !=attenna_id2 and pol0==pol1=='h' : # cross correlation 
            
            H_pol_attennas.append(cp_tup)
            ind_h.append(cp_rec_ind)
            data_h.append(abs(max_res[ii_cp]))
            
            H_pol_attenna_index_x.append(attenna_ind_x)
            H_pol_attenna_index_y.append(attenna_ind_y)

        if attenna_id1 !=attenna_id2 and pol0==pol1=='v' : # cross correlation
            
            V_pol_attennas.append(cp_tup)
            ind_v.append(cp_rec_ind)
            data_v.append(abs(max_res[ii_cp]))
            
            V_pol_attenna_index_x.append(attenna_ind_x)
            V_pol_attenna_index_y.append(attenna_ind_y)
            



  

    
    return ind_h, data_h, H_pol_attennas, ind_v, data_v, V_pol_attennas




def Theshold(matrix) :
    data_ = matrix

    mu, std = norm.fit(data_ )

    # Plot the histogram.
    q_low = np.quantile(data_,0.25)
    q_hi  =  np.quantile(data_,0.75)
    median = np.mean(data_)

    IQR = q_hi - q_low

    lower_bound = q_low - 1.5*IQR # below 2 sigma
    upper_bound = q_hi + 1.5*IQR # above 2 sigma
    
    return  lower_bound, upper_bound






def residual_plot():
    
    

    fig, axs = plt.subplots(2,1, figsize=(8,7), sharex=True)
    axs[0].scatter(ind_h, data_h, s=20, color='blue',label='H-pol')
    axs[0].scatter(ind_v, data_v, s=20,color='red', label='V-pol')
    
    axs[0].axhline(y=3,c="blue",linewidth=3,label='H-pol threshold')
    axs[0].axhline(y=3,c="red",linewidth=3,label='V-pol threshold')
    
    axs[0].grid()
    axs[0].legend()
    axs[0].set_ylabel('Max Residual')
    axs[0].set_xlim([-1,64])
    axs[0].set_xlabel('Antenna Index')
    axs[0].set_title('Auto correlation Max absolute folding residual x {}',fontsize=15)
    
    axs[1].scatter(ind_cross_h, (np.array(data_cross_h)), s=20, color='blue',label='H-pol')
    axs[1].scatter(ind_cross_v, (np.array(data_cross_v)), s=20,color='red', label='V-pol')
    
    axs[1].axhline(y=3,c="blue",linewidth=3,label='H-pol threshold')
    axs[1].axhline(y=3,c="red",linewidth=3, label='V-pol threshold')
    
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel('Max Residual')
    axs[1].set_xlim([-1,64])
    axs[1].set_xlabel('Antenna Index')
    axs[1].set_title('Cross correlation Max absolute folding residual x {}',fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    
    
    


    
def artifacts(cp_loaded,max_res ) :

    # Plot maximum residuals assuming only cross were selected

    ind_h, data_h,ind_v, data_v = [], [], [], []
    
    H_pol_attennas = []
    V_pol_attennas = []
    
    V_pol_attenna_index_x = []
    V_pol_attenna_index_y = []

    H_pol_attenna_index_x = []
    H_pol_attenna_index_y = []

    for ii_cp in range(cp_loaded.shape[0]):
        
        cp_rec_ind = int(cp_loaded[ii_cp,:][0][1:-1])
        
        attenna_ind_x = int(cp_loaded[ii_cp,:][0][1:-1])
        attenna_ind_y = int(cp_loaded[ii_cp,:][1][1:-1])
        
        
        cp_tup = cp_loaded[ii_cp,:]

        attenna_id1 = cp_loaded[ii_cp,:][0][1:4]   
        attenna_id2 = cp_loaded[ii_cp,:][1][1:4]

        pol0 = cp_loaded[ii_cp,:][0][-1]
        pol1 = cp_loaded[ii_cp,:][1][-1]
        
        
        
        if attenna_id1==attenna_id2 and pol0== pol1=='h': # auto correlation
            H_pol_attennas.append(cp_tup)
            ind_h.append(cp_rec_ind)
            data_h.append(abs(max_res[ii_cp]))
            
            H_pol_attenna_index_x.append(attenna_ind_x)
            H_pol_attenna_index_y.append(attenna_ind_y)

        if attenna_id1==attenna_id2 and pol0== pol1=='v': # auto correlation
            V_pol_attennas.append(cp_tup)
            ind_v.append(cp_rec_ind)
            data_v.append(abs(max_res[ii_cp]))
            
            V_pol_attenna_index_x.append(attenna_ind_x)
            V_pol_attenna_index_y.append(attenna_ind_y)


        if attenna_id1 !=attenna_id2 and pol0==pol1=='h' : # cross correlation 
            
            H_pol_attennas.append(cp_tup)
            ind_h.append(cp_rec_ind)
            data_h.append(abs(max_res[ii_cp]))
            
            H_pol_attenna_index_x.append(attenna_ind_x)
            H_pol_attenna_index_y.append(attenna_ind_y)

        if attenna_id1 !=attenna_id2 and pol0==pol1=='v' : # cross correlation
            
            V_pol_attennas.append(cp_tup)
            ind_v.append(cp_rec_ind)
            data_v.append(abs(max_res[ii_cp]))
            
            V_pol_attenna_index_x.append(attenna_ind_x)
            V_pol_attenna_index_y.append(attenna_ind_y)
            



  

    
    return ind_h, data_h, H_pol_attennas, ind_v, data_v, V_pol_attennas




    


    
    
    
def normalize_mean_err(data):
    data_norm = np.array(data)/np.nanmedian(data)
    
    return (data_norm)

def try2(a):
    
    return(a+1)





def identify_max_res(matrix):
    
    maximum_residual_v = 4 # The maximum residual threshold for v polarization 

    final_matrix_v_normalized = np.copy(matrix) 

    for i in range(len(final_matrix_v_normalized)):
        
        for j in range(len(final_matrix_v_normalized[i])):

            if (final_matrix_v_normalized[i][j]) < maximum_residual_v:
                
                final_matrix_v_normalized[i][j] = None  # This matrix will only sow possble artifacts residual only
                                                        # Everything else will be zero
                    
    return final_matrix_v_normalized


def normalize_mean_std(data):
    
    data = np.array(data)
    
    mean = np.median(data)
    std = np.std(data)
    
    return (data - mean)/std
    
    
        
        
def attenna_res_info(matrix):
    
    threshold = 4
    
    print(' Possible offending antennas')
    
    final_matrix_h_normalized =  np.copy(matrix)
    maximum_residual_h = threshold
    
    print('Attena index with maximum residuals in the ' + str(matrix))
    condition = (final_matrix_h_normalized >= maximum_residual_h)
    location = np.argwhere(condition == True) 

    residual_list_h =[]
    no_of_offending_attenas_h = 0

    for i in location: 
        no_of_offending_attenas_h = no_of_offending_attenas_h +1
        residual_list_h.append(i)

    print(' ')
    print('Number of offending  attenas = ' + str(no_of_offending_attenas_h))
    print(' ')
    print('List of the attenna index :' )
    print(' ')
    print((residual_list_h))

    print(' ')
    
    return residual_list_h


def offending_antenna_idx_v(residual_list_v):
    
    offending_antenna_v = []

    for ii in range(len(residual_list_v)):
        rname1 = "m{}v".format(str(residual_list_v[ii][0]).zfill(3))
        rname2 = "m{}v".format(str(residual_list_v[ii][1]).zfill(3))
        offending_antenna_v.append([rname1,rname2])

      
        
        
    return offending_antenna_v


def offending_antenna_idx_h(residual_list_v):
    
    offending_antenna_v = []

    for ii in range(len(residual_list_v)):
        rname1 = "m{}h".format(str(residual_list_v[ii][0]).zfill(3))
        rname2 = "m{}h".format(str(residual_list_v[ii][1]).zfill(3))
        offending_antenna_v.append([rname1,rname2])

        
    return offending_antenna_v


def matrix_(Att_cross_h,max_res_cross ) : 
    
    #max_res_cross = stats.zscore(max_res_cross)
    
    matrix=  np.nan*np.ones((64,64))
    
    


    Att_cross_h = np.array(Att_cross_h)

    a = Att_cross_h[:,0]
    empty = []

    for i in a: 
        if i not in empty:
            # attena x
            attena_name = i
            attenna_idx = int(attena_name[1:-1])
            index = np.where(a==i)[0]
            res_ = [(max_res_cross[val]) for val in index] # max residuals

            # attenna y
            The_Other_attenna_ = [Att_cross_h[val] for val in index]
            The_Other_attenna = [The_Other_attenna_[i][1] for i in range(len(The_Other_attenna_))]
            The_Other_attenna_idx = np.array([int(The_Other_attenna[i][1:-1]) for i in range(len(The_Other_attenna))])

            empty.append(i)

            for j in range(len(The_Other_attenna_idx)) :
                matrix[attenna_idx,The_Other_attenna_idx[j]] = (res_[j])
                
                
                
                
    return matrix
    
    
# Threshold detect for receivers with artefacts





def artefact_spectra(cp_loaded_auto,vis_auto,d_auto,offending_antenna,Title):
    
    
    
    show_fold_selection = True
    #ch_fold = np.s_[7000:25000]
    fold_factor_list = 64
    dp_start = 2

    
    # Compute inputs with artefacts above threshold
    bad_recs = []

    index_auto = []
    index_cross = []

    empty = []

    for rec in offending_antenna:

        a= np.where(np.array(np.array(cp_loaded_auto)) ==  rec)[0]

        index_auto.append(a[0])
   
    
    # Number of visibilities to plot per figure:
    vis_per_fig = 3

    # Determine number of plots
    n_plts = np.ceil(len(offending_antenna)/vis_per_fig).astype(int) # ceil.. takes the nearest whole number, cp_load = number of atifact
    
    if n_plts == 1 :
        
        fig, ax2 = plt.subplots(figsize=(8,4))
            
        for i in range(len(offending_antenna)):
        
            idx =  index_auto[i]
            cp_tup = cp_loaded_auto[idx,:]
            cp_mean = np.abs(np.mean(vis_auto[dp_start:,:,idx], axis=0))


            ax2.plot(d_auto.freqs/1e6, 10*np.log10(cp_mean), label=str(cp_tup))
            ax2.set_xlabel('Frequency [MHz]')
            ax2.set_ylabel('Power [dB]')
            ax2.set_title(str(Title))
            ax2.legend(loc = 'lower center')

        
        
        
    if n_plts > 1: 
        
        fig, ax = plt.subplots(n_plts, 1, figsize=(8,n_plts*3))
    
        for i in range(len(offending_antenna)):
        
            idx =  index_auto[i]
            cp_tup = cp_loaded_auto[idx,:]
            cp_mean = np.abs(np.mean(vis_auto[dp_start:,:,idx], axis=0))

            ax[i//vis_per_fig].plot(d_auto.freqs/1e6, 10*np.log10(cp_mean), label=str(cp_tup))
            ax[i//vis_per_fig].set_xlabel('Frequency [MHz]')
            ax[i//vis_per_fig].set_ylabel('Power [dB]')
            ax[i//vis_per_fig].set_title(str(Title))
            ax[i//vis_per_fig].legend(loc = 'lower center')



        
    

    plt.tight_layout()
    plt.show()
    
    
    


def trial(a):
    return a+1