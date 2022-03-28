"""
BBpipe is a basic burst properties pipeline

Kenzie Nimmo 2022

Given burst filterbanks, masks and time (in seconds) into the filterbanks where the burst occurs, this will calculate:
- the burst extent in time and frequency using an ACF analysis
- the burst TOA using the centroid of a 2D Gaussian fit and barycentring using the DM, telescope position and source position
- the burst fluence, peak flux density, S/N, and energy using the known distance to the source

Output the burst properties in a pandas dataframe, which can be used to plot the family plot of bursts or create LaTeX tables of burst properties for publication.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ACF_funcs import autocorr_2D, lorentz, autocorr
from scipy.optimize import curve_fit, leastsq
from lmfit import minimize, Parameters, fit_report, Model
from load_file import load_filterbank
import pandas as pd
from interactive_sub_burst_identifier_2d import identify_bursts
import burst_2d_Gaussian as fitter
from scipy.stats import chisquare
from radiometer import radiometer

def loaddata(filename, t_burst, DM=0, maskfile=None, fullpol=False, window=10):
    """
    Reads in the filterbank file into a numpy array, applies the mask
    and outputs the burst dynamic spectrum as a numpy array.
    Inputs:
    - filename: Name of filterbank filename with full path
    - t_burst: time of the burst in seconds into the filterbank file
    - maskfile: text file containing a list of frequency channels to zap (get it from pazi command)
    - fullpol: True/False. full polarisation filterbanks are not currently supported in this analysis. Stokes I only.
    - window: window in [ms] around burst to extract for analysis (default +-10ms)
    Outputs:
    - Stokes I dynamic spectrum
    - Off burst dynamic spectrum
    - time resolution in seconds
    - frequency resolution in MHz
    """
    if fullpol==False:
        ds,dsoff,extent,tsamp,begbin=load_filterbank(filename,dm=DM,fullpol=False,burst_time=t_burst)
        StokesI_ds = np.zeros_like(ds)
        StokesI_off = np.zeros_like(dsoff)
        #removing bandpass
        for fr in range(ds.shape[0]):
            StokesI_ds[fr,:]=convert_SN(ds[fr,:],dsoff[fr,:])
            StokesI_off[fr,:]=convert_SN(dsoff[fr,:],dsoff[fr,:])

        # frequency resolution
        freqres=(extent[3]-extent[2])/ds.shape[0]
        # frequency array
        frequencies = np.linspace(extent[2],extent[3],ds.shape[0])

        if maskfile!=None:
            maskchans=np.loadtxt(maskfile,dtype='int')
            maskchans = [StokesI_ds.shape[0]-1-x for x in maskchans]
            StokesI_ds[maskchans,:]=0
            StokesI_off[maskchans,:]=0

    if fullpol==True:
        raise ValueError('Full pol filterbank data is not currently supported.')

    return StokesI_ds, StokesI_off, tsamp, freqres, begbin, frequencies

def window(ds,window,tsamp):
    """
    chop out a window around the burst (peak of the profile)
    """
    profile = np.mean(ds,axis=0)
    begin=np.argmax(profile)-int(window/(1000*tsamp))
    end=np.argmax(profile)+int(window/(1000*tsamp))
    burstds=ds[:,begin:end]
    return burstds,begin
def downsamp(ds,tdown=1,fdown=1):
    if fdown!=1:
        ds=ds.reshape(ds.shape[0]//fdown, fdown,ds.shape[-1]).sum(axis=1)
    if tdown!=1:
        ds=ds.reshape(ds.shape[0], ds.shape[-1]/tdown, tdown).sum(axis=2)
    return ds

def convert_SN(burst_prof, off_prof):
    burst_prof-=np.mean(off_prof)
    off_prof-=np.mean(off_prof)
    burst_prof/=np.std(off_prof)
    return burst_prof

def twodacf(burstid, ds, timeres, freqres, acf_load=None, save=False, plot=False, outdir='./'):
    """
    Performs a 2D ACF on the burst dynamic spectrum.
    Fits Gaussians to the broad shape, to measure the burst time and frequency extent.

    Inputs:
    - burstid is the burst identifier/name
     - ds is a numpy array containing the Stokes I dynamic spectrum of the burst
     - timeres and freqres are the time and frequency resolution of the dynamic spectrum in seconds and MHz respectively.
     - acf_load is a numpy file containing a previously calculated 2D ACF of the dynamic spectrum ds, if it exists. Default is to calculate the ACF.
     - if you want to save the ACF as a numpy array, save=True. Default not to save.
     - if you want diagnostic plots plotted to screen, set plot=True, else it will save the plots to the output directory, outdir.
    """
    if acf_load == None:
        ACF=autocorr_2D(ds)
    else:
        ACF=np.load(acf_load)

    if save==True:
        np.save(str(outdir)+'/B'+str(burstid)+'_2D_ACF.npy', ACF)

    ACF/=np.max(ACF)
    ACFmasked = np.ma.masked_where(ACF==np.max(ACF),ACF) # mask the zero-lag spike

    ACFtime = np.sum(ACF,axis=0)
    ACFfreq = np.sum(ACF,axis=1)
    ACFtime = np.ma.masked_where(ACFtime==np.max(ACFtime),ACFtime)
    ACFfreq = np.ma.masked_where(ACFfreq==np.max(ACFfreq),ACFfreq)

    #make the time and frequency axes
    time_one = np.arange(1,ds.shape[1],1)*timeres*1000 #ms
    times = np.concatenate((-time_one[::-1],np.concatenate(([0],time_one))))
    freq_one = np.arange(1,ds.shape[0],1)*freqres
    freqs = np.concatenate((-freq_one[::-1],np.concatenate(([0],freq_one))))

    #1D Gaussian fitting to ACFtime and ACF freq
    try:
        poptt, pcovt = curve_fit(gaus, times, ACFtime, p0=[1,0,np.max(times)])
        poptf, pcovf = curve_fit(gaus, freqs, ACFfreq, p0=[1,0,np.max(freqs)])
    except:
        print("Could not do basic fitting")
        return 0,0,0,0,0,0

    if plot == True:
        #we want to see if this initial fit is a good initial guess for the 2D fit
        fit=False
        while fit==False:
            fig,ax=plt.subplots(2)
            ax[0].plot(times, ACFtime)
            ax[0].plot(times,gaus(times,*poptt))
            ax[0].set_ylabel('ACF time')
            ax[1].plot(freqs, ACFfreq)
            ax[1].plot(freqs,gaus(freqs,*poptf))
            ax[1].set_ylabel('ACF freq')
            plt.show()

            answer = raw_input("Are you happy with this fit? (y/n): ")
            if answer == 'n':
                q = raw_input("Does this burst need downsampled in time? (y/n) ")
                if q == 'y':
                    return 0,0,0,0,0,0
                else:
                    guess = input("Give an initial guess (time_amp,time_mean,time_sigma,freq_amp,freq_mean,freq_sigma) ")
                    poptt, pcovt = curve_fit(gaus, times, ACFtime, p0=[guess[0],guess[1],guess[2]])
                    poptf, pcovf = curve_fit(gaus, freqs, ACFfreq, p0=[guess[3],guess[4],guess[5]])
            if answer == 'y':
                fit=True

    #if plot=False we just trust that this is a good initial guess and proceed.
    #2D Gaussian fitting
    timesh, freqs_m = np.meshgrid(times, freqs)
    timesh = timesh.astype('float64')
    freqs_m = freqs_m.astype('float64')

    #defining the parameters
    params = Parameters()
    params.add('amplitude', value=1)
    params.add('xo',value=0,vary=False)
    params.add('yo',value=0,vary=False)
    params.add('sigma_x',value=poptt[2],min=poptt[2]-0.2*poptt[2], max=poptt[2]+0.2*poptt[2])
    params.add('sigma_y',value=poptf[2],min=poptf[2]-0.2*poptf[2], max=poptf[2]+0.2*poptf[2])
    params.add('theta',value=0)

    out = minimize(twoD_Gaussian_fit, params, kws={"x_data_tuple": (timesh,freqs_m), "data": ACFmasked})
    print("*** Gaussian fit to 2D ACF for burst %s ***"%burstid)
    print("Times (x) are in milliseconds and Frequencies (y) are in MHz")
    print(fit_report(out))

    data_fitted = twoD_Gaussian((timesh, freqs_m), out.params['amplitude'],out.params['xo'],out.params['yo'],out.params['sigma_x'],out.params['sigma_y'],out.params['theta'])
    data_fitted=data_fitted.reshape(len(freqs),len(times))

    #residuals
    ACFtimeresid = ACFtime-np.sum(data_fitted,axis=0)
    ACFfreqresid = ACFfreq-np.sum(data_fitted,axis=1)

    #plot
    fig = plt.figure(figsize=(8, 8))
    rows=3
    cols=3
    widths = [3, 1,1]
    heights = [1,1,3]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    cmap = plt.cm.gist_yarg

    ax1 = fig.add_subplot(gs[0,0]) # Time ACF
    ax1.plot(times,ACFtime,color='k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_xlim(times[0],times[-1])
    ax1.plot(times,np.sum(data_fitted,axis=0),color='purple')

    ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # Time ACF residuals
    ax2.plot(times,ACFtimeresid,color='k')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlim(times[0],times[-1])

    ax3 = fig.add_subplot(gs[2,0],sharex=ax2) # 2D ACF
    T,F=np.meshgrid(times, freqs)
    ax3.imshow(ACFmasked,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times[0],times[-1],freqs[0],freqs[-1]))
    ax3.contour(T,F,data_fitted,4, colors='r', linewidths=.5)
    ax3.set_ylabel('Freq lag [MHz]')
    ax3.set_xlabel('Time lag [ms]')

    ax4 = fig.add_subplot(gs[2,1],sharey=ax3) #Freq ACF residuals
    ax4.plot(ACFfreqresid,freqs,color='k')
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylim(freqs[0],freqs[-1])
    #ax4.plot(lorentz(freqs,result_freq.params['gamma'],result_freq.params['y0'],result_freq.params['c']),freqs,color='orange')

    ax5 = fig.add_subplot(gs[2,2],sharey=ax4) #Freq ACF
    ax5.plot(ACFfreq,freqs,color='k')
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set_ylim(freqs[0],freqs[-1])
    ax5.plot(np.sum(data_fitted,axis=1),freqs,color='purple')

    plt.savefig(str(outdir)+'/B%s_2d_acf_burst.pdf'%burstid,dpi=300,format='pdf')
    if plot==True:
        plt.show()

    #return sigma_time, sigma_time_error, sigma_frequency, sigma_frequency_error, drift theta
    return np.abs(out.params['sigma_x'].value),out.params['sigma_x'].stderr, np.abs(out.params['sigma_y'].value),out.params['sigma_y'].stderr,out.params['theta'].value,out.params['theta'].stderr

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def twoD_Gaussian(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):
    (x,y)=x_data_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def twoD_Gaussian_fit(params,x_data_tuple,data):
    amplitude=params['amplitude']
    xo=params['xo']
    yo=params['yo']
    sigma_x = params['sigma_x']
    sigma_y = params['sigma_y']
    theta = params['theta']

    fit=twoD_Gaussian(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta)

    resid = data.ravel()-fit
    return resid

def fit_gaus(burstid,ds,frequencies,tsamp,twidth_guess,fwidth_guess, plot=False,outdir='./'):
    """
    Fits a 2D gaussian to the burst dynamic spectrum

    Inputs:
    - burstid: burst identifier
    - ds: dynamic spectrum of the burst
    - frequencies: array of frequencies matching the y-axis of the dynamic spectrum
    - tsamp: sampling time of the data
    - twidth_guess and fwidth_guess are the time [ms] and frequency [MHz] guesses for the burst widths.
    """
    #if you want to plot on screen then let the user identify the subbursts to fit
    #else just use the peak of the dynamic spectrum as the initial guess
    if plot==True:
        subbursts = identify_bursts(ds,tsamp)
        time_guesses = np.array(subbursts.peak_times)
        freq_guesses = np.array(subbursts.peak_freqs)
        amp_guesses = np.array(subbursts.peak_amps)
    else:
        print("We are assuming there is only one burst component to fit a Gaussian for burst B%s"%burstid)
        time_guesses = np.array([np.argmax(ds)[1]])
        freq_guesses = np.array([np.argmax(ds)[0]])
        amp_guesses = np.array([np.max(ds)])

    # Get the times at the pixel centers in ms.
    times = (np.arange(ds.shape[1]) * tsamp + tsamp/2) * 1e3
    time_guesses = time_guesses*tsamp*1e3
    freq_guesses= np.array(map(float, freq_guesses))
    freq_guesses = freq_guesses*(frequencies[1]-frequencies[0])
    freq_guesses += frequencies[0]

    n_sbs = len(time_guesses)
    freq_std_guess = [fwidth_guess] * n_sbs
    t_std_guess = [twidth_guess] * n_sbs

    model = fitter.gen_Gauss2D_model(time_guesses, amp_guesses, f0=freq_guesses,bw=freq_std_guess, dt=t_std_guess, verbose=True)

    bestfit, fitLM = fitter.fit_Gauss2D_model(ds, times, frequencies, model)
    bestfit_params, bestfit_errors, corr_fig = fitter.report_Gauss_parameters(bestfit,fitLM,verbose=True)

    timesh, freqs_m = np.meshgrid(times, frequencies)
    chisq, pvalue = chisquare(ds, f_exp=bestfit(timesh, freqs_m),ddof=6*n_sbs, axis=None)
    print("Chi^2 and p-value:", chisq, pvalue)

    fig, res_fig = fitter.plot_burst_windows(times, frequencies,ds, bestfit, ncontour=8,res_plot=True)  # diagnostic plots
    corr_fig.savefig(outdir+'/B%s_gausfit_correlation.pdf'%burstid)
    fig.savefig(outdir+'/B%s_gausfit.pdf'%burstid)
    res_fig.savefig(outdir+'/B%s_gausfit_residuals.pdf'%burstid)
    if plot==True:
        plt.show()
    return bestfit_params, bestfit_errors

def scint_bw(burstid,ds,tcent,t_width,fres,maskfile=None,start_time=0,outdir='./',plot=False):
    """
    Creates the burst spectrum using the +-2sigma region in time, and then autocorrelates that spectrum.
    The scintillation bandwidth is measured using a Lorentzian fit to the central component.


    """
    tburst = tcent - start_time #time in milliseconds into the ds where the burst is
    tburst /= (tres*1000) # in bins
    tburst = int(tburst)
    #compute the 2sigma region
    begin_t=int(tburst-((2*t_width)/ (tres*1000)))
    end_t=int(tburst+((2*t_width)/ (tres*1000)))

    burst_ds = ds[:, begin_t:end_t]
    spectrum = np.mean(burst_ds,axis=1)

    maskchans=np.loadtxt(maskfile,dtype='int')
    maskchans = [len(spectrum)-1-x for x in maskchans]
    mask = np.ones_like(spectrum)
    mask[maskchans]=0

    ACF_one=autocorr(spectrum, len(spectrum), v=mask,zerolag=False)
    ACF=np.concatenate((ACF_one[::-1],ACF_one))
    freq_one=np.arange(1,ds.shape[0]+1,1)*fres
    freqs = np.concatenate((-freq_one[::-1],freq_one))
    #prep for scint bw fit
    ACFf_for_fit = ACF[int(len(ACF)/2.-(30/fres)):int(len(ACF)/2.+(30/fres))]
    freq_for_fit = freqs[int(len(ACF)/2.-(30/fres)):int(len(ACF)/2.+(30/fres))]

    #do scint bw fit
    gmodel = Model(lorentz)
    try:
        result_freq = gmodel.fit(ACFf_for_fit, x=freq_for_fit, gamma=1, y0=1, c=0)
        print("*** Lorentzian fit to freq ACF for burst %s ***"%burstid)
        print(result_freq.fit_report())
    except:
        print("Could not do scintillation bandwidth fitting")
        return 0,0

    fig,ax=plt.subplots(2)
    ax[0].plot(freqs,ACF,color='orange')
    ax[0].plot(freqs,lorentz(freqs,result_freq.params['gamma'],result_freq.params['y0'],result_freq.params['c']),color='green',label='Scint bw: %.2f MHz'%result_freq.params['gamma'])
    ax[1].plot(freq_for_fit,ACFf_for_fit,color='orange')
    ax[1].plot(freq_for_fit,lorentz(freq_for_fit,result_freq.params['gamma'],result_freq.params['y0'],result_freq.params['c']),color='green',label='Scint bw: %.2f MHz'%result_freq.params['gamma'])
    ax[1].legend()
    plt.savefig(outdir+'/B%s_scintbw.pdf'%burstid,format='pdf',dpi=300)
    if plot==True:
        plt.show()
    #return scint bw and scint bw errors
    return result_freq.params['gamma'].value,result_freq.params['gamma'].stderr

def compute_fluence(burstid,ds,dsoff,tcent,fcent,t_width,f_width,tres,fres,freqs,SEFD,distance=None,start_time=0,outdir='./',plot=False):
    """
    Converts burst profile to physical units

    Inputs:
    - ds is the dynamic spectrum,dsoff is the dynamic spectrum of off burst data
    - tcent, fcent are the centre time and frequency of the burst from the filterbank in ms and MHz, respectively
    - twidth, fwidth are the 1sigma width of the burst in time and frequency in ms and MHz, respectively.
    - tres and fres are the time resolution and frequency resolution in seconds and MHz, respectively.
    - freqs is the array of frequencies matching the y-axis of ds
    - start_time is the time in milliseconds from the start of the filterbank where the dynamic spectrum (ds) begins
    - SEFD is the system equivalent flux density of your telescope during your observation
    - distance is the distance to the FRB source in Mpc
    """
    #chop out the burst
    #let's do the +-2sigma region of the burst in time and frequency to compute the burst properties

    #figure out the bins where the centre of the burst is
    tburst = tcent - start_time #time in milliseconds into the ds where the burst TOA is
    tburst /= (tres*1000) # in bins
    tburst = int(tburst)

    fburst = int((fcent-freqs.min())/fres)

    #compute the 2sigma region
    begin_t=int(tburst-((2*t_width)/ (tres*1000)))
    end_t=int(tburst+((2*t_width)/ (tres*1000)))
    begin_f=int(fburst-((2*f_width)/ (fres)))
    end_f=int(fburst+((2*f_width)/ (fres)))

    if begin_f < 0:
        begin_f=0
    if end_f >= ds.shape[0]:
        end_f=ds.shape[0]-1

    burst_ds = ds[begin_f:end_f, begin_t:end_t]
    off = dsoff[begin_f:end_f,100:100+(end_t-begin_t)]

    profile_burst = np.mean(burst_ds,axis=0)
    profile_off = np.mean(off,axis=0)
    profile_full = np.mean(ds[begin_f:end_f,:],axis=0)

    #in S/N units
    profile_burst=convert_SN(profile_burst , profile_off)
    profile_full=convert_SN(profile_full , profile_off)
    profile_off=convert_SN(profile_off , profile_off)

    # convert to physical units using the radiometer equation
    bw = (end_f-begin_f)*fres #bandwidth is the bandwidth (in MHz) over which we compute the profile
    profile_burst_flux=profile_burst*radiometer(tres*1000,bw,2,SEFD)
    fluence = np.sum(profile_burst_flux*tres*1000)

    width = (end_t - begin_t) # S/N as per the definition in PRESTO
    #sigma = sum(signal-bkgd_level)/RMS/sqrt(boxcar_width)
    # width in bins
    print("S/N of burst B%s is "%(burstid)+str(np.sum(profile_burst)/np.sqrt(width)))
    print("Peak S/N of burst B%s is "%(burstid)+str(np.max(profile_burst)))
    print("Peak flux density of burst B%s is "%(burstid)+str(np.max(profile_burst_flux))+" Jy")
    print("Fluence of burst B%s is "%(burstid)+str(fluence)+" Jy ms")

    if distance !=None:
        #convert Jy ms to J s
        fluence_Jys = fluence*1e-3
        #convert Mpc to cm
        distance_lum_cm = 3.086e24*distance
        energy_spec= fluence_Jys*4*np.pi*(distance_lum_cm**2)*1e-23
        energy_iso = energy_spec * bw*1e6 # convert from spectral energy to isotropic by multiplying by the burst width in frequency in Hz
        lum_spec = energy_spec/(len(profile_burst_flux)*tres)
        print("Isotropic energy is "+str(energy_iso)+" erg")
        print("Spectral energy is "+str(energy_spec)+" erg Hz^-1")
        print("Spectral luminosity is "+str(lum_spec)+" erg s^-1 Hz^-1")

    #spectrum
    spectrum = np.mean(ds[:,begin_t:end_t],axis=1)

    #centre time array on the burst TOA
    time_array=np.arange(ds.shape[1]) - tburst
    time_array=time_array*tres*1000 #in milliseconds

    plot_ds(burstid, ds, profile_full, spectrum, time_array, freqs, begin_t,end_t,begin_f,end_f, window=t_width*6,outdir=outdir)
    if plot==True:
        plt.show()

    #return S/N, peak S/N, peak flux density, fluence, isotropic energy, spectral energy, spectral luminosity
    return np.sum(profile_burst)/np.sqrt(width), np.max(profile_burst), np.max(profile_burst_flux), fluence, energy_iso, energy_spec, lum_spec

def plot_ds(burstid, ds, profile, spectrum, time_array, freq_array, begin_t,end_t,begin_f,end_f, window=1,outdir='./'):
    """
    Plot the burst dynamic spectrum and profile

    Inputs:
    - burstid is the burst identifier
    - ds is the dynamic spectrum, with x-axis time_array (in ms) and y-axis freq_array (in MHz)
    - profile is the frequency averaged profile of ds, converted to S/N units
    - spectrum is the time-averaged (within the burst +-2sigma region) burst spectrum
    - begin_t, end_t are the begin and end time bins where the burst is identified (+-2sigma)
    - begin_f, end_f are the begin and end frequency bins where the burst is identified (+-2sigma)
    - window is the plotting region (+-) around the burst in milliseconds
    """
    fig = plt.figure(figsize=(8, 8))
    rows=2
    cols=2
    widths = [3, 1]
    heights = [1,3]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    ax1 = fig.add_subplot(gs[0,0]) # Time profile in S/N units
    ax1.plot(time_array,profile,color='k',linestyle='steps-mid')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(-window,window)
    ax1.axvline(time_array[begin_t],color='r')
    ax1.axvline(time_array[end_t],color='r')

    ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # Dynamic spectrum
    ax2.imshow(ds, aspect='auto',origin='lower',extent=[time_array[0],time_array[-1],freq_array.min(), freq_array.max()])
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Frequency [MHz]')
    ax2.axvline(time_array[begin_t],color='r')
    ax2.axvline(time_array[end_t],color='r')
    ax2.axhline(freq_array[begin_f],color='r')
    ax2.axhline(freq_array[end_f],color='r')

    ax3 = fig.add_subplot(gs[1,1],sharey=ax2) # Spectrum
    ax3.plot(spectrum, freq_array, color='k', linestyle='steps-mid')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.axhline(freq_array[begin_f],color='r')
    ax3.axhline(freq_array[end_f],color='r')

    plt.savefig(outdir+'/B%s_dynamic_spectrum.pdf'%burstid,format='pdf',dpi=300)

    return

if __name__ == "__main__":
    # first we have the inputs needed

    #****bursts*****#
    burstids=np.arange(52)+1 #B1--> B52
    #we don't have data for B2 so remove this one
    burstids=np.delete(burstids,1)

    #where are the filterbank files stored
    indir_fil='/data1/kenzie/M81_monitoring/bursts/Jan142022/filterbank/5.12us/'
    #where are the mask files stored?
    indir_masks='/data1/kenzie/M81_monitoring/bursts/Jan142022/filterbank/flags/2048chan/'
    #import text file of burst times (or alternatively create your array of burst times, matching order of burstids)
    burst_times=np.loadtxt('/data1/kenzie/M81_monitoring/bursts/Jan142022/Jan14_burst_times.txt')
    #what DM?
    DM=87.7527
    #downsample?
    tdown=np.ones_like(burstids)
    fdown=32

    #SEFD of your observations
    SEFD = 20/1.54
    #distance to your source
    distance=3.63 #Mpc

    #if we want to amend an already existing dataframe, load it in here:
    existing_df = None #name (with full path) of the csv file containing an existing df
    if existing_df!=None:
        df = pd.read_csv(existing_df, index_col=0)
    #otherwise start a new one
    else:
        #create the pandas dataframe of the information
        #'Fil_file' is the filterbank filename with full path
        #'Mask_file' is the mask filename with full path
        #'Burst_time' is the time into the filterbank file where the burst occurs
        #'Time_downsample' is the factor with which you downsample the time resolution of the data for the analysis.
        #'Time_width', 'Time_width_error','Freq_width', 'Freq_width_error' are the time and freq widths with errors from the ACF analysis (times in ms, freqs in MHz)
        #'Theta', 'Theta_err' are the ACF drift angle and error
        #'Scint_bw', 'Scint_bw_error' is the scintillation bandwidth with error
        #'ncomp' is the number of components in the burst. Note if you don't plot things to the screen it will just assume a single component.
        #'TOA','FOA' is the time and frequency on arrival of the bursts (or the centroid of time and frequency)
        # 'S/N' signal to noise boxcar, Peak_S/N is the peak S/N ratio
        # 'Peak_flux' is the peak flux density in Jy, 'Fluence' is the fluence in Jy ms
        # Eiso is the isotropic energy (erg)
        # Espec is the spectral energy (erg Hz^-1)
        # Lspec is the spectral luminosity (erg Hz^-1 s^-1)
        df = pd.DataFrame(index=burstids, columns=['Fil_file', 'Mask_file', 'Burst_time', 'Time_downsample', 'Time_width', 'Time_width_error','Freq_width', 'Freq_width_error', 'Theta','Theta_err','Scint_bw','Scint_bw_error','ncomp','TOA','FOA', 'S/N', 'Peak_S/N', 'Peak_flux', 'Fluence', 'Eiso', 'Espec', 'Lspec'])

        #let's fill in what we need to start the analysis -- fil_file, mask_file and burst_time
        for i,burst in enumerate(burstids):
            df.loc[burst,'Fil_file']=indir_fil+'B'+str(burst)+'_cDD_DM87.7527_F2048_b32_d1.fil'
            df.loc[burst,'Mask_file']=indir_masks+'B'+str(burst)+'_2048ch.flag'
            df.loc[burst,'Burst_time']=burst_times[i]

    #do you want to plot things on screen as you go through -- diagnostic plots and plots to check it's going smoothly?
    plot = True
    #for plots saved to disk, and numpy files, give an output directory
    outdir='/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/'
    #name for final csv file containing all results
    out_filename='Jan14_burst_properties.csv'


    #you may not want to analyse all the bursts so give here a list of the bursts you want to analyse from your burstids
    bursts = burstids
    for bn,burst in enumerate(bursts):
        #load in the filterbank data
        print("*** Loading in data for burst B%s ***"%burst)
        dynspec,dynspec_off,tres,fres,begin_bin,freqs=loaddata(df.loc[burst]['Fil_file'], df.loc[burst]['Burst_time'], DM=DM, maskfile=df.loc[burst]['Mask_file'], fullpol=False)
        begintime=begin_bin*tres #seconds
        twidth=0
        fwidth=0

        dynspec_orig = dynspec.copy()
        dynspec_off_orig = dynspec_off.copy()
        freqs_orig=freqs.copy()
        fres_orig=fres
        tres_orig=tres
        while twidth==0 and fwidth==0:
            if tdown[bn]!=1 or fdown!=1:
                while dynspec_orig.shape[1]%tdown[bn] !=0:
                    dynspec_orig=dynspec_orig[:,:-1]
                while dynspec_off_orig.shape[1]%tdown[bn] !=0:
                    dynspec_off_orig=dynspec_off_orig[:,:-1]

                dynspec=downsamp(dynspec_orig,tdown=tdown[bn],fdown=fdown)
                dynspec_off=downsamp(dynspec_off_orig,tdown=tdown[bn],fdown=fdown)
                #correct the frequency array for the downsampling
                min_f=freqs_orig.min() - fres/2
                max_f = freqs_orig.max() + fres/2
                new_res = (max_f-min_f)/(dynspec.shape[0])
                fres=new_res
                freqs=np.linspace((min_f+fres/2),(max_f-fres/2),dynspec.shape[0])
                #new sampling time
                tres=tres_orig*tdown[bn]

            #perform the ACF analysis
            print("*** Performing ACF analysis for burst B%s ***"%burst)
            burstds,beg_sm=window(dynspec,0.5,tres) #chop out the burst
            twidth,twidtherr,fwidth,fwidtherr,theta,thetaerr = twodacf(burst, burstds, tres, fres, acf_load=None, save=True, plot=plot, outdir=outdir)
            if twidth==0 and fwidth==0:
                print("Downsampling burst B%s by a factor of 2 in time"%burst)
                tdown[bn]*=2



        df.loc[burst,'Time_downsample']=tdown[bn]
        df.loc[burst,'Time_width']=twidth
        df.loc[burst,'Time_width_error']=twidtherr
        df.loc[burst,'Freq_width']=fwidth
        df.loc[burst,'Freq_width_error']=fwidtherr
        df.loc[burst,'Theta']=theta
        df.loc[burst,'Theta_err']=thetaerr

        #perform the 2D gaussian fit to the dynamic spectrum
        print("*** Performing 2D Gaussian fit analysis for burst B%s ***"%burst)
        burstds,beg_sm=window(dynspec,5,tres)
        #output for fit is in ms and MHz
        gausfit,gausfit_errors=fit_gaus(burst,burstds,freqs,tres,twidth,fwidth,plot=plot,outdir=outdir)
        #peak time in overall filterbank is begin_bin + beg_sm + peak bin in burstds
        #figure out when beginning of chopped out dynamic spectrum is
        beg_window=(begintime + beg_sm*tres) #seconds
        #need to somehow account for multi-components
        gausfit=np.array(gausfit)
        gausfit_errors=np.array(gausfit_errors)
        df.loc[burst,'ncomp']=gausfit.shape[0]
        #add appropriate columns to the dataframe for each component
        for comp in range(gausfit.shape[0]):
            if 'Comp%s_ctime'%(int(comp+1)) not in df.columns:
                new_df = pd.DataFrame(columns=['Comp%s_ctime'%(int(comp+1)), 'Comp%s_ctime_error'%(int(comp+1)),'Comp%s_cfreq'%(int(comp+1)), 'Comp%s_cfreq_error'%(int(comp+1)), 'Comp%s_wtime'%(int(comp+1)), 'Comp%s_wtime_error'%(int(comp+1)), 'Comp%s_wfreq'%(int(comp+1)), 'Comp%s_wfreq_error'%(int(comp+1)), 'Comp%s_angle'%(int(comp+1)), 'Comp%s_angle_error'%(int(comp+1))])
                df = pd.concat([df,new_df], sort=False)
            df.loc[burst,'Comp%s_ctime'%(int(comp+1))]=gausfit[comp][1]+(beg_window*1000)
            df.loc[burst,'Comp%s_ctime_error'%(int(comp+1))]=gausfit_errors[comp][1]
            df.loc[burst,'Comp%s_cfreq'%(int(comp+1))]=gausfit[comp][2]
            df.loc[burst,'Comp%s_cfreq_error'%(int(comp+1))]=gausfit_errors[comp][2]
            df.loc[burst,'Comp%s_wtime'%(int(comp+1))]=gausfit[comp][3]
            df.loc[burst,'Comp%s_wtime_error'%(int(comp+1))]=gausfit_errors[comp][3]
            df.loc[burst,'Comp%s_wfreq'%(int(comp+1))]=gausfit[comp][4]
            df.loc[burst,'Comp%s_wfreq_error'%(int(comp+1))]=gausfit_errors[comp][4]
            df.loc[burst,'Comp%s_angle'%(int(comp+1))]=gausfit[comp][5]
            df.loc[burst,'Comp%s_angle_error'%(int(comp+1))]=gausfit_errors[comp][5]

        # need to compute a TOA and central frequency for multi-component bursts
        #just going to find the mid point between the two extreme (like 1st and last components)
        # otherwise use the tcent and fcent from the 2d fits
        if gausfit.shape[0] >1:
            TOA = ((gausfit[gausfit.shape[0]-1][1]-gausfit[0][1])/2.) +	(gausfit[0][1]+(beg_window*1000))
            FOA = (gausfit[gausfit.shape[0]-1][2]-gausfit[0][2])/2. + min(gausfit[gausfit.shape[0]-1][2],gausfit[0][2])
        else:
            TOA = gausfit[0][1]+(beg_window*1000)
            FOA = gausfit[0][2]
        df.loc[burst,'TOA']=TOA
        df.loc[burst,'FOA']=FOA

        print("*** Performing scintillation bandwidth analysis for burst B%s ***"%burst)
        print("Note we use the original frequency resolution for this (i.e. ignoring downsampling factor given above).")

        burstds_origfreq,beg_sm_origfreq=window(dynspec_orig,5,tres)
        scintbw,scintbwerr=scint_bw(burst,burstds_origfreq,TOA,twidth,fres_orig,maskfile=df.loc[burst]['Mask_file'],start_time=beg_window*1000,outdir=outdir,plot=plot)
        df.loc[burst,'Scint_bw']=scintbw
        df.loc[burst,'Scint_bw_error']=scintbwerr

        print("*** Performing fluence calculations for burst B%s ***"%burst)
        #use all this information to compute the fluence etc of the burst and make a final plot
        sn, peak_sn, peak_flux, fluence, isotropic_energy, spectral_energy, spectral_luminosity=compute_fluence(burst,burstds,dynspec_off,TOA,FOA,twidth,fwidth,tres,fres,freqs,SEFD,distance=distance,start_time=beg_window*1000,outdir=outdir,plot=plot)

        df.loc[burst,'S/N']=sn
        df.loc[burst,'Peak_S/N']=peak_sn
        df.loc[burst,'Peak_flux']=peak_flux
        df.loc[burst,'Fluence']=fluence
        df.loc[burst,'Eiso']=isotropic_energy
        df.loc[burst,'Espec']=spectral_energy
        df.loc[burst,'Lspec']=spectral_luminosity


        df.to_csv(outdir+out_filename)
