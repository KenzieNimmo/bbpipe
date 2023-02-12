"""
A full polarisation burst properties pipeline

Kenzie Nimmo 2022

"""
import numpy as np
import matplotlib.pyplot as plt
from load_file import load_archive
import pandas as pd
from presto import filterbank
from tqdm import tqdm
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, fit_report
from pol_prof import get_profile
import matplotlib.gridspec as gridspec
from parallactic_angle import parangle as pa
from astropy.coordinates import Angle
from presto import filterbank

def archive_toa(arfile,dm,stokesi_prof,starttime,fil_toa,t_samp_fil,t_downsamp_fil,f_downsamp=1):
    """
    """
    answer='n'
    crop=[0,0]
    while answer=='n':
        archive,extent,tsamp=load_archive(arfile,extent=True,rm=0,dm=dm,remove_baseline=True, tscrunch=t_downsamp_fil,fscrunch=f_downsamp)

        if round( tsamp/t_samp_fil)!=1:
            if t_samp_fil < tsamp:
                tdown=int(np.ceil(tsamp/t_samp_fil))
                
                while len(stokesi_prof)%tdown!=0:
                    stokesi_prof=stokesi_prof[:-1]
                stokesi_prof=stokesi_prof.reshape(len(stokesi_prof)//tdown, tdown).sum(axis=1)
                t_samp_fil = tsamp
            elif tsamp < t_samp_fil:
                t_downsamp_fil = int(round(t_samp_fil/tsamp))
                archive,extent,tsamp=load_archive(arfile,extent=True,rm=0,dm=dm,remove_baseline=True, tscrunch=t_downsamp_fil,fscrunch=f_downsamp)
        print("Downsamp factor "+str(t_downsamp_fil))
        print("Time res is "+str(tsamp)+"s")
        #Stokes I profiles in filterbank and archive data products
        if crop[1]==0:
            crop[1]=archive.shape[2]
        stokesi_prof_ar=np.mean(archive[0,:,:]+archive[1,:,:],axis=0)[int(crop[0]):int(crop[1])]
        stokesi_prof_fil=stokesi_prof
        
        #TOA relative to peak in filterbank
        peakbin_fil = np.argmax(stokesi_prof_fil)
        peak_fil=starttime+(peakbin_fil * tsamp)/(3600*24.)
        deltat_fil = (fil_toa -peak_fil)*24*3600./tsamp #bins
        TOAbin_fil = (fil_toa-starttime)*24*3600/tsamp

        
        #determine TOA bin in archive file
        peakbin_ar=np.argmax(stokesi_prof_ar)
        TOAbin=int(peakbin_ar+deltat_fil)
        
        print(TOAbin)
        plt.plot(stokesi_prof_ar,color='r')
        plt.scatter(TOAbin,stokesi_prof_ar[TOAbin],color='k',marker='x')
        plt.show()

        answer_ds=input("Do you want to downsample in time? (y/n)")
        if answer_ds=='n':
            answer=input("has it identified the burst accurately? (y/n)")
            if answer=='n':
                crop=input("Provide a restricted time range containing the burst (beginbin,endbin)").split(',')
                #stokesi_prof_ar=stokesi_prof_ar[int(crop[0]):int(crop[1])]
            if answer=='y':
                TOAbin = int(TOAbin+int(crop[0]))
        if answer_ds=='y':
            t_downsamp_fil *=2
            answer='n'
            
    return archive, extent, TOAbin, tsamp

def downsamp(ds,tdown=1,fdown=1):
    tdown=int(tdown)
    fdown=int(fdown)

    if fdown!=1:
        ds=ds.reshape(ds.shape[0]//fdown, fdown,ds.shape[-1]).sum(axis=1)

    dst=False
    if tdown!=1:
        while dst==False:
            try:
                ds=ds.reshape(ds.shape[0], ds.shape[-1]//tdown, tdown).sum(axis=2)
                dst=True
            except:
                dst=False
                ds=ds[:,:-1]
    return ds

def scaleoff(archive,beginbin,endbin):
    """
    """
    off=np.concatenate((archive[:,:,0:beginbin],archive[:,:,endbin:]),axis=2)
    for chan in range(archive.shape[1]):
        archive[0,chan,:]-=np.mean(off[0,chan,:])
        off[0,chan,:]-=np.mean(off[0,chan,:])
        archive[0,chan,:]/=np.std(off[0,chan,:])
        archive[1,chan,:]-=np.mean(off[1,chan,:])
        off[1,chan,:]-=np.mean(off[1,chan,:])
        archive[1,chan,:]/=np.std(off[1,chan,:])
        archive[2,chan,:]-=np.mean(off[2,chan,:])
        off[2,chan,:]-=np.mean(off[2,chan,:])
        archive[2,chan,:]/=np.std(off[2,chan,:])
        archive[3,chan,:]-=np.mean(off[3,chan,:])
        off[3,chan,:]-=np.mean(off[3,chan,:])
        archive[3,chan,:]/=np.std(off[3,chan,:])
    return archive


def gaussian(x,a,x0,sigma):
    return a*np.exp(-np.power((x - x0)/sigma, 2.)/2.)
    
def faraday_spec(Qspec,Uspec,freqs,RM,Q=False,U=False):
    """
    freqs in MHz

    """

    lambdas = 299792458.0/(freqs*1e6)
    vals=[]
    for l in range(len(lambdas)):
        c=np.cos(-2*RM*lambdas[l]**2)
        s=np.sin(-2*RM*lambdas[l]**2)
        if Q==True and U == False:
            Uspec=np.zeros_like(Uspec)
        elif Q==False and U==True:
            Qspec = np.zeros_like(Qspec)
        val = (Qspec[l]+1.0j*Uspec[l])*(c+1.0j*s)
        vals = np.append(vals,val)

    return np.abs(np.sum(vals))/len(lambdas)

def rmsynth(ds,begintime,endtime,freqs,rm,delay):
    """
    ds is the full polarisation dynamic spectrum (shape pol,freq,time)
    begintime and endtime are the begin and end bins to chopp the burst out for the analysis (integers)
    freqs is an array of frequencies matching the length of ds.shape[1]
    rm is an array of min RM, max RM and steps in RM for the search
    delay is the instrumental delay in nanoseconds to remove from the data before RM synthesis
    """
    I=ds[0,:,:]+ds[1,:,:]
    Iprof=np.mean(I,axis=0)
    
    Q = 2*ds[2,:,:]
    U =	-2*ds[3,:,:]

    remove_delay=np.exp(-2*1j*(freqs*1e6)*delay*1e-9*np.pi)

    Qc=np.zeros_like(Q)
    Uc=np.zeros_like(U)
    for f in range(Q.shape[1]):
        Qspec = Q[:,f]
        Uspec = U[:,f]
        lin=Qspec+1j*Uspec
        lin*=remove_delay
        Qc[:,f]=lin.real
        Uc[:,f]=lin.imag
        

    Q = Qc[:,begintime:endtime]
    U = Uc[:,begintime:endtime]
    Qoff = np.concatenate((Qc[:,0:begintime],Qc[:,endtime:]),axis=1)
    Uoff = np.concatenate((Uc[:,0:begintime],Uc[:,endtime:]),axis=1)
    
    for i in range(Uoff.shape[0]):
        if np.std(Uoff[i,:])!=0:
            U[i,:]-=np.mean(Uoff[i,:])
            Uoff[i,:]-=np.mean(Uoff[i,:])
            U[i,:]/=np.std(Uoff[i,:])
        if np.std(Qoff[i,:])!=0:
            Q[i,:]-=np.mean(Qoff[i,:])
            Qoff[i,:]-=np.mean(Qoff[i,:])
            Q[i,:]/=np.std(Qoff[i,:])

    Qspec = np.mean(Q,axis=1)
    Uspec = np.mean(U,axis=1)

    RMs=np.arange(rm[0],rm[1],rm[2])
    faraday_spectrum = np.zeros_like(RMs)
    for r in tqdm(range(len(RMs))):
        faraday_spectrum[r]=faraday_spec(Qspec,Uspec,freqs,RMs[r])
        
    predict_width = 2*np.sqrt(3)/((3e8/(np.min(freqs)*1e6))**2-(3e8/(np.max(freqs)*1e6))**2)
    plt.plot(RMs,faraday_spectrum)
    try:
        popt,pcov = curve_fit(gaussian,RMs,faraday_spectrum,p0=[np.max(faraday_spectrum),RMs[np.argmax(faraday_spectrum)],predict_width])
        print("Best RM is", popt[1])
        print("Width is", popt[2], "predicted width is", predict_width)
        plt.plot(RMs,gaussian(RMs,*popt))
        RMbest=popt[1]
    except RuntimeError:
        print("Optimal fit could not be found")
        print("Taking peak as best RM")
        print("Best RM is", RMs[np.argmax(faraday_spectrum)])
        RMbest=RMs[np.argmax(faraday_spectrum)]
    except ValueError:
        print("Optimal fit could not be found")
        print("Taking peak as best RM")
        print("Best RM is", RMs[np.argmax(faraday_spectrum)])
        RMbest=RMs[np.argmax(faraday_spectrum)]
    plt.show()
    return RMbest

def Q_func(params,freq,p=False):
    if p==False:
        delay=params['delay']
        offset=params['offset']
        A = params['A']
        RM = params['RM']
    else:
        delay=params['delay']
        offset=params['offset_p']
        A = params['A_p']
        RM = params['RM_p']
    return A*np.cos(2*(((299792458.0)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))

def U_func(params,freq,p=False):
    if p==False:
        delay=params['delay']
        offset=params['offset']
        A = params['A']
        RM = params['RM']
    else:
        delay=params['delay']
        offset=params['offset_p']
        A = params['A_p']
        RM = params['RM_p']
    return A*np.sin(2*(((299792458.0)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))

def linear_pol(params,freq=None,Qdata=None,Udata=None,freq_p=None,Qdata_p=None,Udata_p=None,weights=None,weights_p=None):
    delay=params['delay']
    offset=params['offset']
    A = params['A']
    RM = params['RM']
    Q = A*np.cos(2*(((299792458.0)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))
    U = A*np.sin(2*(((299792458.0)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))
    if Qdata_p is not None:
        #pulsar
        offset_p = params['offset_p']
        A_p = params['A_p']
        RM_p = params['RM_p']
        Qp = A_p*np.cos(2*(((299792458.0)**2/(freq_p*1e6)**2)*(RM_p) + (freq_p*1e6)*delay*np.pi + offset_p))
        Up = A_p*np.sin(2*(((299792458.0)**2/(freq_p*1e6)**2)*(RM_p) + (freq_p*1e6)*delay*np.pi + offset_p))
        if weights_p is not None:
            residQp = (Qdata_p - Qp)*weights_p
            residUp = (Udata_p -Up)*weights_p
        else:
            residQp = Qdata_p - Qp
            residUp = Udata_p -Up
    if weights is not None:
        residQ = (Qdata - Q)*weights
        residU = (Udata - U)*weights
    else:
        residQ = Qdata - Q
        residU = Udata - U
        
    
    
    if Qdata_p is not None:
        return np.concatenate((residQ,residU,residQp,residUp))
    else:
        return np.concatenate((residQ,residU))

def QU_fit_plot(freqs_p,Qpspec,Upspec,weights_p,params,freqs,Qspec,Uspec,weights):
    """
    """
    fig,axes=plt.subplots(2,figsize=[8,8],sharex=True)

    axes[0].scatter(freqs_p,Qpspec,c=weights_p,cmap=plt.cm.get_cmap('Purples'))
    axes[0].plot(freqs_p,Q_func(params,freqs_p,p=True),'r',lw=3)
    axes[1].scatter(freqs_p,Upspec,c=weights_p,cmap=plt.cm.get_cmap('Purples'))
    axes[1].plot(freqs_p,U_func(params,freqs_p,p=True),'r',lw=3)
    plt.text(0.16,1.1,"PULSAR fit for delay  %.2f ns"%(params['delay'].value*1e9),transform=axes[0].transAxes)
    plt.show()

    fig,axes=plt.subplots(2,figsize=[8,8],sharex=True)

    axes[0].scatter(freqs,Qspec,c=weights,cmap=plt.cm.get_cmap('Purples'))
    axes[0].plot(freqs,Q_func(params,freqs),'r',lw=3)
    axes[1].scatter(freqs,Uspec,c=weights,cmap=plt.cm.get_cmap('Purples'))
    axes[1].plot(freqs,U_func(params,freqs),'r',lw=3)
    plt.text(0.16,1.1,"Fit for RM %.2f, fit for delay  %.2f seconds"%(params['RM'].value,params['delay'].value*1e9),transform=axes[0].transAxes)
    plt.show()
    return

def QUspec(archive,begintime,endtime):
    """
    """
    Qspec= np.mean(2*archive[2,:,begintime:endtime],axis=1)
    Uspec = np.mean(-2*archive[3,:,begintime:endtime],axis=1)
    Ispec = np.mean(archive[0,:,begintime:endtime]+archive[1,:,begintime:endtime],axis=1)

    Ispec = np.ma.masked_where(Ispec==0,Ispec)
    Qspec = np.ma.masked_where(Ispec==0,Qspec)
    Uspec = np.ma.masked_where(Ispec==0,Uspec)

    Qdata = Qspec/np.sqrt(Qspec**2+Uspec**2)
    Udata = Uspec/np.sqrt(Qspec**2+Uspec**2)

    weights = np.zeros_like(Qdata)
    for val in range(len(Qspec)):
        if np.sqrt(Qspec[val]**2+Uspec[val]**2) !=0:
            weights[val] = (np.sqrt(Qspec[val]**2+Uspec[val]**2))
        else:
            weights[val] = 0

    return Qdata, Udata, weights

def plot_pol(Iprof,Ltrue,Vprof,pdf):
    """
    """
    fig = plt.figure(figsize=(8, 5))
    rows=2
    cols=1
    widths = [1]
    heights = [1, 3]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
    x=np.arange(0,len(Iprof),1)
    cmap = plt.cm.gist_yarg

    ax1.imshow(pdf,cmap=cmap,aspect='auto',interpolation='nearest',origin='lower',extent=[x[0],x[-1],-90,90])
    ax2.step(x,Iprof,color='k',where='mid')
    ax2.step(x,Ltrue,color='r',where='mid')
    ax2.step(x,Vprof,color='b',where='mid')

    plt.show()
    return 

if __name__ == "__main__":
    # first we have the inputs needed

    #****bursts*****#
    burstids=np.arange(1)+53
    
    #where are the archive files stored
    indir_ar='/data1/kenzie/M81_monitoring/bursts/Jan142022/archives/calib/'

    #what DM?
    DM=87.7527

    #what delay (nanoseconds)?
    # give a delay measured using either noise diode or pulsar observations
    delay=0.676406751983799 #ns
    
    #what RMs to search?
    rm=[-200,200,0.5] #min max steps
    
    RMbest=-29.8

    #load in the properties dataframe
    props_df_file = '/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/Jan14_burst_properties.final.csv'
    props_df = pd.read_csv(props_df_file, index_col=0)
    #burstids=props_df.sort_values(by =['S/N'],ascending=[False]).index.values.tolist()

    #downsample in frequency?
    fds=np.zeros(len(burstids))+16
    
    # load in the data pkl file
    data_df_file='/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/Jan14_bursts_data.pkl'
    data_df = pd.read_pickle(data_df_file)

    # telescope properties
    longitude =(06. +53./60. + 00.99425/3600.)
    latitude =Angle('50d31m29.39459s') #Angle

    #source properties
    RA_frb = Angle('09h57m54.69935s')#Angle
    dec_frb = Angle('68d49m00.8529s')#Angle

    #if we want to amend an already existing polarimetry props dataframe, load it in here:
    existing_pol_prop_df = '/data1/kenzie/M81_monitoring/bursts/Jan142022/fullpol/Jan14_fullpol.csv'
    #if we want to amend an already existing polarimetry data pickle file, load it in here:
    existing_pol_data_pkl = '/data1/kenzie/M81_monitoring/bursts/Jan142022/fullpol/Jan14_fullpol_data.pkl'

    if existing_pol_prop_df!=None:
        df = pd.read_csv(existing_pol_prop_df, index_col=0)
    else:
        df = pd.DataFrame(index=burstids, columns=['Archive_file','Downsample_f','Delay_init [ns]','RM_rmsynth','Delay [ns]','RM_QU','Delay_sigma [ns]','RM_QU_sigma','Pulsar_RM_fixed','Phase_offset','Phase_offset_error','Phase_offset_pulsar','Phase_offset_pulsar_error','Par_angle','Linear_frac','Linear_frac_err','Circular_frac','Circular_frac_err','Abs_Circular_frac','PA_mean','Linear_PA_fit_dof','Linear_PA_fit_chisq','Linear_PA_fit_redchisq'])

    if existing_pol_data_pkl!=None:
        poldata_df = pd.read_pickle(existing_pol_data_pkl)
    else:
        poldata_df = pd.DataFrame(index=burstids, columns=['Fullpol_ds', 'Fullpol_ds_TOAbin', 'Iprof', 'Lprof', 'Vprof', 'PA_pdf', 'PA', 'PAerr', 'freqs', 'tres', 'fres','tcent','Time_axis'])

    outdir='/data1/kenzie/M81_monitoring/bursts/Jan142022/fullpol/'
    #name for final csv file containing all results
    out_filename='Jan14_fullpol.csv'
    #name for final pkl file containing the data products
    outdataname='Jan14_fullpol_data.pkl'
    
    # pulsar calibrator data
    pulsar=False
    pulsar_arfile = '/data1/kenzie/M81_monitoring/bursts/Jan142022/archives/calib/B47_fullpol.l-10'
    if pulsar_arfile!=None:
        pulsar=True
        pulsar_dm = 87.7527#57.1420
        pulsar_rm = -29.8#81.5
        pulsar_archive,pulsar_extent,pulsar_tsamp=load_archive(pulsar_arfile,extent=True,rm=0,dm=pulsar_dm,remove_baseline=True,fscrunch=16)
        plt.plot(np.mean(pulsar_archive[0,:,:]+pulsar_archive[1,:,:],axis=0))
        plt.show()
        answer=input("Give begin and end bin of burst (begin,end)").split(',')
        beginbin=int(answer[0])
        endbin=int(answer[1])
        pulsar_archive=scaleoff(pulsar_archive,beginbin,endbin)
        freqs_p = np.linspace(pulsar_extent[2],pulsar_extent[3],pulsar_archive.shape[1])
        Qpspec,Upspec,weights_p=QUspec(pulsar_archive,beginbin,endbin)
        
    for bn,burst in enumerate(burstids):
        print("*** Loading in data for burst B%s ***"%burst)
        arfile=indir_ar+'B'+str(burst)+'_fullpol.l-10'

        df.loc[burst,'Archive_file']=arfile
        df.loc[burst,'Downsample_f']=int(fds[bn])
        
        # some properties from the Stokes I analysis
        fil=filterbank.FilterbankFile(props_df.loc[burst,'Fil_file'])
        tstart=fil.header['tstart']
        tsamp=fil.header['tsamp']*int(props_df.loc[burst,'Time_downsample'])
        fil_toa = tstart + ((props_df.loc[burst,'TOA']))/(24*3600.*1000)
        ds = data_df.loc[burst,'Dynamic_spectrum']
        #downsamp the filterbank data
        ds=downsamp(ds,tdown=int(props_df.loc[burst,'Time_downsample']))
        stokesi_prof = np.mean(ds,axis=0)
        
        starttime = tstart + data_df.loc[burst,'start_time']/(24*3600.) 
        t_downsamp_fil = int(props_df.loc[burst,'Time_downsample'])

        #load in archive data and find TOA bin
        archive,extent,tsamp_ar=load_archive(arfile,extent=True,rm=0,dm=DM,remove_baseline=True)
        t_downsamp_fil=int((fil.header['tsamp']/tsamp_ar)*t_downsamp_fil)

        if t_downsamp_fil==0: t_downsamp_fil=1
        archive, extent, TOAbin, tsamp = archive_toa(arfile,DM,stokesi_prof,starttime,fil_toa,tsamp,t_downsamp_fil,f_downsamp=int(fds[bn]))

        #use this TOA and burst width to cut out the burst and compute the Faraday spectrum -- RM synthesis.
    
        print("*** Performing RM synthesis on burst B%s ***"%burst)
        print("*** Assuming a delay of %.2f ns and a circular basis ***"%delay)
        print("*** Searching from %d to %d rad/m^2 in steps of %.1f rad/m^2 ***"%(rm[0],rm[1],rm[2]))
        freqs = np.linspace(extent[2],extent[3],archive.shape[1])
        begintime= int(TOAbin - (2*props_df.loc[burst,'Time_width']/(1000*tsamp)))
        endtime= int(TOAbin + (2*props_df.loc[burst,'Time_width']/(1000*tsamp)))
        cent_f = np.argmin(np.abs(props_df.loc[burst,'FOA']-freqs))
        beginfreq= cent_f - 2*int(props_df.loc[burst,'Freq_width']/(freqs[1]-freqs[0]))
        endfreq = cent_f + 2*int(props_df.loc[burst,'Freq_width']/(freqs[1]-freqs[0]))

        if beginfreq<0:
            beginfreq=0
        
        archive=scaleoff(archive[:,beginfreq:endfreq,:],begintime,endtime)
        freqs=freqs[beginfreq:endfreq]
        RMguess = rmsynth(archive,begintime,endtime,freqs,rm,delay)

        df.loc[burst,'Delay_init [ns]']=delay
        df.loc[burst,'RM_rmsynth']=RMguess
        
        print("*** Performing QU fitting on burst B%s ***"%burst)
        Qspec,Uspec,weights=QUspec(archive,begintime,endtime)
        params = Parameters()
        params.add('delay', value=delay*1e-9,min=-10e-9,max=10e-9)
        params.add('offset', value=0,min=-50, max=50)
        params.add('A', value=1)
        params['A'].vary = False
        params.add('RM',value=RMguess,min=-1000,max=1000)
        if pulsar==True:
            params.add('offset_p', value=0,min=-50, max=50)
            params.add('A_p', value=1)
            params.add('RM_p',value=pulsar_rm)
            params['RM_p'].vary = False
            params['A_p'].vary = False
            out = minimize(linear_pol, params, kws={"freq": freqs, "Qdata": Qspec, "Udata": Uspec, "freq_p": freqs_p, "Qdata_p": Qpspec, "Udata_p": Upspec, "weights":weights, "weights_p":None})
        if pulsar==False:
            out = minimize(linear_pol, params, kws={"freq": freqs, "Qdata": Qspec, "Udata": Uspec, "weights":weights})

        print(fit_report(out))

        df.loc[burst,'Delay [ns]']=(out.params['delay'].value)*1e9
        df.loc[burst,'RM_QU']=out.params['RM'].value
        df.loc[burst,'Delay_sigma [ns]']=(out.params['delay'].stderr)*1e9
        df.loc[burst,'RM_QU_sigma']=out.params['RM'].stderr
        df.loc[burst,'Pulsar_RM_fixed']=out.params['RM_p'].value
        df.loc[burst,'Phase_offset']=out.params['offset'].value
        df.loc[burst,'Phase_offset_error']=out.params['offset'].stderr
        df.loc[burst,'Phase_offset_pulsar']=out.params['offset_p'].value
        df.loc[burst,'Phase_offset_pulsar_error']=out.params['offset_p'].stderr
                                                   
        QU_fit_plot(freqs_p,Qpspec,Upspec,weights_p,out.params,freqs,Qspec,Uspec,weights)

        answer=input("Are the QU-fits reliable? (y/n) ")
        if answer=='y':
            df.loc[burst,'RM-reliable']='y'
            rm_touse=out.params['RM'].value
        else:
            df.loc[burst,'RM-reliable']='n'
            #bestburst=int(props_df[['S/N']].idxmax())
            rm_touse=RMbest#float(df.loc[bestburst,'RM_QU'])
            #print("Using the RM measured for the highest S/N burst in this sample, B%s (RM=%.2f), to compute the polarisation properties"%(bestburst, rm_touse))
            print("Using previous RM measurement (RM=%.2f) to compute the pol properties"%rm_touse)
            
        print("*** Correcting the parallactic angle of burst B%s ***"%burst)
        filfile = props_df.loc[burst,'Fil_file']
        fil=filterbank.FilterbankFile(filfile)
        tstart=fil.header['tstart']
        bursttoa_mjd = tstart + (props_df.loc[burst,'TOA']/1000.)/(24*3600)
        parang = pa(bursttoa_mjd,longitude,RA_frb,dec_frb,latitude) 

        df.loc[burst,'Par_angle']=parang
                                                   
        print("*** Calculating polarisation fractions of burst B%s ***"%burst)
        
        StokesI_ds, I, Ltrue, V, PA, x, PAmean, dof, chisq_k, redchisq, PAerror, pdf, weights,burstbeg,burstend, linfrac, linfracerr, abscircfrac, circfrac, circfracerr  = get_profile(archive,freqs,out.params['delay'].value,rm_touse,parang,burstbeg=begintime, burstend=endtime, tcent=int((endtime-begintime)//2 + begintime), PAoffset=None)

        df.loc[burst,'Linear_frac']=linfrac
        df.loc[burst,'Linear_frac_err']=linfracerr
        df.loc[burst,'Circular_frac']=circfrac
        df.loc[burst,'Circular_frac_err']=circfracerr
        df.loc[burst,'Abs_Circular_frac']=abscircfrac
        df.loc[burst,'PA_mean']=PAmean
        df.loc[burst,'Linear_PA_fit_dof']=dof
        df.loc[burst,'Linear_PA_fit_chisq']=chisq_k
        df.loc[burst,'Linear_PA_fit_redchisq']=redchisq
        
        print("*** Plotting final polarisation profile of burst B%s ***"%burst)
        plot_pol(I,Ltrue,V,pdf)

        df.to_csv(outdir+out_filename)

        print("*** Writing data of burst B%s to file %s***"%(burst,outdataname))
        
        poldata_df.loc[burst,'Fullpol_ds']=archive
        poldata_df.loc[burst,'Fullpol_ds_TOAbin']=TOAbin
        poldata_df.loc[burst,'tcent']=int((endtime-begintime)//2 + begintime)
        poldata_df.loc[burst,'Iprof']=I
        poldata_df.loc[burst,'Lprof']=Ltrue
        poldata_df.loc[burst,'Vprof']=V
        poldata_df.loc[burst,'Time_axis']=x
        poldata_df.loc[burst,'PA_pdf']=pdf
        poldata_df.loc[burst,'PA']=PA
        poldata_df.loc[burst,'PAerr']=PAerror
        poldata_df.loc[burst,'freqs']=freqs
        poldata_df.loc[burst,'tres']=tsamp
        poldata_df.loc[burst,'fres']=freqs[1]-freqs[0]
        
        poldata_df.to_pickle(outdir+outdataname)
