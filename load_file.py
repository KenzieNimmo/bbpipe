
import numpy as np
from presto import filterbank


def load_archive(archive_name,rm=0,dm=0,tscrunch=None,fscrunch=None,remove_baseline=True,extent=False,model=False,pscrunch=False,cent=False):
    import psrchive
    print("Loading in archive file {}".format(archive_name))
    archive = psrchive.Archive_load(archive_name)
    if cent==True:
        archive.centre()

    archive.tscrunch()
    if pscrunch == True:
        archive.pscrunch()

    ardm =archive.get_dispersion_measure()
    ardd = archive.get_dedispersed()
    arfc = archive.get_faraday_corrected()
    arrm = archive.get_rotation_measure()


    if ardd==True:
        print("Archive file is already dedispersed to a DM of {} pc/cc".format(ardm))
        dm = dm - ardm
        


    if remove_baseline == True:
        #remove the unpolarised background -- note this step can cause Stokes I < Linear
        archive.remove_baseline()

    if tscrunch!=None and tscrunch!=1:
        archive.bscrunch(tscrunch)

    if fscrunch!=None and fscrunch!=1:
        archive.fscrunch(fscrunch)

    if dm!=0:
        if ardd==True and dm==0:
            pass
        if ardd==True and ardm!=dm:
            print("Dedispersing the data to a DM of {} pc/cc".format(dm+ardm))
        else: print("Dedispersing the data to a DM of {} pc/cc".format(dm))
        archive.set_dispersion_measure(dm)
        archive.set_dedispersed(False)
        archive.dedisperse()

    if rm!=0:
        if arfc==True:
            print("Faraday derotating the data to a RM of {} rad/m^2".format(rm+arrm))
        else: print("Faraday derotating the data to a RM of {} rad/m^2".format(rm))
        archive.set_rotation_measure(rm)
        archive.set_faraday_corrected(False)
        archive.defaraday()

    ds = archive.get_data().squeeze()
    w = archive.get_weights().squeeze()
    
    if model==False:
        if len(ds.shape)==3:
            weights = np.ones_like(ds)
            weights[:,w==0,:] = 0
            ds = np.ma.masked_where(weights==0,ds)
            ds = np.flip(ds,axis=1)
        else: 
            weights = np.ones_like(ds)
            weights[w==0,:]=0
            ds = np.ma.masked_where(weights==0,ds)
            ds = np.flip(ds,axis=0)

    
    if model==True:
        ds = ds

    tsamp = archive.get_first_Integration().get_duration()/archive.get_nbin()

    
    if extent==True:
        extent = [0, archive.get_first_Integration().get_duration()*1000,\
        archive.get_centre_frequency()+archive.get_bandwidth()/2.,\
        archive.get_centre_frequency()-archive.get_bandwidth()/2.]
        return ds, extent,tsamp
    else:
        return ds


def load_filterbank(filterbank_name,dm=None,fullpol=False,burst_time=None):
    fil = filterbank.FilterbankFile(filterbank_name)
    tsamp=fil.header['tsamp']
    if burst_time!=None:
        #if you know where the burst is in seconds within the filterbank file, chop out only necessary chunk of data
        burst_bin = int(burst_time/tsamp)
        if burst_bin < fil.nspec//2:
            off = fil.get_spectra(fil.nspec//2 + 100,(fil.nspec//2)-10)
        else:
            off = fil.get_spectra(0,(fil.nspec//2)-100)


        #consider how much delay the DM would cause
        tdel=np.abs(4.149377593360996e-3*dm*((1./np.min(fil.frequencies/1000.)**2)-(1./np.max(fil.frequencies/1000.)**2))) #seconds
        
        bt=50e-3 #burst is at 50ms unless burst is within 50ms of the start of the file
        if (burst_bin-int(50e-3/tsamp)+int(200e-3/tsamp) < fil.nspec) & (burst_bin-int(50e-3/tsamp) >= 0):
            spec = fil.get_spectra(burst_bin-int(50e-3/tsamp),int(2*tdel/tsamp))
            begbin=burst_bin-int(50e-3/tsamp)
        elif burst_bin-int(50e-3/tsamp) < 0:
            spec = fil.get_spectra(0,int(2*tdel/tsamp))
            bt = burst_bin*tsamp
            begbin=0
        else:
            dur = fil.nspec - (burst_bin-int(50e-3/tsamp))
            spec = fil.get_spectra(burst_bin-int(50e-3/tsamp),dur)
            begbin=burst_bin-int(50e-3/tsamp)
            
    else:
        spec = fil.get_spectra(0,fil.nspec)
        begbin=0

    if burst_time!=None and fullpol==True:
        raise ValueError("Definining the burst time in ful pol data is not currently supported")
    if dm!=None and fullpol==True:
        raise ValueError("If filterbank contains full polarisation information, dedispersion won't work properly")
    if dm != None:
        spec.dedisperse(dm)
        if burst_time!=None:
            off.dedisperse(dm)
            offarr=off.data
    
    arr=spec.data
    if burst_time!=None:
        #chop off the end where the DM delay creates an angled edge
        arr=arr[:,:-int(tdel/tsamp)]
    
    if fullpol==True:
        arr_reshape = arr.reshape(fil.header['nchans'],-1,4)
        arr = arr_reshape
        
    if fil.header['foff'] < 0:
        #this means the band is flipped
        arr = np.flip(arr,axis=0)
        foff = fil.header['foff']*-1
        if burst_time!=None:
            offarr = np.flip(offarr,axis=0)
    else: foff = fil.header['foff']

    #header information
    tsamp = fil.header['tsamp']
    begintime = 0
    endtime = arr.shape[1]*tsamp
    fch_top = fil.header['fch1']
    nchans = fil.header['nchans']
    fch_bottom = fch_top+foff-(nchans*foff)

    extent = (begintime,endtime,fch_bottom,fch_top)

    if burst_time==None:
        return arr, extent, tsamp
    else:
        return arr,offarr, extent, tsamp, begbin
    
