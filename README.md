# bbpipe
Basic burst properties pipeline

Given a list of burst IDs, filterbank files, mask files and burst times (in the filterbank file) this pipeline will compute the burst width in time and frequency using an autocorrelation analysis, the time of arrival and centroid of frequency (alongside independent measurements of the time width and frequency width) using a 2D Gaussian fit, the scintillation bandwidth, and the fluence and energy of the burst. 

Inputs to bbpipe.py:
 - line 505-507: give the burst IDs
 - line 510: give the input directory where your filterbank files are stored
 - line 512: give the directory where the mask files are stored (mask files are just text files containing a list of channels to zap -- determined using pazi)
 - line 514: give a list of burst times in seconds within the filterbank file
 - line 516: DM used to incoherently dedisperse all bursts
 - line 518-519: downsampling factor for time and frequency if needed. Note that the scintillation bandwidth analysis will use the intrinsic resolution of the data. 
 - line 522: SEFD of your observations
 - line 524: distance to your source in Mpc
 - line 527: if you have an existing dataframe that you want to edit or complete, inport here using pd.read_csv
 - line 551-552: change here the basename of your filterbank and mask files 
 - line 556: plot=True to have plots shown on screen (recommended to make sure things are going smoothly -- sometimes downsampling is required and so it's good to intervene) 
 - line 558: output directory for all the figures and final csv file to be saved
 - line 560: name for the final csv file
 - line 564: give here the list of bursts you actually want to analyse 
 
Example output for a single burst:
 
Fil_file             /data1/kenzie/M81_monitoring/bursts/Jan142022/... # filterbank file with path\
Mask_file            /data1/kenzie/M81_monitoring/bursts/Jan142022/... # mask file with path\
Burst_time                                                    0.559406 # time in seconds within the filterbank where the burst occurs\
Time_downsample                                                      2 # downsample factor used in the analysis\
Time_width                                                   0.0965569 # width in time determined by ACF analysis [ms]\
Time_width_error                                            0.00198742 # 1 sigma error on the time width \
Freq_width                                                     126.141 # width in frequency determined by ACF analysis [MHz]\
Freq_width_error                                               2.59839 # 1 sigma error on the frequency width\
Theta                                                     -0.000349832 # angle of the 2D gaussian to the ACF. Can be used to measure the burst drift. \
Theta_err                                                  2.22694e-05 # error on the angle\
Scint_bw                                                       13.1788 # scintillation bandwidth \
Scint_bw_error                                                 62.5175 # 1 sigma error on the scint bw\
ncomp                                                                1 # number of components (by selecting by eye, otherwise assume 1)\
TOA                                                            559.603 # time of arrival in milliseconds from the beginning of the filterbank file\
FOA                                                            1396.51 # central frequency of the burst\
S/N                                                            13.6473 # S/N of the burst (boxcar S/N)\
Peak_S/N                                                       6.06182 # peak S/N\
Peak_flux                                                     0.870112 # peak flux density [Jy] \
Fluence                                                       0.122017 # fluence [Jy ms]\
Eiso                                                       7.69095e+32 # Isotropic energy [erg]\
Espec                                                      1.92413e+24 # spectral energy [erg Hz^-1]\
Lspec                                                      5.07847e+27 # spectral luminosity [erg s^-1 Hz^-1]\
Comp1_ctime                                                    559.603 # central time of component 1\
Comp1_ctime_error                                             0.278083 # error\
Comp1_cfreq                                                    1396.51 # central frequency of component 1\
Comp1_cfreq_error                                              372.766 # error\
Comp1_wtime                                                  0.0698507 # time width from the 2D gaussian fit of component 1 [ms]\
Comp1_wtime_error                                             0.253845 # error\
Comp1_wfreq                                                    100.593 # frequency width from the 2D gaussian fit of component 1 [MHz]\
Comp1_wfreq_error                                              402.983 # error\
Comp1_angle                                                0.000305019 # angle of the 2D gaussian (due to drifting or incorrect DM)\
Comp1_angle_error                                           0.00364614 # error on angle\
