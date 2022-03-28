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
 
