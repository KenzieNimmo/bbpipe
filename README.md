# bbpipe
Basic burst properties pipeline

Given a list of burst IDs, filterbank files, mask files and burst times (in the filterbank file) this pipeline will compute the burst width in time and frequency using an autocorrelation analysis, the time of arrival and centroid of frequency (alongside independent measurements of the time width and frequency width) using a 2D Gaussian fit, the scintillation bandwidth, and the fluence and energy of the burst. 

