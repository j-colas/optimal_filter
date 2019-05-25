# Optimal Filter
Optimal Filtering based on Gatti, E. &amp; Manfredi, P.F. Riv. Nuovo Cim. (1986) 9: 1. https://doi.org/10.1007/BF02822156

## How it works ? 
Just clone the repository and execute the file : optimal_filtering_analysis.py 

## Algorithm step 

### [1] DATA GENERATION
Exponential pulses with white gaussian noise (all parameters can be adjusted easily in the beginning of the script).

### [2] OPTIMAL FILTER DEFINITION 
Definition of the OF transfer function which depend of the pulse template and noise estimation. 

### [3] PROCESSING THE DATA (Filtering + Trigger)
Trigger threshold sets to 3 times theoretical resolution. Can be improved with respect of the desired output.

### [4] QUALITY CUT
A simple cut based on the quality of the fit (chi squared) to reject pile-up event and errors. 

### [5] PLOT THE RESULTS 

![alt stream visualisation](https://raw.githubusercontent.com/j-colas/optimal_filter/master/stream_visu.png)

![alt chi squared vs. amp](https://raw.githubusercontent.com/j-colas/optimal_filter/master/chi2_vs_amp.png)

![alt amplitude spectrums](https://raw.githubusercontent.com/j-colas/optimal_filter/master/amp_spectrum.png)

