#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 02:39:29 2019

@author: jules
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019

DEMO OF THE OPTIMAL FILTERING ALGORITHM 
INSPIRED BY : 
Gatti, E. & Manfredi, P.F. Riv. Nuovo Cim. (1986) 9: 1. 
DOI : https://doi.org/10.1007/BF02822156

[1] DATA GENERATION 
[2] FILTER FUNCTION
[3] PROCESSING
[4] FINE TUNING OF THE RESULTS 

@author: Jules Colas 
"""

#%%### imports

import numpy as np 
import FLSA as flsa
import matplotlib.pyplot as plt

from scipy import stats  
from multiprocessing import Pool

###############################################################################
#%%###

def gen_data_file(fname):
    print('DATA GENERATION')
    [N,M,K,branch,low,high,stdLow,stdHigh,stdNoise,t_up,t_down,n_win,x] = params
    out = open(fname,'wb')
    out.seek(0,2)   
    for i in range(M):
        y = np.zeros((N),dtype=np.float32)
        for j in range(K):
            if np.random.rand()>branch:
                #a = np.random.normal(high,stdHigh,1)
                a = high
            else:
                #a = np.random.normal(low,stdLow,1)
                a = low
            pos = np.random.randint(10,N-n_win/2)
            d = flsa.model1(x,a,t_up,t_down,pos)
            y += d
        
        y += np.random.normal(0,stdNoise,y.size) 
        y.tofile(out)
    out.close()

def gen_template():
    shift = np.linspace(-20,20,21) # time shift 
    mdls = list() # list of models
    for i in range(shift.size):
        model = flsa.model1(np.arange(n_win),1,t_up,t_down,n_win/3+shift[i])
        mdls.append(model)
    model = mdls[int(shift.size/2)+1]
    return model,mdls

def estim_psd_noise():
    noise = np.random.normal(0,stdNoise,100*N) 
    psd_noise = flsa.psd(noise,1,n_win)
    return psd_noise

def matched_filter(fname):
    flsa.matched_filter(fname,n_win,H)
    
def trigger(fname):
    flsa.trigger(tresh,fname,n_win) # trigger function


def chi2_fit(fname):
    flsa.find_amp(fname,fname+'_loc_trigger.npy',mdls,n_win,psd_noise[1],1)

###############################################################################
#%%### INITIALISATION

N = 10000  # number of points / sample
M = 800   # number of sample
K = 10    # number of pulse / sample

branch   = .5   # low/high ratio
low      = 25  # low amplitude / energy
high     = 60 # high amplitude / energy
stdLow   = 30  # std amplitude (low)
stdHigh  = 50  # std amplitude (high)
stdNoise = 25  # std noise (gaussian)

t_up = 20
t_down = 70

global n_win
n_win = 500   # size of the window for analysis
 
x = np.arange(N) # time vector (points)

global params
params = [N,M,K,branch,low,high,stdLow,stdHigh,stdNoise,t_up,t_down,n_win,x]


cpu_number = 4

###############################################################################
#%%### [1] GENERATION OF DATA STREAM FILE (pulses+noise)
fnames = ['data1','data2','data3','data4']

p = Pool(cpu_number)    
result = p.map(gen_data_file,fnames)
p.close()
p.join()
    
###############################################################################
#%%### [2] OPTIMAL FILTER
print('OPTIMAL FILTER INITIALIZATION')

# TEMPLATE DEFINITION
model,mdls = gen_template()

# NOISE ESTIMATION
psd_noise = estim_psd_noise()

# TRANSFER FUNCTION INITIALIZATION
global H
H = flsa.make_of(psd_noise,model,n_win,1)

###############################################################################
#%%### [3] PROCESSING STREAM 
print('MATCHED FILTERING')
p = Pool(cpu_number)
p.map(matched_filter,fnames)
p.close()
p.join()

#%%
### RESOLUTION ESTIMATION
S        = flsa.rfft(model)
coef     = (2/(np.size(S[1::])))
sigma_OF = np.sqrt(1/(np.sum(np.abs(S[1::])**2*coef/np.abs(psd_noise[1][1::]))))
print("theoretical resolution = %.2f"%round(sigma_OF,2))

#%% TRIGGER 
print("TRIGGER")

tresh   = 3*sigma_OF
print("threshold = %.2f"%round(tresh,2)) 


resName = []
for f in fnames:
    resName.append(f+"_result.bin")

p = Pool()
p.map(trigger,resName)
p.close()
p.join()


#%%
### CHI2 FIT OF TRIGGED EVENTS

p = Pool()
p.map(chi2_fit,fnames)
p.close()
p.join()


#%%
chi2 = np.array([])
amp = np.array([])

for f in fnames:
    chi2amp = np.loadtxt(f+"_chi2Amp.txt")
    amp = np.append(amp,chi2amp[0][::])
    chi2 = np.append(chi2,chi2amp[1][::])

#%%
plt.close('all')
plt.figure()
plt.subplot(211)
plt.plot(amp,chi2,marker='o',ls='',ms=2,color='k',mfc='gray',alpha=.01)
plt.subplot(212)
plt.hist(amp[::],bins=500,histtype='step',color='k')
