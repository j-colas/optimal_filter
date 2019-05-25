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

###############################################################################
#%%### INITIALISATION
fname = "data_ofa.bin" 
N = 10000 # number of points / sample
M = 200   # number of sample
K = 10    # number of pulse / sample

branch   = .5   # low/high ratio
low      = 25  # low amplitude / energy
high     = 60 # high amplitude / energy
stdLow   = 30  # std amplitude (low)
stdHigh  = 50  # std amplitude (high)
stdNoise = 25  # std noise (gaussian)

t_up = 20
t_down = 70

n_win = 500   # size of the window for analysis
 
x = np.arange(N) # time vector (points)
###############################################################################
#%%### [1] GENERATION OF DATA STREAM FILE (pulses+noise)
print('DATA GENERATION')

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
###############################################################################
#%%### [2] OPTIMAL FILTER
print('OPTIMAL FILTER INITIALIZATION')

# TEMPLATE DEFINITION
shift = np.linspace(-20,20,21) # time shift 
mdls = list() # list of models
for i in range(shift.size):
    model = flsa.model1(np.arange(n_win),1,t_up,t_down,n_win/3+shift[i])
    mdls.append(model)
model = mdls[int(shift.size/2)+1]

# NOISE ESTIMATION
noise = np.random.normal(0,stdNoise,100*y.size) 
psd_noise = flsa.psd(noise,1,n_win)

# TRANSFER FUNCTION INITIALIZATION
H = flsa.make_of(psd_noise,model,n_win,1)

###############################################################################
#%%### [3] PROCESSING STREAM 
print('MATCHED FILTERING')
flsa.matched_filter(fname,n_win,H)

### RESOLUTION ESTIMATION
S        = flsa.rfft(model)
coef     = (2/(np.size(S[1::])))
sigma_OF = np.sqrt(1/(np.sum(np.abs(S[1::])**2*coef/np.abs(psd_noise[1][1::]))))
print("theoretical resolution = %.2f"%round(sigma_OF,2))

#%% TRIGGER 
print("TRIGGER")

tresh   = 4*sigma_OF
print("threshold = %.2f"%round(tresh,2)) 

flsa.trigger(tresh,"res_filtre.bin",n_win) # trigger function

### CHI2 FIT OF TRIGGED EVENTS
amp2,chi2 = flsa.find_amp(fname,'loc_trigger.npy',mdls,n_win,psd_noise[1],1)

### CHI2 FIT OF NOISE EVENTS
loc = np.load('loc_trigger.npy')
data = np.memmap(fname,dtype=np.float32,mode='r')
noise = flsa.noiseFromLoc(data,loc,n_win)[1] # get noise from data given the loc of trigged events
ampn,chin = flsa.find_noise_amp(noise,n_win,model,psd_noise[1]) # CHI2 fit of noise events
# FIND BASELINE RESOLUTION
m, s  = stats.norm.fit(ampn)
print("baseline resolution = %.2f"%round(s,2))

#%%### [4] QUALITY CUT  (QUADRATIC)
# MUST BE TUNNED IF THE ANALYSIS WINDOW IS CHANGED
a = .012
b = 0
c = 1.1*(int(n_win/2)+1)
lamp = np.linspace(np.min(amp2),np.max(amp2),1000)
cut = a*lamp**2+b*lamp+c
cutf = a*amp2**2+b*amp2+c
icut = np.where(chi2<cutf)[0]

#%%### [5] PLOT RESULTS
data = np.memmap(fname,dtype=np.float32,mode='r')[:N]
filt = np.memmap("res_filtre.bin",dtype=np.float32,mode='r')[:N]
loc  = np.load('loc_trigger.npy')
amp  = np.load('amp_trigger.npy')
idx  = np.where(loc<= N)[0]

#### PLOT PARAMETERS 
W   = 5.2
L   = 4.5
DPI = 180
legend_size = 10
log_histo = False
log_chi2  = False

plt.figure(num=1,figsize=(W,L),dpi=DPI,edgecolor='k',clear=True)
plt.plot(data,'k',label='data')
plt.plot(filt,'r',label='OF')
plt.plot(loc[idx],amp[idx],'oc',label='detection')

plt.xlabel('time')
plt.ylabel('signal')
plt.title('Stream Visualisation')
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.savefig('stream_visu.png')

plt.figure(num=2,figsize=(W,L),dpi=DPI,edgecolor='k',clear=True)
#plt.grid(which='both')
bins = 100
plt.axvline(low,color='g',linewidth=1.5,linestyle="--")
plt.axvline(high,color='g',linewidth=1.5,linestyle="--",label='true')
plt.hist(amp2,bins,histtype='step',log=log_histo,color='k',density=False, range=(np.min(ampn),np.max(amp2)),label='detect. [no cut]')
plt.hist(ampn,bins,histtype='step',log=log_histo,color='b',density=False, range=(np.min(ampn),np.max(amp2)),label='noise')
plt.hist(amp2[icut],bins,histtype='step',log=log_histo,color='r',density=False, range=(np.min(ampn),np.max(amp2)), label='detect. [cut]')
plt.xlabel('fitted amp.')
plt.ylabel('count')
plt.ylim(ymin=1)
plt.title('Amplitude Spectrum')
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.savefig('amp_spectrum.png')

plt.figure(num=3,figsize=(W,L),dpi=DPI,edgecolor='k',clear=True)
plt.plot(ampn,chin,'b.',alpha=.3,label='noise')
plt.plot(amp2,chi2,'k.',alpha=.1,label='detect.')
plt.axhline(int(n_win/2+1),color='r',linewidth=2,linestyle="--",label='DoF')
plt.plot(lamp,cut,'r',label='cut')
if log_chi2: 
    plt.yscale('log')
plt.grid(which='both')

plt.xlabel('fitted amp.')
plt.ylabel(r'$\chi^2$')
plt.legend(fontsize=legend_size)
plt.title(r'$\chi^2$ vs. Amplitude')
plt.tight_layout()
plt.savefig('chi2_vs_amp.png')
