#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:23:59 2019

@author: jules
"""

import numpy as np 
import FLSA as flsa
import matplotlib.pyplot as plt
import time
from scipy import stats  
from multiprocessing import Pool


def gen_data(fname):
    #print('DATA GENERATION')
    [rays,rates] = data_params
    noise = 'wgn' # = White Gaussian Noise
    out = open(fname,'wb')
    out.seek(0,2)   
    for i in range(M):
        y = np.zeros((N),dtype=np.float32)
        for j in range(len(rays)):
            K = int(rates[j]*N)   
            for k in range(K):
                pos = np.random.randint(10,N-n_win/2)
                d = flsa.model1(x,rays[j],t_rise,t_decay,pos)
                y += d
        
        if noise == 'wgn':
            y += np.random.normal(0,std_noise,y.size) 
        
        y.tofile(out)
    out.close()
    
def mp_gen_datafiles(fnames, cpu_number):
    t_begin = time.time()
    p = Pool(cpu_number)
    p.map(gen_data,data_files)
    p.close()
    p.join()
    t_gen = time.time()-t_begin
    print("time_gen_data : %.1fs"%t_gen)
    
def gen_template():
    shift = np.linspace(-20,20,21) # time shift 
    mdls = list() # list of models
    for i in range(shift.size):
        model = flsa.model1(np.arange(n_win),1,t_rise,t_decay,n_win/3+shift[i])
        mdls.append(model)
    model = mdls[int(shift.size/2)+1]
    return model,mdls

def estim_psd_noise():
    noise = np.random.normal(0,std_noise,200*N) 
    psd_noise = flsa.psd(noise,1,n_win)
    return psd_noise

def matched_filter(fname):
    flsa.matched_filter(fname,n_win,H)
    
def trigger(fname):
    flsa.trigger_mp(tresh,fname,n_win) # trigger function


def chi2_fit(fname):
    flsa.find_amp_mp(fname,fname+'_loc_trigger.txt',mdls,n_win,psd_noise[1],1)

def mp_filter(fnames,cpu_number):
    p = Pool(cpu_number)
    p.map(matched_filter,fnames)
    p.close()
    p.join()
#%%### INITIALISATION
cpu_number = 3

N = 4000  # number of points / sample
M = 1000   # number of sample

rays  = [10,20,30,40]#,80]#,30,50]
rates = [1/3000,1/2000,1/1000,1/1000]#,1/1000]#,1/500,1/500]

# pulses params
global t_rise,t_decay
t_rise = 4
t_decay = 12

global data_params
data_params = [rays,rates]

global n_win
n_win = 200   # size of the window for analysis

global std_noise
std_noise = 5

global x 
x = np.arange(N) # time vector (points)

data_files = ['data1','data2','data3']
mp_gen_datafiles(data_files, cpu_number)


# TEMPLATE DEFINITION
model,mdls = gen_template()

# NOISE ESTIMATION
psd_noise = estim_psd_noise()

# TRANSFER FUNCTION INITIALIZATION
global H
H = flsa.make_of(psd_noise,model,n_win,1,100)

S        = flsa.rfft(model)
coef     = (2/(np.size(S[1::])))
sigma_OF = np.sqrt(1/(np.sum(np.abs(S[1::])**2*coef/np.abs(psd_noise[1][1::]))))
print("theoretical resolution = %.2f"%round(sigma_OF,2))

print('MATCHED FILTERING')
t_begin = time.time()
p = Pool(cpu_number)
p.map(matched_filter,data_files)
p.close()
p.join()
t_mfilter = time.time()-t_begin
print("time_mfilter : %.1fs"%t_mfilter)
    

global tresh
tresh   = 5*sigma_OF
print("threshold = %.2f"%round(tresh,2)) 


#%%
resName = []
for f in data_files:
    resName.append(f+"_result.bin")

p = Pool(cpu_number)
p.map(trigger,resName)
p.close()
p.join()

#%%

### CHI2 FIT OF TRIGGED EVENTS
p = Pool(cpu_number)
p.map(chi2_fit,data_files)
p.close()
p.join()



#%%
chi2 = np.array([])
amp = np.array([])
ampTrig = np.array([])
for f in data_files:
    chi2amp = np.loadtxt(f+"_chi2Amp.txt")
    amp = np.append(amp,chi2amp[0][::])
    chi2 = np.append(chi2,chi2amp[1][::])
    ampTrig = np.append(ampTrig,np.loadtxt(f+"_amp_trigger.txt"))
#%%
amp_noise = np.array([])
chi2_noise = np.array([])
for f in data_files:
    loc = np.loadtxt(f+'_loc_trigger.txt')
    loc = loc.astype(int)
    data = np.memmap(f,dtype=np.float32,mode='r')
    noise = flsa.noiseFromLoc(data,loc,n_win)[1] # get noise from data given the loc of trigged events
    ampn,chin = flsa.find_noise_amp(noise,n_win,model,psd_noise[1],int(amp.size/len(rays))) # CHI2 fit of noise events
    amp_noise = np.append(amp_noise, ampn)
    chi2_noise = np.append(chi2_noise, chin)
    
    
ampTot = np.append(amp,amp_noise)

m, s  = stats.norm.fit(amp_noise)
print("baseline resolution = %.2f"%round(s,2))

#%%
a = 11/std_noise
b = 0
c = 1.2*(int(n_win/2)+1)
lamp = np.linspace(np.min(ampTot),np.max(ampTot),1000)
cut = a*lamp**2+b*lamp+c
cutf = a*amp**2+b*amp+c
icut = np.where(chi2<cutf)[0]


#%%
data = np.memmap(data_files[0],dtype='float32',mode='r')[:N]
dres = np.memmap(data_files[0]+"_result.bin",dtype='float32',mode='r')[:N]

loc = np.loadtxt(data_files[0]+"_loc_trigger.txt")
loc = loc[np.where(loc<N)]
loc = loc.astype(int)


#%%
plt.close('all')

plt.figure()
plt.subplot(311)
plt.title(str(int(amp.size))+' events')
plt.plot(data,color='k',ls='-',alpha=.1)
plt.plot(dres,'r-')
plt.plot(loc,dres[loc],marker='o',ls='',mfc='r',color='k',alpha=.4)
plt.axhline(tresh,color='b',lw=2.5,ls="--",alpha=.3)
#plt.subplot(412)

plt.subplot(312)
plt.plot(amp_noise,chi2_noise,marker='o',ls='',ms=2,color='k',mfc='gray',alpha=.01)
plt.plot(amp,chi2,marker='o',ls='',ms=2,color='r',mfc='r',alpha=.01)
plt.axhline(H.size/2,color='b',lw=2.5,ls="--",alpha=.3)
plt.plot(lamp,cut,color='gray',lw=1.5,ls="--",label='cut',alpha=.7)

plt.yscale('log')
plt.subplot(313)
plt.axvline(0,color='g',lw=2.5,ls="--",alpha=0.2)
for r in rays:
    plt.axvline(r,color='g',lw=2.5,ls="--",alpha=0.2)
plt.hist(amp_noise,bins=200,histtype='step',color='gray',range=(ampTot.min(),ampTot.max()))
plt.hist(ampTrig,bins=200,histtype='step',color='b',range=(ampTot.min(),ampTot.max()),alpha=.2)
plt.hist(amp,bins=200,histtype='step',color='r',range=(ampTot.min(),ampTot.max()),alpha=.2)
plt.hist(amp[icut],bins=200,histtype='step',color='k',range=(ampTot.min(),ampTot.max()),alpha=1)

