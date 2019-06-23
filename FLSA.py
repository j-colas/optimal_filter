
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2019
Function Library for Stream Analysis 

essential functions for Gatti & Manfredi optimal filtering
algorithm 

@author: Jules Colas
"""

#%%##### imports
import numpy             as np
import scipy.signal      as signal


#%%##### exponential pulse 
def model1(t,a1,t11,t12,t01):
    t_var1 = np.dtype(np.float128)
    t_var1 = t - t01
    pulse1 = np.heaviside(t_var1, 1.)
    ind1 = np.where(pulse1 != 0.)
    pulse1[ind1] = -1 * (np.exp(-t_var1[ind1]/t11)-np.exp(-t_var1[ind1]/t12))
    den = np.max(np.abs(pulse1))
    if den == 0 : # to avoid / by 0
        return np.float32(a1*pulse1)
    else:
        return np.float32(a1*pulse1/den)
    
#%%##### power spectrum density estimation
def psd(x,fs,n,scale='density'):
    return signal.welch(x,fs,window='boxcar',nperseg=n,noverlap=None,nfft=None,return_onesided=True, scaling=scale)


#%%#####  load data function
def load_data(file_name,dtype,mode):    
    return np.memmap(file_name,dtype,mode)

#%%##### 
def make_of(noise_psd,model,n_window,fs):
    f           = noise_psd[0]
    noise_psd   = noise_psd[1]
    tf_model    = rfft(model)
    tf_model[0] = 0
    d    = n_window/2
    h    = d/np.sum(np.abs(tf_model)**2/noise_psd) # normalization coef. 
    idM  = np.argmax(tf_model) 
    H    = h*np.conj(tf_model)*np.exp(-1j*2*np.pi*f*idM/fs)/noise_psd 
    hi   = irfft(H)
    hid  = zeropadarray(hi) # 0 padding 
    hids = smoothD(hid)  # smoothing discontinuity
    H_OF = rfft(hids)
    return H_OF

def zeropadarray(x,M = 0):
    N  = np.size(x)
    K  = int(N/2)
    Zm   = np.zeros((N))
    x1 = x[0:K]
    x2 = x[K::]
    return np.concatenate((x2,Zm,x1))

def smoothD(x,L=500):
    N = np.size(x)
    K = int(N/4)
    dL = int(L/4) 
    t = np.linspace(0,dL,dL)
    smooth = np.ones((2*K))
    smooth[K::]      = 0
    smooth[K-dL:K:]  = np.cos(2*np.pi*t/L) 
    sm = np.concatenate((smooth[::],smooth[::-1]))
    return sm*x

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def matched_filter(file,nwin,H):
    
    data   = load_data(file,np.float32,'r')
    n0     = data.size 
    data   = np.append(np.zeros(int(nwin/2)),data)
    n      = data.size
    n_win  = 2*nwin
    Ndec   = int(n/n_win)
    Rest   = n-Ndec*n_win
    dn     = int(n_win-Rest)
    data   = np.append(data,np.zeros(dn)) # zero padding
    
    a = 0; res = np.array([],dtype=np.float32)
    for i in range(2*Ndec+1):
        x     = data[a:a+n_win]
        a    += nwin
        xfilt = np.real(irfft(H*rfft(x)))
        xgood = np.float32(xfilt[int(nwin/2):int(3*nwin/2):])
        res = np.append(res,xgood)
    if res.size > n0:
        res = res[:n0]
        
    filename = file+'_result.bin'
    print("filename saved = ",filename)
    out   = open(filename,'wb')
    out.seek(0)
    res.tofile(out)
    out.close()
    return

def trigger(tresh, file, n_win):
    num_loc   = 0
    width_min = 80 # points 
    
    loc = list(); amp1 = list();
    data = load_data(file,np.float32,'r') 
    i = np.where(data>tresh)[0]
    r_alpha = ranges(i)
    for n in r_alpha:
        w = n[1]-n[0]
        if w > width_min:
            loc.append(n[0]+np.argmax(data[n[0]:n[1]]))
            num_loc += 1
            amp1.append(np.max(data[n[0]:n[1]]))
   
    # élimination des détections doubles
    del_i = list()
    for i in range(len(loc)-1):
        d = loc[i+1]-loc[i]
        if d < 200:
            if amp1[i]<amp1[i+1]:
                del_i.append(i)
            else:
                del_i.append(i+1)
                
    loc = np.asarray(loc)
    amp1 = np.asarray(amp1)
    cpt = 0
    for i in del_i:
        loc = np.delete(loc,(i-cpt),axis=0)
        amp1 = np.delete(amp1,(i-cpt),axis=0)
        cpt += 1  
          
    np.save(file[:5]+'_loc_trigger.npy',loc)
    np.save(file[:5]+'_amp_trigger.npy',amp1)

    print("num. pulses = ",num_loc)
    return;   
    
def find_amp(file,loc_file,models,nwin,J,fs): 
    
    a = np.array([],np.float32)
    chi2 = np.array([],np.float32)
    
    loc  = np.load(loc_file)
    data = load_data(file,np.float32,'r')
    Ls = loc.size
    for j in range(Ls):
        locplus = loc[j]+int(nwin/2)
        if locplus < data.size:
            
            istart = loc[j]-int(nwin/2)
            n = np.arange(istart,istart+nwin,1)
            
            V    = np.fft.rfft(data[n])[1::]
            at = np.zeros(len(models))
            for k in range(len(models)): 
                S    = np.fft.rfft(models[k])[1::]
                num  = np.sum(np.real(np.conj(S)*V)/J[1::])
                den  = np.sum(np.real(np.conj(S)*S)/J[1::])
            
                at[k]=num/den
            
            a2 = at[np.argmax(np.abs(at))]
            
            c2 = np.sum(np.abs(V-a2*S)**2/J[1::])*(2/(fs*nwin))
            a = np.append(a,np.float32(a2))
            chi2 = np.append(chi2,np.float32(c2))
        else :
            continue
    np.savetxt(file[:5]+"_chi2Amp.txt",[a,chi2])
    #return a,chi2

def noiseFromLoc(data,loc,n_win):
    '''
    extract noise from data given trigged events location (size of analysis window needed)
    '''
    mask = np.ones(np.shape(data))
    www  = int(n_win/2)
    for i in range(np.size(loc)):
        mask[loc[i]-www:loc[i]+www] = np.zeros(np.shape(mask[loc[i]-www:loc[i]+www]))
        
    
    # FIND CONSECUTIVE RANGES OF ONES
    index = np.arange(0,np.size(data)) 
    rng   = np.asarray(ranges(mask*index))
    # FOR ALL RANGES DO ...
    deli = list()
    for j in range(len(rng)):
        rng_size  = rng[j][1]-rng[j][0]
        # S'IL Y A LA PLACE DE FAIRE TENIR AU MOINS UNE FENETRE D'ANALYSE
        if rng_size >= n_win:
            # ESTIMATE THE NUMBR OF NOISE WINDOW
            nwin = np.floor(rng_size/n_win)
            # RESIDUAL 
            resn = rng_size-nwin*n_win
            rng[j][0] += int(np.floor(resn/2)) 
            rng[j][1] -= int(np.ceil(resn/2))
        else:
            # PAS DE BRUIT, PAS DE BRUIT...
            deli.append(j)
            continue
    cpt = 0
    for i in deli:
        rng = np.delete(rng,(i-cpt),axis=0)
        cpt += 1
    
    noise = list()
    itot  = np.array([],dtype=int)

    for j in range(len(rng)):
        r1 = int(rng[j][0])
        r2 = int(rng[j][1])
     
        if r1 < 0 : r1 = 0
        if r2 > data.size : r2 = data.size
     
        idx    = np.arange(r1,r2) 
        itot = np.append(itot,idx)
        noise  = np.append(noise,data[idx])
    
    mask2 = np.ones(np.shape(mask))
    mask2[itot] = 0
    return mask,noise,mask2

def find_noise_amp(noise,nwin,model,J,npulse=1000):

    a = np.array([],np.float32)
    chi2 = np.array([],np.float32)
    
    N = noise.size
    nshift = int(N/nwin)
    if nshift>npulse:
        nshift=npulse
    istart = 0
    for j in range(nshift): #1
            
            n = np.arange(istart,istart+nwin,1)
            istart += nwin
            V    = np.fft.rfft(noise[n])[1::]
         
            S = rfft(model)[1::]
    
            num  = np.sum(np.real(np.conj(S)*V)/J[1::])
            den  = np.sum(np.real(np.conj(S)*S)/J[1::])
                
            a2 = num/den 
            c2 = np.sum(np.abs(V-a2*S)**2/J[1::])*(2/(nwin))
            a = np.append(a,np.float32(a2))
            chi2 = np.append(chi2,np.float32(c2))
                
    return a,chi2


def fft(x):
    return np.fft.fft(x)
def ifft(q):
    return np.fft.ifft(q)
def rfft(x):
    return np.fft.rfft(x)
def irfft(q):
    return np.fft.irfft(q)





