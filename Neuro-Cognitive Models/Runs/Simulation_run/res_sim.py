#!/home/a.ghaderi/.conda/envs/envjm/bin/python
# Model 2
import pystan 
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
import utils

data = pd.read_csv('sim1.csv')      #loading dateset

mis = np.where((data['n200lat']<.05)|(data['n200lat']>.250))[0] # missing data for n200lat
obs = np.where((data['n200lat']>.05)&(data['n200lat']<.250))[0] # observation and missing data for n200lat

N_mis = mis.shape[0]  # number of missing data
N_obs = obs.shape[0]   # number of observed data

modelfile = '../../stans/sim/res_sim.stan' #reading the model span
f = open(modelfile, 'r')
model_wiener = f.read()
sm = pystan.StanModel(model_code=model_wiener)# Compile the model stan

ncohers = 2  #Number of coherence conditions
nspats = 2   #Number of spatial conditions
nconds = 4   #Number of conditions
y = data['y'].to_numpy()
cond_coher = data['cond_coher'].to_numpy()
cond_spat = data['cond_spat'].to_numpy()
conds = data['conds'].to_numpy()
n200lat = data['n200lat'].to_numpy()

#set inistial data for molde span
data_winner =  {'N_obs':N_obs,      #Number of trial-level observations
                'N_mis':N_mis,      #Number of trial-level mising data
                'ncohers':ncohers,  #Number of coherence conditions
                'nspats':nspats,    #Number of spatial conditions
                'nconds':nconds,    #Number of conditions
                'y':np.concatenate([y[obs],y[mis]]),     #acc*rt in seconds for obervation and missing data
                'cond_coher':np.concatenate([cond_coher[obs],cond_coher[mis]]),     #Coherence index for each trial 
                'cond_spat':np.concatenate([cond_spat[obs],cond_spat[mis]]),        #sptial index for each trial 
                'conds':np.concatenate([conds[obs],conds[mis]]),        #sptial index for each trial
                'n200lat_obs':n200lat[obs]};    #n200 latency for each trial observation

# setting MCMC arguments
niter = 10000
nwarmup = 4000
nchains = 1
thin = 1

initials = [] # initial sampling
for c in range(0, nchains):
    chaininit = {
        'delta': np.random.uniform(.1, .5, size=ncohers),
        'alpha': np.random.uniform(.5, 1.),
        'res': np.random.uniform(.01, .02, size=nspats),     
        'n200sub': np.random.uniform(.1, .2, size=nconds),
        'lambda': np.random.uniform(.01, .2),
        'n200lat_mis': np.random.uniform(.1, .2, size = N_mis)
    }
    initials.append(chaininit)    
      
# Train the model and generate samples
fit = sm.sampling(data=data_winner, iter=niter, chains=nchains, warmup=nwarmup, thin=thin, init=initials)

utils.to_pickle(stan_model=sm, stan_fit=fit, save_path='../../save/sim/sim3_res.pkl')

