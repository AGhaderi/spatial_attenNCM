#!/home/a.ghaderi/.conda/envs/envjm/bin/python
import pystan 
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
import utils

data = utils.get_data()          #loading dateset
mis = np.where((data['n200lat']<.101)|(data['n200lat']>.248))[0] # missing data for n200lat
obs = np.where((data['n200lat']>.101)&(data['n200lat']<.248))[0] # observation and missing data for n200lat
N_mis = mis.shape[0]  # number of missing data
N_obs = obs.shape[0]   # number of observed data

modelfile = '../../stans/non-boot/res_lambda_modelt.stan' #reading the model span
f = open(modelfile, 'r')
model_wiener = f.read()
sm = pystan.StanModel(model_code=model_wiener)# Compile the model stan

ncohers = 2  #Number of coherence conditions
nspats = 2   #Number of spatial conditions
nconds = 4   #Number of conditions
nparts = 15  #Number of subjects
y = data['y']
participant = data['participant']
cond_coher = data['cond_coher']
cond_spat = data['cond_spat']
conds = data['conds']
n200lat = data['n200lat']

#set inistial data for molde span
data_winner =  {'N_obs':N_obs,      #Number of trial-level observations
                'N_mis':N_mis,      #Number of trial-level mising data
                'ncohers':ncohers,  #Number of coherence conditions
                'nspats':nspats,    #Number of spatial conditions
                'nparts':nparts,    #Number of subjects
                'nconds':nconds,    #Number of conditions
                'y':np.concatenate([y[obs],y[mis]]),     #acc*rt in seconds for obervation and missing data
                'participant':np.concatenate([participant[obs],participant[mis]]),  #subject index
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
        'deltasd': np.random.uniform(0.01, 2.),
        'alphasd': np.random.uniform(.01, .1),        
        'etasd': np.random.uniform(.01, .1),
        'ressd': np.random.uniform(0.01, .02),
        'n200condsd': np.random.uniform(.01, .1),
        'lambdasd': np.random.uniform(.01, .02),
        'n200trialsd': np.random.uniform(.01, .1),        
        'deltahier': np.random.uniform(.01, 2., size=ncohers),
        'alphahier': np.random.uniform(.01, 1.),       
        'etahier': np.random.uniform(.01, .2),
        'reshier': np.random.uniform(.11, .2, size=nspats),   
        'n200cond': np.random.uniform(.11, .2, size=nconds),
        'lambdahier': np.random.uniform(.01, .02, size=ncohers),        
        'delta': np.random.uniform(1, 3, size=(nparts,ncohers)),
        'alpha': np.random.uniform(.5, 1., size=nparts),
        'res': np.random.uniform(.01, .02, size=(nparts, nspats)),     
        'n200sub': np.random.uniform(.11, .2, size=(nparts, nconds)),
        'lambda': np.random.uniform(.01, .02, size=(nparts, nspats)),
        'n200lat_mis': np.random.uniform(.11, .2, size = N_mis)
    }
    initials.append(chaininit)    
      
# Train the model and generate samples
fit = sm.sampling(data=data_winner, iter=niter, chains=nchains, warmup=nwarmup, thin=thin, init=initials)

utils.to_pickle(stan_model=sm, stan_fit=fit, save_path='../../save/non-boot/res_lambda_modelt.pkl')

