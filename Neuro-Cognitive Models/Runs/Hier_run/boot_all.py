#!/home/a.ghaderi/.conda/envs/envjm/bin/python
import pystan 
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
import utils

#loading dateset
data = utils.get_data()      
#non parametric bootstrap
run = 5
rand_parts, boot_data = utils.nonparaboot(y=data['y'].to_numpy(), parts=data['participant'].to_numpy(), 
                                cond_coher=data['cond_coher'].to_numpy(), cond_spat=data['cond_spat'].to_numpy(),
                                conds=data['conds'].to_numpy(), n200lat=data['n200lat'].to_numpy(), run=str(run))

mis = np.where((boot_data['boot_n200lat']<.101)|(boot_data['boot_n200lat']>.248))[0] # missing data for n200lat
obs = np.where((boot_data['boot_n200lat']>.101)&(boot_data['boot_n200lat']<.248))[0] # observation and missing data for n200lat
N_mis = mis.shape[0]  # number of missing data
N_obs = obs.shape[0]   # number of observed data

modelfile = '../../stans/res_lambda_all.stan' #reading the model span
f = open(modelfile, 'r')
model_wiener = f.read()
sm = pystan.StanModel(model_code=model_wiener)# Compile the model stan

ncohers = 2  #Number of coherence conditions
nspats = 2   #Number of spatial conditions
nconds = 4   #Number of conditions
nparts = 30  #Number of subjects
y = boot_data['boot_y']
participant = boot_data['boot_part']
cond_coher = boot_data['boot_cond_coher']
cond_spat = boot_data['boot_cond_spat']
conds = boot_data['boot_conds']
n200lat = boot_data['boot_n200lat']

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
niter = 8000
nwarmup = 2000
nchains = 1
thin = 1

initials = [] # initial sampling
for c in range(0, nchains):
    chaininit = {
        'delta_ressd': np.random.uniform(.1, .4),
        'alpha_ressd': np.random.uniform(.1, .4),        
        'tau_ressd': np.random.uniform(.01, .02),
        'eta_ressd': np.random.uniform(.1, .4),
        'delta_lambdasd': np.random.uniform(.1, .4),
        'alpha_lambdasd': np.random.uniform(.1, .4),        
        'tau_lambdasd': np.random.uniform(.01, .02),
        'eta_lambdasd': np.random.uniform(.1, .4),
        'n200condsd': np.random.uniform(.01, .1),
        'n200trialsd': np.random.uniform(.01, .1),        
        'delta_reshier': np.random.uniform(1., 3., size=nconds),
        'alpha_reshier': np.random.uniform(1., 3., size=nspats),       
        'tau_reshier': np.random.uniform(.01, .02, size=nspats),
        'eta_reshier': np.random.uniform(.01, 1, size=nspats),
        'delta_lambdahier': np.random.uniform(2., 5., size=nconds),
        'alpha_lambdahier': np.random.uniform(2., 5., size=nspats),       
        'tau_lambdahier': np.random.uniform(.1, .2, size=nspats),   
        'eta_lambdahier': np.random.uniform(.01, 1, size=nspats),
        'n200cond': np.random.uniform(.11, .2, size=nconds),
        'delta_res': np.random.uniform(1., 2, size=(nparts,nconds)),
        'alpha_res': np.random.uniform(1., 2., size=(nparts,nspats)),
        'tau_res': np.random.uniform(.01, .02, size=(nparts, nspats)),
        'eta_res': np.random.uniform(.5, 1.5, size=(nparts, nspats)),
        'delta_lambda': np.random.uniform(2., 5, size=(nparts,nconds)),
        'alpha_lambda': np.random.uniform(2., 5., size=(nparts,nspats)),    
        'tau_lambda': np.random.uniform(.1, .2, size=(nparts, nspats)), 
        'eta_lambda': np.random.uniform(1, 3, size=(nparts, nspats)), 
        'n200sub': np.random.uniform(.11, .2, size=(nparts, nconds)),
        'n200lat_mis': np.random.uniform(.11, .2, size = N_mis)
    }
    initials.append(chaininit)    
      
# Train the model and generate samples
fit = sm.sampling(data=data_winner, iter=niter, chains=nchains, warmup=nwarmup, thin=thin, init=initials)

utils.to_pickle(stan_model=sm, stan_fit=fit, save_path="../../save/boot/"+str(run)+"/"+str(run)+"_res_lambda_all_1.pkl")