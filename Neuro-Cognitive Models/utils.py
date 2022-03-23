import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Some codes are taken from bellow
# https://github.com/laurafontanesi/rlssm/blob/main/rlssm/utils.py 
def rhat(stan_fit, pars):
    """Extracts rhat from stan model's summary as a pandas dataframe.
    Only considers parameters (Not all variables specified in stan's model).
    Note that, when DDM parameters are estimated at a trial level, these are included in the rhat stats.
    Returns
    -------
    convergence: DataFrame
        Data frame with rows the parameters and columns the rhat and variable names.
    """
    summary = stan_fit.summary(pars=pars)
    convergence = pd.DataFrame({'rhat': np.array(summary['summary'])[:, 9],
                                'variable': summary['summary_rownames']})
    return convergence

def n_eff(stan_fit, pars):
    """Extracts n_eff from stan model's summary as a pandas dataframe.
    Only considers parameters (Not all variables specified in stan's model).
    Note that, when DDM parameters are estimated at a trial level, these are included in the n_eff stats.
    Returns
    -------
    convergence: DataFrame
        Data frame with rows the parameters and columns the rhat and variable names.
    """
    summary = stan_fit.summary(pars=pars)
    effective_num = pd.DataFrame({'n_eff': np.array(summary['summary'])[:, 8],
                                'variable': summary['summary_rownames']})
    return effective_num
  
def waic(log_likelihood):
    """Calculates the Watanabe-Akaike information criteria.
    Calculates pWAIC1 and pWAIC2
    according to http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    Parameters
    ----------
    pointwise : bool, default to False
        By default, gives the averaged waic.
        Set to True is you want additional waic per observation.
    Returns
    -------
    out: dict
        Dictionary containing lppd (log pointwise predictive density),
        p_waic, waic, waic_se (standard error of the waic), and
        pointwise_waic (when `pointwise` is True).
    """
    
    N = log_likelihood.shape[1]
    likelihood = np.exp(log_likelihood)

    mean_l = np.mean(likelihood, axis=0) # N observations

    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)

    pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
    var_l = np.sum(pointwise_var_l)

    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(N * np.var(pointwise_waic))

    out = {'lppd':lppd,
           'p_waic':var_l,
           'waic':waic,
           'waic_se':waic_se}
    return out


def get_data():
    """Load dataset which has n200 latency
    """
    #list of participants
    list_subj = ['sub-001', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009',
                 'sub-010', 'sub-011', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-017']
    ncohers = 2      #Number of coherence-level
    nspats = 2       #Number of spatial-level conditions
    nparts = 15      #Number of participant           
    nconds = 4       #Number of conditions
    y = np.zeros(nparts*288)    #acc*rt in seconds (negative and positive RTs 
    participant = np.zeros(nparts*288, dtype=int)   #participant index for each trial  
    condition   = np.zeros(nparts*288, dtype=int)   #condition index for each trial   
    n200lat = np.zeros(nparts*288, dtype=float)         # latency for each trial
 
    idx =  0
    for i in list_subj:
        examples_dir = "/home/a.ghaderi/Data/raw/"+i+"/sourcedata-eeg_outside-MRT/beh/"+i+"_task-pdm_acq-outsideMRT_runs_beh_n200lat.csv"
        df = pd.read_csv(examples_dir)
        df['response_corr']=df['response_corr'].replace(0,-1)                         #conver 0 (incorrect response) to -1
        y[idx*288:(idx+1)*288] = np.array(df['response_time']*df['response_corr'])    #acc*rt  
        participant[idx*288:(idx+1)*288] = idx+1                                      #participant for each trial
        condition[idx*288:(idx+1)*288]   = df['condition']                            #condition for each trial
        n200lat[idx*288:(idx+1)*288] = df['n200lat']
        idx = idx +1
    #delete non int from arrays
    idx_non_int = np.where((condition==-9223372036854775808)|((y<0.15)&(y>-0.15)))[0]
    #|(n200lat==-9223372036854775808), |(y<0.15)&(y>-0.15)
    condition = np.delete(condition,idx_non_int)
    participant = np.delete(participant,idx_non_int)
    y = np.delete(y,idx_non_int)
    n200lat = np.delete(n200lat,idx_non_int)

    #convert four conditions to two coherence condition
    cond_coher = np.where(condition==2, 1, condition) 
    cond_coher = np.where(cond_coher==3, 2, cond_coher) 
    cond_coher = np.where(cond_coher==4, 2, cond_coher) 

    #convert four conditions to two spatia condition
    cond_spat = np.where(condition==3, 1, condition) 
    cond_spat = np.where(cond_spat==4, 2, cond_spat) 

    data = pd.DataFrame({'y':y,
                           'participant':participant,
                           'cond_coher':cond_coher,
                           'cond_spat':cond_spat,
                           'conds':condition,
                           'n200lat':n200lat})
    return data


def to_pickle(stan_model, stan_fit, save_path):
    """Save pickle the fitted model's results with .pkl format.
    """
    try:
        with open(save_path, "wb") as f:   #Pickling
            pickle.dump({"model" : stan_model, "fit" : stan_fit}, f, protocol=pickle.HIGHEST_PROTOCOL)       
            f.close()
            print('Saved results to ', save_path)
    except:
        print("An exception occurred")

def load_pickle(load_path):
    """Load model results from pickle.
    """
    try:
        with open(load_path, "rb") as fp:   # Unpickling
            results_load = pickle.load(fp)
            return results_load
    except:
        print("An exception occurred")
    
def init_modelt(y, parts, ncohers=2, nspats=2, nparts=24, nchains=1):
    """ Initialization for modelt
    """
    rt = np.abs(y)
    minrt = np.zeros(nparts)
    #minimum rt for observation data
    for k in range(0,nparts):
        minrt[k] = np.min(rt[(parts == k+1)])
        
    initials = []
    for c in range(0, nchains):
        chaininit = {
            'deltasd': np.random.uniform(.1, 3.),
            'tersd': np.random.uniform(.01, .2),
            'alphasd': np.random.uniform(.01, 1.),
            'etasd': np.random.uniform(.01, 1.),
            'deltahier': np.random.uniform(.5, 2., size=ncohers),
            'terhier': np.random.uniform(0, .2, size = nspats),
            'alphahier': np.random.uniform(0, 2.),
            'etahier': np.random.uniform(0, 1.),
            'delta': np.random.uniform(1, 3, size=(nparts,ncohers)),
            'ter': np.random.uniform(.1, .2, size=(nparts, nspats)),
            'alpha': np.random.uniform(.5, 2., size=nparts),
            'eta': np.random.uniform(.1, 1, size=nparts)
        }
        chaininit['tersd'] = np.random.uniform(0., np.min(minrt)/2)
        chaininit['terhier'] = np.random.uniform(0.,np.min(minrt)/2, size = nspats)
        for p in range(0, nparts):
            chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2, size = nspats)
        initials.append(chaininit)
        
    return initials

def maxrt(data):
    """
    Calculating absolute maximum of response time for each participant 
    """
    nparts = len(np.unique(data['participant']))
    maxrt = np.zeros([nparts])

    for i in range(nparts):
        maxrt[i] = np.abs(data['y']).max()
    return maxrt

def minrt(data):
    """
    Calculating absolute maximum of response time for each participant 
    """
    nparts = len(np.unique(data['participant']))
    minrt = np.zeros([nparts])

    for i in range(nparts):
        minrt[i] = np.abs(data['y']).min()
        #minrt[i] = np.abs(data[data['participant']==i+1]['y']).min()
    return minrt


#non-parametric bootstrap sampling from all subjects
def nonparaboot(y, parts, cond_coher, cond_spat, conds, n200lat, run):
    if not os.path.isfile("../../save/boot/"+run+"/"+run+"_rand_parts.csv"): 
        os.mkdir("../../save/boot/"+run)
        
        nparts = 15
        rand_parts = np.random.choice(nparts, 30, replace=True) + 1
        boot_y     = []
        boot_part = []
        boot_cond_coher  = []
        boot_cond_spat  = []
        boot_conds  = []
        boot_n200lat  = []
        part = 1
        for i in rand_parts:
            boot_y.extend(y[parts==i])
            boot_part.extend(list(np.repeat(part, parts[parts==i].shape[0])))
            boot_cond_coher.extend(cond_coher[parts==i])
            boot_cond_spat.extend(cond_spat[parts==i])
            boot_conds.extend(conds[parts==i])
            boot_n200lat.extend(n200lat[parts==i])
            part +=1

        boot_y = np.array(boot_y)
        boot_part = np.array(boot_part)
        boot_cond_coher  = np.array(boot_cond_coher)
        boot_cond_spat  = np.array(boot_cond_spat)
        boot_conds  = np.array(boot_conds)
        boot_n200lat  = np.array(boot_n200lat)

        rand_parts = pd.DataFrame({'rand_parts':rand_parts})
        rand_parts.to_csv("../../save/boot/"+run+"/"+run+"_rand_parts.csv", index=False)

        bootsrap = pd.DataFrame({'boot_y':boot_y,
                                 'boot_part':boot_part,
                                 'boot_cond_coher':boot_cond_coher,
                                 'boot_cond_spat':boot_cond_spat,
                                 'boot_conds':boot_conds,
                                 'boot_n200lat':boot_n200lat})
        bootsrap.to_csv("../../save/boot/"+run+"/"+run+"_boostrap.csv", index=False)

    else:
        rand_parts = pd.read_csv("../../save/boot/"+run+"/"+run+"_rand_parts.csv")
        bootsrap = pd.read_csv("../../save/boot/"+run+"/"+run+"_boostrap.csv")
        
    return rand_parts, bootsrap
 