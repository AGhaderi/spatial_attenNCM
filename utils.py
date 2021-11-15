import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import itertools
import scipy.io as sio


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
    """Extracts effective number from stan model's summary as a pandas dataframe.
    Only considers parameters (Not all variables specified in stan's model).
    Note that, when DDM parameters are estimated at a trial level, these are included in the n_eff stats.
    Returns
    -------
    effective_number: DataFrame
        Data frame with rows the parameters and columns the n_eff and variable names.
    """
    summary = stan_fit.summary(pars=pars)
    effective_number = pd.DataFrame({'n_eff': np.array(summary['summary'])[:, 8],
                                'variable': summary['summary_rownames']})
    return effective_number

# waic function was taken from bellow
# https://github.com/laurafontanesi/rlssm/blob/main/rlssm/fits.py
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


def init_modelv(y, parts, nconds=4, nparts=24, nchains=1):
    """ Initialization for modelv
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
            'deltahier': np.random.uniform(.5, 2., size=nconds),
            'terhier': np.random.uniform(0, .2),
            'alphahier': np.random.uniform(0, 2.),
            'etahier': np.random.uniform(0, 1.),
            'delta': np.random.uniform(1, 3, size=(nparts,nconds)),
            'ter': np.random.uniform(.1, .2, size=nparts),
            'alpha': np.random.uniform(.5, 2., size=nparts),
            'eta': np.random.uniform(.1, 1, size=nparts)
        }
        chaininit['tersd'] = np.random.uniform(0., np.min(minrt)/2)
        chaininit['terhier'] = np.random.uniform(0.,np.min(minrt)/2)
        for p in range(0, nparts):
            chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2)
        initials.append(chaininit)
        
    return initials


def init_modela(y, parts, ncohers=2, nspats=2, nparts=24, nchains=1):
    """ Initialization for modela
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
            'tersd': np.random.uniform(.01, .1),
            'alphasd': np.random.uniform(.01, 1.),
            'etasd': np.random.uniform(.01, 1.),
            'deltahier': np.random.uniform(.5, 2., size=ncohers),
            'terhier': np.random.uniform(0, .2),
            'alphahier': np.random.uniform(0, 2., size = nspats),
            'etahier': np.random.uniform(0, 1.),
            'delta': np.random.uniform(1, 3, size=(nparts,ncohers)),
            'ter': np.random.uniform(.1, .2, size=nparts),
            'alpha': np.random.uniform(.5, 2., size=(nparts, nspats)),
            'eta': np.random.uniform(.1, 1, size=nparts)
        }
        chaininit['tersd'] = np.random.uniform(0., np.min(minrt)/2)
        chaininit['terhier'] = np.random.uniform(0.,np.min(minrt)/2)
        for p in range(0, nparts):
            chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2)
        initials.append(chaininit)
        
    return initials

def init_modelp(y, parts, ncohers=2, nparts=24, nchains=1):
    """ Initialization for modelp
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
            'terhier': np.random.uniform(0, .2),
            'alphahier': np.random.uniform(0, 2.),
            'etahier': np.random.uniform(0, 1.),
            'delta': np.random.uniform(1, 3, size=(nparts,ncohers)),
            'ter': np.random.uniform(.1, .2, size=nparts),
            'alpha': np.random.uniform(.5, 2., size=(nparts)),
            'eta': np.random.uniform(.1, 1, size=nparts)
        }
        chaininit['tersd'] = np.random.uniform(0., np.min(minrt)/2)
        chaininit['terhier'] = np.random.uniform(0.,np.min(minrt)/2)
        for p in range(0, nparts):
            chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2)
        initials.append(chaininit)
        
    return initials



def get_data(list_subj=None, idx_subj = None):
    """Load dataset
    """
    if not os.path.isfile('cross-validation/data.csv'):   
        #list of participants
        if list_subj is None:
            list_subj = ['sub-001', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009',
                         'sub-010', 'sub-011', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017']
        N = 16*288       #Number of trial-level observations
        ncohers = 2      #Number of coherence-level
        nspats = 2       #Number of spatial-level conditions
        nparts = 16      #Number of participant           
        nconds = 4       #Number of conditions
        y = np.zeros(16*288)    #acc*rt in seconds (negative and positive RTs 

        participant = np.zeros(16*288, dtype=int)   #participant index for each trial  
        condition   = np.zeros(16*288, dtype=int)   #condition index for each trial    
        maxrt = np.zeros([nparts,nconds])          # max response time for each subject and each of four condition

        idx =  0
        subj = 1
        for i in list_subj:
            examples_dir = "/home/a.ghaderi/Data/behavioral_data/raw/"+i+"/sourcedata-eeg_outside-MRT/beh/"+i+"_task-pdm_acq-outsideMRT_runs_beh.csv"
            df = pd.read_csv(examples_dir)
            df['response_corr']=df['response_corr'].replace(0,-1)                          #conver 0 (incorrect response) to -1
            y[idx*288:(idx+1)*288] = np.array(df['response_time']*df['response_corr'])     #acc*rt  
            if subj == 2:
                subj += 1
            if idx_subj is None:
                participant[idx*288:(idx+1)*288] = subj                                      #participant for each trial
            if idx_subj is not None:
                participant[idx*288:(idx+1)*288] = idx_subj[idx]                             #participant for each trial           
            condition[idx*288:(idx+1)*288]   = df['condition']                               #condition for each trial
            maxrt[idx] = np.array([df[df['condition']==1]['response_time'].max(),            #Maximum response time across subject and conditions
                  df[df['condition']==2]['response_time'].max(),
                  df[df['condition']==3]['response_time'].max(),
                  df[df['condition']==4]['response_time'].max()])
            idx = idx +1
            subj += 1
        #delete non int from arrays
        idx_non_int = np.where((condition==-9223372036854775808)|(y<0.15)&(y>-0.15))[0]
        condition = np.delete(condition,idx_non_int)
        N = N - idx_non_int.shape[0]
        participant = np.delete(participant,idx_non_int)
        y = np.delete(y,idx_non_int)

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
                               'conds':condition})
        data.to_csv("cross-validation/data.csv", index=False)

    else:
        data = pd.read_csv("cross-validation/data.csv")
    return data


def rsquared_pred(trueval,predval):
    """
    RSQUARED_PRED  Calculates R^2_prediction for data and statistics derived from data
    """
    divisor = np.sum(np.isfinite(trueval)) -1
    # Mean squared error of prediction
    MSEP = np.nansum(np.power(trueval - predval,2)) / divisor
    # Variance estimate of the true values
    vartrue = np.nansum(np.power(trueval - np.nanmean(trueval),2)) / divisor
    # R-squared definition
    rsquared = 1 - (MSEP / vartrue)
    return rsquared

#non-parametric bootstrap sampling from known subjects
def nonparaboot(y, parts, cond_coher, cond_spat, conds, run):
    if not os.path.isfile("boot/"+run+"/"+run+"_rand_parts.csv"): 
        os.mkdir("boot/"+run)
        
        unknown_part, known_parts = unknown_known()
 
        rand_parts = np.random.choice(known_parts, 24, replace=True)
        boot_y     = []
        boot_part = []
        boot_cond_coher  = []
        boot_cond_spat  = []
        boot_conds  = []
        part = 1
        for i in rand_parts:
            boot_y.extend(y[parts==i])
            boot_part.extend(list(itertools.repeat(part, parts[parts==i].shape[0])))
            boot_cond_coher.extend(cond_coher[parts==i])
            boot_cond_spat.extend(cond_spat[parts==i])
            boot_conds.extend(conds[parts==i])
            part +=1

        boot_y     = np.array(boot_y)
        boot_part = np.array(boot_part)
        boot_cond_coher  = np.array(boot_cond_coher)
        boot_cond_spat  = np.array(boot_cond_spat)
        boot_conds  = np.array(boot_conds)

        rand_parts = pd.DataFrame({'rand_parts':rand_parts})
        rand_parts.to_csv("boot/"+run+"/"+run+"_rand_parts.csv", index=False)

        bootsrap = pd.DataFrame({'boot_y':boot_y,
                                 'boot_part':boot_part,
                                 'boot_cond_coher':boot_cond_coher,
                                 'boot_cond_spat':boot_cond_spat,
                                 'boot_conds':boot_conds})
        bootsrap.to_csv("boot/"+run+"/"+run+"_boostrap.csv", index=False)

    else:
        rand_parts = pd.read_csv("boot/"+run+"/"+run+"_rand_parts.csv")
        bootsrap = pd.read_csv("boot/"+run+"/"+run+"_boostrap.csv")
        
    return rand_parts, bootsrap
 
#split data to 2/3 sector as train and 1/3 sector as test
def train_val(y, parts, cond_coher, cond_spat, conds):
    if not os.path.isfile('cross-validation/validation.csv'):   
        tr_y     = []
        tr_parts = []
        tr_cond_coher  = []
        tr_cond_spat  = []
        tr_conds  = []

        val_y     = []
        val_parts = []
        val_cond_coher  = []
        val_cond_spat  = []
        val_conds  = []
        
        unknown_part, known_parts = unknown_known()
        
        for p in known_parts:
            lenn = parts[parts==p].shape[0]
            lenn_tr = int((lenn*2)/3)
            indices = np.random.permutation(lenn)

            tr_y.extend(y[parts==p][indices[0:lenn_tr]])
            val_y.extend(y[parts==p][indices[lenn_tr:]])

            tr_parts.extend(parts[parts==p][indices[0:lenn_tr]])
            val_parts.extend(parts[parts==p][indices[lenn_tr:]])

            tr_cond_coher.extend(cond_coher[parts==p][indices[0:lenn_tr]])
            val_cond_coher.extend(cond_coher[parts==p][indices[lenn_tr:]])

            tr_cond_spat.extend(cond_spat[parts==p][indices[0:lenn_tr]])
            val_cond_spat.extend(cond_spat[parts==p][indices[lenn_tr:]])

            tr_conds.extend(conds[parts==p][indices[0:lenn_tr]])
            val_conds.extend(conds[parts==p][indices[lenn_tr:]])

        tr_y     = np.array(tr_y)
        tr_parts = np.array(tr_parts)
        tr_cond_coher  = np.array(tr_cond_coher)
        tr_cond_spat  = np.array(tr_cond_spat)
        tr_conds  = np.array(tr_conds)

        val_y     = np.array(val_y)
        val_parts = np.array(val_parts)
        val_cond_coher  = np.array(val_cond_coher)
        val_cond_spat  = np.array(val_cond_spat)
        val_conds  = np.array(val_conds)

        train = pd.DataFrame({'tr_y':tr_y,
                              'tr_parts':tr_parts,
                              'tr_cond_coher':tr_cond_coher,
                              'tr_cond_spat':tr_cond_spat,
                              'tr_conds':tr_conds})
        train.to_csv("cross-validation/train.csv", index=False)

        validation = pd.DataFrame({'val_y':val_y,
                                   'val_parts':val_parts,
                                   'val_cond_coher':val_cond_coher,
                                   'val_cond_spat':val_cond_spat,
                                   'val_conds':val_conds})
        validation.to_csv("cross-validation/validation.csv", index=False)
    else:
        train = pd.read_csv("cross-validation/train.csv")
        validation = pd.read_csv("cross-validation/validation.csv")
  
    return train, validation
 
 
#define unknown and known participants
def unknown_known():
    if not os.path.isfile('cross-validation/unknown_parts.csv'):   
        parts = np.array([1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        #unknown_part = np.random.choice(parts, 4)
        unknown_part = np.array([3,7,11,14])
        unknown = pd.DataFrame({'part':unknown_part})
        unknown.to_csv("cross-validation/unknown_parts.csv", index=False)
    else:
        unknown_part = np.array([3,7,11,14])
    
    if not os.path.isfile('cross-validation/test.csv'):   
        data = pd.read_csv('cross-validation/data.csv')
        ts_data = data[(data['participant']==3) | (data['participant']==7) | (data['participant']==11) | (data['participant']==14)]
        ts_data.to_csv("cross-validation/test.csv", header=['ts_y', 'ts_parts', 'ts_cond_coher', 'ts_cond_spat', 'ts_conds'], index=False)


    if not os.path.isfile('cross-validation/known_parts.csv'):   
        known_parts = np.array([1,4,5,6,8,9,10,12,13,15,16,17])    
        known = pd.DataFrame({'part':known_parts})
        known.to_csv("cross-validation/known_parts.csv", index=False)
    else:
        known_parts = np.array([1,4,5,6,8,9,10,12,13,15,16,17])    

    return unknown_part, known_parts


#quantile boostrap training data
def tr_data_quantile():
    
    if not os.path.isfile("r_square/tr_quantile.mat"): 
        # number of bootstrap runs
        nboots  = 18

        # number of condition
        nconds = 4

        # number of quantiles (2th, mean, median and 75th quantile)
        nqnts  = 4

        # number of participatns
        nparts = 24

        tr_rt  = np.zeros((nboots,nparts,nconds,nqnts))
        tr_acc  = np.zeros((nboots,nparts,nconds))

        for r in range(nboots):
            boot_data = pd.read_csv("boot/"+str(r+1)+"/"+str(r+1)+"_boostrap.csv")
            for p in range(nparts):
                boot_rt_con1 = boot_data[(boot_data['boot_part']==p+1)&(boot_data['boot_conds']==1)]['boot_y'].to_numpy()
                boot_rt_con2 = boot_data[(boot_data['boot_part']==p+1)&(boot_data['boot_conds']==2)]['boot_y'].to_numpy()
                boot_rt_con3 = boot_data[(boot_data['boot_part']==p+1)&(boot_data['boot_conds']==3)]['boot_y'].to_numpy()
                boot_rt_con4 = boot_data[(boot_data['boot_part']==p+1)&(boot_data['boot_conds']==4)]['boot_y'].to_numpy()

                tr_rt[r,p,0,0]  = np.quantile(np.abs(boot_rt_con1), .25)
                tr_rt[r,p,0,1]  = np.quantile(np.abs(boot_rt_con1), .75)
                tr_rt[r,p,0,2]  = np.mean(np.abs(boot_rt_con1))
                tr_rt[r,p,0,3]  = np.median(np.abs(boot_rt_con1))
                tr_acc[r,p,0]   = len(boot_rt_con1[boot_rt_con1>0])/boot_rt_con1.shape[0]

                tr_rt[r,p,1,0]  = np.quantile(np.abs(boot_rt_con2), .25)
                tr_rt[r,p,1,1]  = np.quantile(np.abs(boot_rt_con2), .75)
                tr_rt[r,p,1,2]  = np.mean(np.abs(boot_rt_con2))
                tr_rt[r,p,1,3]  = np.median(np.abs(boot_rt_con2))
                tr_acc[r,p,1]   = len(boot_rt_con2[boot_rt_con2>0])/boot_rt_con2.shape[0]

                tr_rt[r,p,2,0]  = np.quantile(np.abs(boot_rt_con3), .25)
                tr_rt[r,p,2,1]  = np.quantile(np.abs(boot_rt_con3), .75)
                tr_rt[r,p,2,2]  = np.mean(np.abs(boot_rt_con3))
                tr_rt[r,p,2,3]  = np.median(np.abs(boot_rt_con3))
                tr_acc[r,p,2]   = len(boot_rt_con3[boot_rt_con3>0])/boot_rt_con3.shape[0]

                tr_rt[r,p,3,0]  = np.quantile(np.abs(boot_rt_con4), .25)
                tr_rt[r,p,3,1]  = np.quantile(np.abs(boot_rt_con4), .75)
                tr_rt[r,p,3,2]  = np.mean(np.abs(boot_rt_con4))
                tr_rt[r,p,3,3]  = np.median(np.abs(boot_rt_con4))
                tr_acc[r,p,3]   = len(boot_rt_con4[boot_rt_con4>0])/boot_rt_con4.shape[0]

        #save rt and acc by .mat file
        tr_data  = dict()
        tr_data['tr_rt'] = tr_rt
        tr_data ['tr_acc'] = tr_acc
        sio.savemat('r_square/tr_quantile.mat', tr_data)
    else:
        tr_data = sio.loadmat("r_square/tr_quantile.mat")
        tr_rt = tr_data['tr_rt']
        tr_acc = tr_data['tr_acc']
    return tr_rt, tr_acc

#quantile validataion data
def val_data_quantile():
    
    if not os.path.isfile("r_square/val_quantile.mat"): 
        #known participamt
        unknown_part, known_parts = unknown_known()
        
        # number of condition
        nconds = 4

        # number of quantiles (2th, mean, median and 75th quantile)
        nqnts  = 4

        # number of participatns
        nparts = 12

        val_rt  = np.zeros((nparts,nconds,nqnts))
        val_acc  = np.zeros((nparts,nconds))

        val_data = pd.read_csv("cross-validation/validation.csv")
        
        # participant subject
        idx = 0
        for p in known_parts:
            
            val_rt_con1 = val_data[(val_data['val_parts']==p)&(val_data['val_conds']==1)]['val_y'].to_numpy()
            val_rt_con2 = val_data[(val_data['val_parts']==p)&(val_data['val_conds']==2)]['val_y'].to_numpy()
            val_rt_con3 = val_data[(val_data['val_parts']==p)&(val_data['val_conds']==3)]['val_y'].to_numpy()
            val_rt_con4 = val_data[(val_data['val_parts']==p)&(val_data['val_conds']==4)]['val_y'].to_numpy()

            val_rt[idx,0,0]  = np.quantile(np.abs(val_rt_con1), .25)
            val_rt[idx,0,1]  = np.quantile(np.abs(val_rt_con1), .75)
            val_rt[idx,0,2]  = np.mean(np.abs(val_rt_con1))
            val_rt[idx,0,3]  = np.median(np.abs(val_rt_con1))
            val_acc[idx,0]   = len(val_rt_con1[val_rt_con1>0])/val_rt_con1.shape[0]

            val_rt[idx,1,0]  = np.quantile(np.abs(val_rt_con2), .25)
            val_rt[idx,1,1]  = np.quantile(np.abs(val_rt_con2), .75)
            val_rt[idx,1,2]  = np.mean(np.abs(val_rt_con2))
            val_rt[idx,1,3]  = np.median(np.abs(val_rt_con2))
            val_acc[idx,1]   = len(val_rt_con2[val_rt_con2>0])/val_rt_con2.shape[0]

            val_rt[idx,2,0]  = np.quantile(np.abs(val_rt_con3), .25)
            val_rt[idx,2,1]  = np.quantile(np.abs(val_rt_con3), .75)
            val_rt[idx,2,2]  = np.mean(np.abs(val_rt_con3))
            val_rt[idx,2,3]  = np.median(np.abs(val_rt_con3))
            val_acc[idx,2]   = len(val_rt_con3[val_rt_con3>0])/val_rt_con3.shape[0]

            val_rt[idx,3,0]  = np.quantile(np.abs(val_rt_con4), .25)
            val_rt[idx,3,1]  = np.quantile(np.abs(val_rt_con4), .75)
            val_rt[idx,3,2]  = np.mean(np.abs(val_rt_con4))
            val_rt[idx,3,3]  = np.median(np.abs(val_rt_con4))
            val_acc[idx,3]   = len(val_rt_con4[val_rt_con4>0])/val_rt_con4.shape[0]
            
            idx += 1
                        
        #save rt and acc by .mat file
        val_data  = dict()
        val_data['val_rt'] = val_rt
        val_data ['val_acc'] = val_acc
        sio.savemat('r_square/val_quantile.mat', val_data)

    else:
        val_data = sio.loadmat("r_square/val_quantile.mat")
        val_rt = val_data['val_rt']
        val_acc = val_data['val_acc']
    return val_rt, val_acc

#quantile test data
def ts_data_quantile():
    
    if not os.path.isfile("r_square/ts_quantile.mat"): 
        #known participamt
        unknown_part, known_parts = unknown_known()
        
        # number of condition
        nconds = 4

        # number of quantiles (2th, mean, median and 75th quantile)
        nqnts  = 4

        # number of participatns
        nparts = 4

        ts_rt  = np.zeros((nparts,nconds,nqnts))
        ts_acc  = np.zeros((nparts,nconds))
        
        ts_data = pd.read_csv("cross-validation/test.csv")
        
        # participant subject
        idx =  0
        for p in unknown_part:

            ts_rt_con1 = ts_data[(ts_data['ts_parts']==p)&(ts_data['ts_conds']==1)]['ts_y'].to_numpy()
            ts_rt_con2 = ts_data[(ts_data['ts_parts']==p)&(ts_data['ts_conds']==2)]['ts_y'].to_numpy()
            ts_rt_con3 = ts_data[(ts_data['ts_parts']==p)&(ts_data['ts_conds']==3)]['ts_y'].to_numpy()
            ts_rt_con4 = ts_data[(ts_data['ts_parts']==p)&(ts_data['ts_conds']==4)]['ts_y'].to_numpy()

            ts_rt[idx,0,0]  = np.quantile(np.abs(ts_rt_con1), .25)
            ts_rt[idx,0,1]  = np.quantile(np.abs(ts_rt_con1), .75)
            ts_rt[idx,0,2]  = np.mean(np.abs(ts_rt_con1))
            ts_rt[idx,0,3]  = np.median(np.abs(ts_rt_con1))
            ts_acc[idx,0]   = len(ts_rt_con1[ts_rt_con1>0])/ts_rt_con1.shape[0]

            ts_rt[idx,1,0]  = np.quantile(np.abs(ts_rt_con2), .25)
            ts_rt[idx,1,1]  = np.quantile(np.abs(ts_rt_con2), .75)
            ts_rt[idx,1,2]  = np.mean(np.abs(ts_rt_con2))
            ts_rt[idx,1,3]  = np.median(np.abs(ts_rt_con2))
            ts_acc[idx,1]   = len(ts_rt_con2[ts_rt_con2>0])/ts_rt_con2.shape[0]

            ts_rt[idx,2,0]  = np.quantile(np.abs(ts_rt_con3), .25)
            ts_rt[idx,2,1]  = np.quantile(np.abs(ts_rt_con3), .75)
            ts_rt[idx,2,2]  = np.mean(np.abs(ts_rt_con3))
            ts_rt[idx,2,3]  = np.median(np.abs(ts_rt_con3))
            ts_acc[idx,2]   = len(ts_rt_con3[ts_rt_con3>0])/ts_rt_con3.shape[0]

            ts_rt[idx,3,0]  = np.quantile(np.abs(ts_rt_con4), .25)
            ts_rt[idx,3,1]  = np.quantile(np.abs(ts_rt_con4), .75)
            ts_rt[idx,3,2]  = np.mean(np.abs(ts_rt_con4))
            ts_rt[idx,3,3]  = np.median(np.abs(ts_rt_con4))
            ts_acc[idx,3]   = len(ts_rt_con4[ts_rt_con4>0])/ts_rt_con4.shape[0]
            
            idx +=1
            
        #save rt and acc by .mat file
        ts_data  = dict()
        ts_data['ts_rt'] = ts_rt
        ts_data ['ts_acc'] = ts_acc
        sio.savemat('r_square/ts_quantile.mat', ts_data)

    else:
        ts_data = sio.loadmat("r_square/ts_quantile.mat")
        ts_rt = ts_data['ts_rt']
        ts_acc = ts_data['ts_acc']
    return ts_rt, ts_acc