import numpy as np
import os
import scipy.io as sio
import utils


# Simulate diffusion models
# It was taken from bellow
# https://github.com/mdnunez/pyhddmjags/blob/master/pyhddmjagsutils.py
def simulratcliff(N=100,Alpha=1,Tau=.4,Nu=1,Beta=.5,rangeTau=0,rangeBeta=0,Eta=.3, Varsigma=1):
    """
    SIMULRATCLIFF  Generates data according to a drift diffusion model with optional trial-to-trial variability


    Reference:
    Tuerlinckx, F., Maris, E.,
    Ratcliff, R., & De Boeck, P. (2001). A comparison of four methods for
    simulating the diffusion process. Behavior Research Methods,
    Instruments, & Computers, 33, 443-456.

    Parameters
    ----------
    N: a integer denoting the size of the output vector
    (defaults to 100 experimental trials)

    Alpha: the mean boundary separation across trials  in evidence units
    (defaults to 1 evidence unit)

    Tau: the mean non-decision time across trials in seconds
    (defaults to .4 seconds)

    Nu: the mean drift rate across trials in evidence units per second
    (defaults to 1 evidence units per second, restricted to -5 to 5 units)

    Beta: the initial bias in the evidence process for choice A as a proportion of boundary Alpha
    (defaults to .5 or 50% of total evidence units given by Alpha)

    rangeTau: Non-decision time across trials is generated from a uniform
    distribution of Tau - rangeTau/2 to  Tau + rangeTau/2 across trials
    (defaults to 0 seconds)

    rangeZeta: Bias across trials is generated from a uniform distribution
    of Zeta - rangeZeta/2 to Zeta + rangeZeta/2 across trials
    (defaults to 0 evidence units)

    Eta: Standard deviation of the drift rate across trials
    (defaults to 3 evidence units per second, restricted to less than 3 evidence units)

    Varsigma: The diffusion coefficient, the standard deviation of the
    evidence accumulation process within one trial. It is recommended that
    this parameter be kept fixed unless you have reason to explore this parameter
    (defaults to 1 evidence unit per second)

    Returns
    -------
    Numpy array with reaction times (in seconds) multiplied by the response vector
    such that negative reaction times encode response B and positive reaction times
    encode response A 
    
    
    Converted from simuldiff.m MATLAB script by Joachim Vandekerckhove
    See also http://ppw.kuleuven.be/okp/dmatoolbox.
    """

    if (Nu < -5) or (Nu > 5):
        Nu = np.sign(Nu)*5
        warnings.warn('Nu is not in the range [-5 5], bounding drift rate to %.1f...' % (Nu))

    if (Eta > 3):
        warning.warn('Standard deviation of drift rate is out of bounds, bounding drift rate to 3')
        eta = 3

    if (Eta == 0):
        Eta = 1e-16

    #Initialize output vectors
    result = np.zeros(N)
    T = np.zeros(N)
    XX = np.zeros(N)

    #Called sigma in 2001 paper
    D = np.power(Varsigma,2)/2

    #Program specifications
    eps = 2.220446049250313e-16 #precision from 1.0 to next double-precision number
    delta=eps

    for n in range(0,N):
        r1 = np.random.normal()
        mu = Nu + r1*Eta
        bb = Beta - rangeBeta/2 + rangeBeta*np.random.uniform()
        zz = bb*Alpha
        finish = 0
        totaltime = 0
        startpos = 0
        Aupper = Alpha - zz
        Alower = -zz
        radius = np.min(np.array([np.abs(Aupper), np.abs(Alower)]))
        while (finish==0):
            lambda_ = 0.25*np.power(mu,2)/D + 0.25*D*np.power(np.pi,2)/np.power(radius,2)
            # eq. formula (13) in 2001 paper with D = sigma^2/2 and radius = Alpha/2
            F = D*np.pi/(radius*mu)
            F = np.power(F,2)/(1 + np.power(F,2) )
            # formula p447 in 2001 paper
            prob = np.exp(radius*mu/D)
            prob = prob/(1 + prob)
            dir_ = 2*(np.random.uniform() < prob) - 1
            l = -1
            s2 = 0
            while (s2>l):
                s2=np.random.uniform()
                s1=np.random.uniform()
                tnew=0
                told=0
                uu=0
                while (np.abs(tnew-told)>eps) or (uu==0):
                    told=tnew
                    uu=uu+1
                    tnew = told + (2*uu+1) * np.power(-1,uu) * np.power(s1,(F*np.power(2*uu+1,2)));
                    # infinite sum in formula (16) in BRMIC,2001
                l = 1 + np.power(s1,(-F)) * tnew;
            # rest of formula (16)
            t = np.abs(np.log(s1))/lambda_;
            # is the negative of t* in (14) in BRMIC,2001
            totaltime=totaltime+t
            dir_=startpos+dir_*radius
            ndt = Tau - rangeTau/2 + rangeTau*np.random.uniform()
            if ( (dir_ + delta) > Aupper):
                T[n]=ndt+totaltime
                XX[n]=1
                finish=1
            elif ( (dir_-delta) < Alower ):
                T[n]=ndt+totaltime
                XX[n]=-1
                finish=1
            else:
                startpos=dir_
                radius=np.min(np.abs([Aupper, Alower]-startpos))

    result = T*XX
    return result

def boot_genparam(model='modelt', nboots = 1, nsamples=1000, ntrials=300):
    
    if not os.path.isfile('r_square/genparam_'+str(nboots)+'.mat'): 

        # nboots is related to bootstraping run which is 1 to 30
        # number of condition
        nconds = 4

        # number of quantiles (2th, mean, median and 75th quantile)
        nqnts  = 4

        # number of participatns
        nparts = 24

        #rt and acc across participants and conitions
        genparam_rt = np.zeros((nparts,nconds,nsamples*ntrials))
        genparam_acc = np.zeros((nparts,nconds,nsamples*ntrials))

        pkl = utils.load_pickle("boot/"+str(nboots)+"/"+ str(nboots) + "_"+model +".pkl")
        fit = pkl['fit']
        #samples of posterior parameters
        samples = fit.extract(permuted=True)

        for p in range(nparts):
            print(p)
            rt,acc = simulation.get_posterior_predictives_individual(samples=samples, part=p+1, nsamples=nsamples, ntrials=ntrials, model=model)
            genparam_rt[p] = rt
            genparam_acc[p] = acc

        #save rt and acc by .mat file
        genparam  = dict()
        genparam['model_rt'] = genparam_rt
        genparam ['model_acc'] = genparam_acc
        sio.savemat('r_square/genparam_'+str(nboots)+'.mat', genparam)

    else:
        genparam = sio.loadmat('r_square//genparam_'+str(nboots)+'.mat')
        genparam_rt = genparam['model_rt']
        genparam_acc = genparam['model_acc']

    return genparam_rt, genparam_acc

def get_posterior_predictives_group(samples, nsamples=1000, ntrials=300, model = 'modelt'):
    """Calculates posterior predictives of choices and response times.
       Hierarchical and condition-level parameters
    """      
    nconds = 4  # Number of conditions
    ncohers = 2 # Number of spatial   
    nspats = 2  # Number of coherence
    
    rt = np.zeros((nconds,nsamples*ntrials))
    acc = np.zeros((nconds,nsamples*ntrials))
    
    # hierarchical parameters
    deltahier = samples['deltahier']
    terhier = samples['terhier']
    alphahier = samples['alphahier']
    etahier = samples['etahier']
    
        
    #choose randomely 1000 samples
    indices = np.random.choice(range(eta.shape[0]),nsamples)
    
    if model=='modelt':
        for k in range(ncohers):
            for j in range(nspats):             
                indextrack = np.arange(ntrials)
                for i in indices:
                    tempout = simulratcliff(N=ntrials, Alpha= alphahier[i], Tau= terhier[i,j], Beta=.5, 
                        Nu= deltahier[i,k], Eta = etahier[i])
                    tempx = np.sign(np.real(tempout))
                    tempt = np.abs(np.real(tempout))
                    rt[k*2+j,indextrack] = tempt
                    acc[k*2+j,indextrack] = (tempx + 1)/2
                    indextrack += ntrials
    elif model=='modelv':
        for k in range(nconds):
            indextrack = np.arange(ntrials)
            for i in indices:
                tempout = simulratcliff(N=ntrials, Alpha= alphahier[i], Tau= terhier[i], Beta=.5, 
                    Nu= deltahier[i,k], Eta= etahier[i])
                tempx = np.sign(np.real(tempout))
                tempt = np.abs(np.real(tempout))
                rt[k,indextrack] = tempt
                acc[k,indextrack] = (tempx + 1)/2
                indextrack += ntrials
    elif model=='modela':
        for k in range(ncohers):
            for j in range(nspats):
                indextrack = np.arange(ntrials)
                for i in indices:
                    tempout = simulratcliff(N=ntrials, Alpha= alphahier[i,j], Tau= terhier[i], Beta=.5, 
                        Nu= deltahier[i,k], Eta = etahier[i])
                    tempx = np.sign(np.real(tempout))
                    tempt = np.abs(np.real(tempout))
                    rt[k*2+j,indextrack] = tempt
                    acc[k*2+j,indextrack] = (tempx + 1)/2
                    indextrack += ntrials
    elif model=='modelp':
        for k in range(nconds):
            indextrack = np.arange(ntrials)
            for i in indices:
                tempout = simulratcliff(N=ntrials, Alpha= alphahier[i], Tau= terhier[i,j], Beta=.5, 
                    Nu= deltahier[i,int(k/2)], Eta = etahier[i])
                tempx = np.sign(np.real(tempout))
                tempt = np.abs(np.real(tempout))
                rt[k,indextrack] = tempt
                acc[k,indextrack] = (tempx + 1)/2
                indextrack += ntrials
    return rt,acc

def get_posterior_predictives_individual(samples, part=1, nsamples=1000, ntrials=300, model = 'modelt'):
    """Calculates posterior predictives of choices and response times.
       Hierarchical and condition-level parameters
    """      
    nconds = 4  # Number of conditions
    ncohers = 2 # Number of spatial   
    nspats = 2  # Number of coherence
    
    rt = np.zeros((nconds,nsamples*ntrials))
    acc = np.zeros((nconds,nsamples*ntrials))
    
    # participant-level parameters
    delta = samples['delta'][:,part-1]
    ter = samples['ter'][:,part-1]
    alpha = samples['alpha'][:,part-1]
    eta = samples['eta'][:,part-1]
    
    #choose randomely 1000 samples
    indices = np.random.choice(range(eta.shape[0]),nsamples)
    
    if model=='modelt':
        for k in range(ncohers):
            for j in range(nspats):             
                indextrack = np.arange(ntrials)
                for i in indices:
                    tempout = simulratcliff(N=ntrials, Alpha= alpha[i], Tau= ter[i,j], Beta=.5, 
                        Nu= delta[i,k], Eta = eta[i])
                    tempx = np.sign(np.real(tempout))
                    tempt = np.abs(np.real(tempout))
                    rt[k*2+j,indextrack] = tempt
                    acc[k*2+j,indextrack] = (tempx + 1)/2
                    indextrack += ntrials
    elif model=='modelv':
        for k in range(nconds):
            indextrack = np.arange(ntrials)
            for i in indices:
                tempout = simulratcliff(N=ntrials, Alpha= alpha[i], Tau= ter[i], Beta=.5, 
                    Nu= delta[i,k], Eta= eta[i])
                tempx = np.sign(np.real(tempout))
                tempt = np.abs(np.real(tempout))
                rt[k,indextrack] = tempt
                acc[k,indextrack] = (tempx + 1)/2
                indextrack += ntrials
    elif model=='modela':
        for k in range(ncohers):
            for j in range(nspats):
                indextrack = np.arange(ntrials)
                for i in indices:
                    tempout = simulratcliff(N=ntrials, Alpha= alpha[i,j], Tau= ter[i], Beta=.5, 
                        Nu= delta[i,k], Eta = eta[i])
                    tempx = np.sign(np.real(tempout))
                    tempt = np.abs(np.real(tempout))
                    rt[k*2+j,indextrack] = tempt
                    acc[k*2+j,indextrack] = (tempx + 1)/2
                    indextrack += ntrials
    elif model=='modelp':
        for k in range(nconds):
            indextrack = np.arange(ntrials)
            for i in indices:
                tempout = simulratcliff(N=ntrials, Alpha= alpha[i], Tau= ter[i,j], Beta=.5, 
                    Nu= delta[i,int(k/2)], Eta = eta[i])
                tempx = np.sign(np.real(tempout))
                tempt = np.abs(np.real(tempout))
                rt[k,indextrack] = tempt
                acc[k,indextrack] = (tempx + 1)/2
                indextrack += ntrials
    return rt,acc
