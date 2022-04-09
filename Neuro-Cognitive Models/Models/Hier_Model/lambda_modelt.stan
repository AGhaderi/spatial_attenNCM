functions { 
  /* Model 3 
   * Ratcliff diffusion log-PDF for a single response (adapted from brms 1.10.2 and hddm 0.7.8)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ndt: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: mean drift rate parameter across trials
   *   sddrift: standard deviation of drift rates across trials
   * Returns:  
   *   a scalar to be added to the log posterior n200sub
   */ 
   real ratcliff_lpdf(real Y, real boundary, 
                              real ndt, real bias, real drift, real sddrift) { 
    real X;
    X = (fabs(Y) - ndt); // Remove non-decision time   
    if (Y >= 0) {
    return wiener_lpdf( fabs(Y) | boundary, ndt, bias, drift )  + (  ( (boundary*(1-bias)*sddrift)^2 + 2*drift*boundary*(1-bias) - (drift^2)*X ) / (2*(sddrift^2)*X+2)  ) - log(sqrt((sddrift^2)*X+1)) - drift*boundary*(1-bias) + (drift^2)*X*0.5;
    } else {
    return wiener_lpdf( fabs(Y) | boundary, ndt, 1-bias, -drift ) + (  ( (boundary*bias*sddrift)^2 - 2*drift*boundary*bias - (drift^2)*X ) / (2*(sddrift^2)*X+2)  ) - log(sqrt((sddrift^2)*X+1)) + drift*boundary*bias + (drift^2)*X*0.5;
    }
   }
} 
data {
    int<lower=1> N_obs;       // Number of trial-level observations
    int<lower=1> N_mis;       // Number of trial-level missing data
    int<lower=1> ncohers;     // Number of coherence conditions {2}
    int<lower=1> nspats;      // Number of spatial conditions {2}
    int<lower=1> nparts;      // Number of participants
    int<lower=1> nconds;      // Number of conditions
    real y[N_obs + N_mis];    // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> participant[N_obs + N_mis]; // participant index for each trial
    int<lower=1> cond_coher[N_obs + N_mis];  // High and low coherence conditions index {1, 2} for each trial
    int<lower=1> cond_spat[N_obs + N_mis];   // Prioritized and non-prioritized spatial conditions index {1, 2} for each trial
    int<lower=1> conds[N_obs + N_mis];       // Conditions index {1, 2, 3, 4} for each trial
    vector<lower=0>[N_obs] n200lat_obs;      // N200 Latency for observed trials
}
parameters {
    vector<lower=0.101, upper=0.248>[N_mis] n200lat_mis;  // vector of missing data for n200 latency  
    
    /* sigma paameter*/
    real<lower=0> deltasd;      // Between-participant and coherence condition variability in drift rate 
    real<lower=0> alphasd;      // Between-participant variability in boundary    
    real<lower=0> etasd;        // Between-participant variability in standard deviation of drift rate
    real<lower=0> ressd;        // Between-participant and spatial attention condition variability in residual of non-decision time 
    real<lower=0> n200condsd;   // Between-condition variability for spatial attention in N200 latency
    real<lower=0> lambdasd;     // Between-participant variability in coefficient of N200 latency     
    real<lower=0> n200trialsd;  // Between-trial variability in N200 latency 

    /* Hierarchical mu parameter*/                               
    vector<lower=0>[ncohers] deltahier;  // Hierarchical drift rate for high and low coherences
    real alphahier;             // Hierarchical boundary
    real<lower=0> etahier;               // Hierarchical trial-to-trial standard deviation of drift rate
    real reshier;               // Hierarchical residual of Non-decision time
    vector<lower=0>[nconds] n200cond;    // Hierarchical between-condition for spatial attention in N200 latency
    vector[nspats] lambdahier;   // Hierarchical between-participant variability in coefficient of N200 latency 
     
    /* participant-level main paameter*/
    matrix[nparts,ncohers] delta; // drift rate for each participant and coherences conditions
    vector[nparts] alpha;         // Boundary boundary for each participant
    vector<lower=0>[nparts] eta;           // Trial-to-trial standard deviation of drift rate
    vector[nparts] res;          // residual of Non-decision time for each participant
    matrix<lower=0, upper=.4>[nparts, nconds] n200sub;  // n200 mu parameter for each participant and spatial attention 
    matrix[nparts,nspats] lambda;       // coefficient paramter for each participant
}
transformed parameters {
   vector[N_obs + N_mis] n200lat = append_row(n200lat_obs, n200lat_mis);
}
model {
    /* sigma paameter*/
    deltasd ~ gamma(1,1); 
    etasd ~ gamma(1,1); 
    alphasd ~ gamma(1,1); 
    ressd ~ gamma(.1,1);    
    n200trialsd ~ gamma(.1,1); 
    n200condsd ~ gamma(.1,1);
    lambdasd ~ gamma(.1,1);

    /* Hierarchical mu paameter*/                               
    for (c in 1:ncohers){
        deltahier[c] ~ normal(2, 4);
    }
    reshier ~ normal(.2,.4); 
    for (c in 1:nspats){
        lambdahier[c] ~ normal(.5, 2); 
    } 
    alphahier ~ normal(1, 2);
    etahier ~ normal(1, 1);
    
    for (c in 1:nconds){
        n200cond[c] ~ normal(.15,.1); 
    }
    
    /* participant-level main paameter*/
    for (p in 1:nparts) {

        for (c in 1:ncohers) {
            delta[p,c] ~ normal(deltahier[c], deltasd);
        }
    
        res[p] ~ normal(reshier, ressd);
        for (c in 1:nspats){
            lambda[p,c] ~ normal(lambdahier[c], lambdasd);
        }

        for (c in 1:nconds) {
             n200sub[p,c] ~ normal(n200cond[c], n200condsd) T[0, .4];
        }
        alpha[p] ~ normal(alphahier, alphasd);
        eta[p] ~ normal(etahier, etasd);
 
    }
    
    for (i in 1:N_obs) {
        // Note that N200 latencies are censored between 100 and 250 ms for observed data
        n200lat_obs[i] ~ normal(n200sub[participant[i],conds[i]],n200trialsd) T[.101,.248];
    }    
    for (i in 1:N_mis) {
        // Note that N200 latencies are censored between 100 and 250 ms for missing data
        n200lat_mis[i] ~ normal(n200sub[participant[N_obs + i],conds[N_obs + i]],n200trialsd) T[.101,.248];
    }
    
    // Wiener likelihood
    for (i in 1:N_obs + N_mis) { 

        // Log density for DDM process
        y[i] ~ ratcliff_lpdf(alpha[participant[i]], res[participant[i]] + lambda[participant[i],cond_spat[i]]*n200lat[i], .5, delta[participant[i],cond_coher[i]], eta[participant[i]]);
    }
}
generated quantities { 
   vector[N_obs + N_mis] log_lik; 
   vector[N_obs + N_mis] n200lat_lpdf;
    
    // n200lat likelihood
    for (i in 1:N_obs+N_mis) {
        // Note that N200 latencies are censored between 100 and 250 ms for observed data
        n200lat_lpdf[i] = normal_lpdf(n200lat[i] | n200sub[participant[i],conds[i]],n200trialsd);
    }   
    
   // Wiener likelihood
    for (i in 1:N_obs+N_mis) {
        // Log density for DDM process
         log_lik[i] = ratcliff_lpdf(y[i] | alpha[participant[i]], res[participant[i]] + lambda[participant[i],cond_spat[i]]*n200lat[i], .5, delta[participant[i],cond_coher[i]], eta[participant[i]]) + n200lat_lpdf[i];
    }
}
