functions { 
  /* Model 5
   * Ratcliff diffusion log-PDF for a single response (adapted from brms 1.10.2 and hddm 0.7.8)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ndt: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: mean drift rate parameter across trials
   *   sddrift: standard deviation of drift rates across trials
   * Returns:  
   *   a scalar to be added to the log posterior 
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
    real<lower=0> delta_ressd;    // Between-participant and spatial attention condition variability in residual of drit rate 
    real<lower=0> alpha_ressd;    // Between-participant and spatial attention condition variability in residual of boundary 
    real<lower=0> tau_ressd;      // Between-participant and spatial attention condition variability in residual of non-decision time 
    real<lower=0> eta_ressd;      // Between-participant variability in standard deviation residual  of drift rate
    real<lower=0> delta_lambdasd; // Between-participant variability in coefficient of N200 latency for drift rate     
    real<lower=0> alpha_lambdasd; // Between-participant variability in coefficient of N200 latency for boundary     
    real<lower=0> tau_lambdasd;   // Between-participant variability in coefficient of N200 latency for non-decision time
    real<lower=0> eta_lambdasd;   // Between-participant variability in standard deviation of drift rate
    real<lower=0> n200condsd;     // Between-condition variability for spatial attention in N200 latency
    real<lower=0> n200trialsd;    // Between-trial variability in N200 latency 

    /* Hierarchical mu parameter*/                               
    vector[nconds] delta_reshier;     // Hierarchical residual of drift rate 
    vector[nspats] alpha_reshier;     // Hierarchical residual of boundary
    vector[nspats] tau_reshier;       // Hierarchical residual of Non-decision time
    vector<lower=0>[nspats] eta_reshier;       // Hierarchical residual trial-to-trial standard deviation of drift rate
    vector[nconds] delta_lambdahier;  // Hierarchical between-participant variability in coefficient of drift rate
    vector[nspats] alpha_lambdahier;  // Hierarchical between-participant variability in coefficient of alpha 
    vector[nspats] tau_lambdahier;    // Hierarchical between-participant variability in coefficient of N200 latency 
    vector<lower=0>[nspats] eta_lambdahier;    // Hierarchical  between-participant trial-to-trial standard deviation of drift rate
    vector<lower=0>[nconds] n200cond;          // Hierarchical between-condition for spatial attention in N200 latency
     
    /* participant-level main paameter*/
    matrix[nparts,nconds] delta_res;     // residual of drift rate for each participant and conditions
    matrix[nparts,nspats] alpha_res;     // residual of boundary for each participant and spatial attention
    matrix[nparts,nspats] tau_res;      // residual of Non-decision time for each participant and spatial attention
    matrix<lower=0>[nparts,nspats] eta_res;       // residual of Trial-to-trial standard deviation of drift rate
    matrix[nparts,nconds] delta_lambda; // coefficient paramter for each participant for drift rate
    matrix[nparts,nspats] alpha_lambda; // coefficient paramter for each participant for boundary
    matrix[nparts,nspats] tau_lambda;    // coefficient paramter for each participant for non-decision time
    matrix<lower=0>[nparts,nspats] eta_lambda;    // residual of Trial-to-trial standard deviation of drift rate
    matrix<lower=0, upper=.4>[nparts, nconds] n200sub;     // n200 mu parameter for each participant and spatial attention 

}
transformed parameters {
   vector[N_obs + N_mis] n200lat = append_row(n200lat_obs, n200lat_mis);
}
model {
    /* sigma paameter*/
    delta_ressd ~ gamma(1,1);    
    alpha_ressd ~ gamma(1,1);    
    tau_ressd ~ gamma(.1,1); 
    eta_ressd ~ gamma(1,1); 
    delta_lambdasd ~ gamma(1,1);  
    alpha_lambdasd ~ gamma(1,1);  
    tau_lambdasd ~ gamma(.1,1);
    eta_lambdasd ~ gamma(1,1); 
    n200trialsd ~ gamma(.1,1); 
    n200condsd ~ gamma(.1,1);


    /* Hierarchical mu paameter*/                               
    for (c in 1:nconds){
        delta_reshier[c] ~ normal(2, 4);
        delta_lambdahier[c] ~ normal(3,5);
    }
    for (c in 1:nspats){
        alpha_reshier[c] ~ normal(1, 2); 
        alpha_lambdahier[c] ~ normal(3,5);
        
        tau_reshier[c] ~ normal(.2,.4); 
        tau_lambdahier[c] ~ normal(.5,2); 
        
        eta_reshier[c] ~ normal(1,.5); 
        eta_lambdahier[c] ~ normal(2,4); 
    }
    for (c in 1:nconds){
        n200cond[c] ~ normal(.15,.1) T[0, .4]; 
    }
    
    /* participant-level main paameter*/
    for (p in 1:nparts) {

        for (c in 1:nconds) {
            delta_res[p,c] ~ normal(delta_reshier[c], delta_ressd);
            delta_lambda[p,c] ~ normal(delta_lambdahier[c], delta_lambdasd);
        }
        for (c in 1:nspats){
            alpha_res[p,c] ~ normal(alpha_reshier[c], alpha_ressd) T[0, 5];
            alpha_lambda[p,c] ~ normal(alpha_lambdahier[c], alpha_lambdasd);

            tau_res[p,c] ~ normal(tau_reshier[c], tau_ressd);
            tau_lambda[p,c] ~ normal(tau_lambdahier[c], tau_lambdasd);
            
            eta_res[p,c] ~ normal(eta_reshier[c], eta_ressd);
            eta_lambda[p,c] ~ normal(eta_lambdahier[c], eta_lambdasd);
        }    
        for (c in 1:nconds) {
             n200sub[p,c] ~ normal(n200cond[c], n200condsd) T[0, .4];
        } 
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
        y[i] ~ ratcliff_lpdf(alpha_res[participant[i],cond_spat[i]] + alpha_lambda[participant[i],cond_spat[i]]*n200lat[i], tau_res[participant[i],cond_spat[i]] + tau_lambda[participant[i],cond_spat[i]]*n200lat[i], .5, delta_res[participant[i],conds[i]] + delta_lambda[participant[i],conds[i]]*n200lat[i], eta_res[participant[i],cond_spat[i]] + eta_lambda[participant[i],cond_spat[i]]*n200lat[i]);
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
         log_lik[i] = ratcliff_lpdf(y[i] | alpha_res[participant[i],cond_spat[i]] + alpha_lambda[participant[i],cond_spat[i]]*n200lat[i], tau_res[participant[i],cond_spat[i]] + tau_lambda[participant[i],cond_spat[i]]*n200lat[i], .5, delta_res[participant[i],conds[i]] + delta_lambda[participant[i],conds[i]]*n200lat[i], eta_res[participant[i],cond_spat[i]] + eta_lambda[participant[i],cond_spat[i]]*n200lat[i]) + n200lat_lpdf[i];
    }
}