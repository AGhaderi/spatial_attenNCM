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
                              real ndt, real bias, real drift) { 
    real X;
    X = (fabs(Y) - ndt); // Remove non-decision time   
    if (Y >= 0) {
    return wiener_lpdf( fabs(Y) | boundary, ndt, bias, drift );
    } else {
    return wiener_lpdf( fabs(Y) | boundary, ndt, 1-bias, -drift );
    }
   }
} 
data {
    int<lower=1> N_obs;       // Number of trial-level observations
    int<lower=1> N_mis;       // Number of trial-level missing data
    int<lower=1> ncohers;     // Number of coherence conditions {2}
    int<lower=1> nspats;      // Number of spatial conditions {2}
    int<lower=1> nconds;      // Number of conditions
    real y[N_obs + N_mis];    // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> cond_coher[N_obs + N_mis];  // High and low coherence conditions index {1, 2} for each trial
    int<lower=1> cond_spat[N_obs + N_mis];   // Prioritized and non-prioritized spatial conditions index {1, 2} for each trial
    int<lower=1> conds[N_obs + N_mis];       // Conditions index {1, 2, 3, 4} for each trial
    vector<lower=0>[N_obs] n200lat_obs;      // N200 Latency for observed trials
}
parameters {
    vector<lower=0.05, upper=0.250>[N_mis] n200lat_mis;  // vector of missing data for n200 latency  
    
    real<lower=0> n200latstd;     // n200lat std

    /* main paameter*/
    real delta_res[nconds];      // drift rate for coherences conditions
    real delta_lambda[nconds];   // drift rate for coherences conditions
    real alpha_res[nspats];               // Boundary boundary
    real alpha_lambda[nspats];            // Boundary boundary
    real tau_res[nspats];                 // residual of Non-decision time
    real tau_lambda[nspats];              // coefficient paramter
    real<lower=0, upper=.5> n200sub[nconds];  // n200 mu parameter for each condition 
}
transformed parameters {
   vector[N_obs + N_mis] n200lat = append_row(n200lat_obs, n200lat_mis);
}
model {

    n200latstd ~ gamma(.1,1); 

    /* main paameter*/
    for (c in 1:nconds) {
        delta_res[c] ~ normal(2, 2);
        delta_lambda[c] ~ normal(2, 4);
    }

    for (c in 1:nspats){
        tau_res[c] ~ normal(.5, 1);
        tau_lambda[c] ~ normal(1, 2);
        alpha_res[c] ~ normal(1, 2); 
        alpha_lambda[c] ~ normal(1, 2);
    }

    for (c in 1:nconds) {
         n200sub[c] ~ normal(.15,.2) T[0,.5];
    }
    
    for (i in 1:N_obs) {
        // Note that N200 latencies are censored between 50 and 300 ms for observed data
        n200lat_obs[i] ~ normal(n200sub[conds[i]],n200latstd) T[.05,.250];
    }    
    for (i in 1:N_mis) {
        // Note that N200 latencies are censored between 50 and 300 ms for missing data
        n200lat_mis[i] ~ normal(n200sub[conds[N_obs + i]], n200latstd) T[.05,.250];
    }
    
    // Wiener likelihood
    for (i in 1:N_obs + N_mis) { 
    
        // Log density for DDM process
        y[i] ~ ratcliff_lpdf(alpha_res[cond_spat[i]] + alpha_lambda[cond_spat[i]]*n200lat[i], tau_res[cond_spat[i]] + tau_lambda[cond_spat[i]]*n200lat[i], .5, delta_res[conds[i]] + delta_lambda[conds[i]]*n200lat[i]);
    }
}
generated quantities { 
   vector[N_obs + N_mis] log_lik; 
   vector[N_obs + N_mis] n200lat_lpdf;
    
    // n200lat likelihood
    for (i in 1:N_obs+N_mis) {
        // Note that N200 latencies are censored between 50 and 300 ms for observed data
        n200lat_lpdf[i] = normal_lpdf(n200lat[i] | n200sub[conds[i]], n200latstd);
    }    
    
   // Wiener likelihood
    for (i in 1:N_obs+N_mis) {
        // Log density for DDM process
         log_lik[i] = ratcliff_lpdf(y[i] | alpha_res[cond_spat[i]] + alpha_lambda[cond_spat[i]]*n200lat[i], tau_res[cond_spat[i]] + tau_lambda[cond_spat[i]]*n200lat[i], .5, delta_res[conds[i]] + delta_lambda[conds[i]]*n200lat[i]) + n200lat_lpdf[i];
   }
}