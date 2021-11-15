functions { 
  /* Ratcliff diffusion log-PDF for a single response (adapted from brms 1.10.2 and hddm 0.7.8)
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
    int<lower=1> N;           // Number of trial-level observations
    int<lower=1> ncohers;     // Number of coherence conditions {2}
    int<lower=1> nspats;      // Number of spatil conditions {2}
    int<lower=1> nparts;      // Number of participants
    real y[N];                // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> participant[N];          // participant index
    int<lower=1> cond_coher[N];           // coherence conditions index {1, 2}
    int<lower=1> cond_spat[N];            // spatial attention conditions index {1, 2} 
}
parameters {

    /* std paameter*/
    real<lower=0> deltasd; // Between-condition-participant variability in mean drift rate 
    real<lower=0> tersd;   // Between-participant variability in non-decision time 
    real<lower=0> alphasd; // Between-condition-participant variability in Speed-accuracy trade-off
    real<lower=0> etasd;   // Between-participant variability in standard deviation of drift rate

    
    /* Hierarchical mu paameter*/                                                   
    vector<lower=0>[ncohers] deltahier;   // Hierarchical drift rate and coherence conditions
    real<lower=0> terhier;                // Hierarchical Non-decision time 
    vector<lower=0>[nspats] alphahier;    // Hierarchical boundary parameter (speed-accuracy tradeoff) and spatial attention conditions
    real<lower=0> etahier;                // Hierarchical trial-to-trial standard deviation of drift rate

    /* participant-level main parameter*/
    matrix<lower=0, upper=6>[nparts,ncohers] delta;  // drift rate for each participant and coherences conditions
    vector<lower=0, upper=1>[nparts] ter;            // Non-decision time for each participant
    matrix<lower=0, upper=3>[nparts,nspats] alpha;   // Boundary parameter for each participant and spatial spatial
    vector<lower=0, upper=3>[nparts] eta;            // Trial-to-trial standard deviation of drift rate

}
model {
    /* std parameters variablility*/
    deltasd ~ gamma(1,1);
    tersd ~ gamma(.3,1);
    alphasd ~ gamma(1,1);
    etasd ~ gamma(1,1);
    
    
    /* Hierarchical mu paameter*/ 
    for (c1 in 1:ncohers){
        deltahier[c1] ~ normal(2, 3);
    }
    for (c2 in 1:nspats){
        alphahier[c2] ~ normal(1, 2);
    }
    terhier ~ normal(.3,.2);
    etahier ~ normal(1, .5);


    /* participant-condition-level main parameter*/
    for (p in 1:nparts) {
        for (c1 in 1:ncohers) {
            delta[p,c1] ~ normal(deltahier[c1], deltasd) T[0, 6];
        }
        for (c2 in 1:nspats) {
            alpha[p,c2] ~ normal(alphahier[c2], alphasd) T[0, 3];  
        }      
        ter[p] ~ normal(terhier, tersd) T[0, 1];
        eta[p] ~ normal(etahier, etasd) T[0, 3];

    }
    // Wiener likelihood for observation data
    for (i in 1:N) {
        target += ratcliff_lpdf( y[i] | alpha[participant[i],cond_spat[i]], ter[participant[i]], .5, delta[participant[i],cond_coher[i]], eta[participant[i]]);
    }
}
generated quantities {
    vector[N] log_lik;    
    for (i in 1:N) {
        log_lik[i] = ratcliff_lpdf(y[i] | alpha[participant[i],cond_spat[i]], ter[participant[i]], .5, delta[participant[i],cond_coher[i]], eta[participant[i]]);
    }
}