# spatial_attenNCM
The current repository is for a project named "Neuro-cognitive models of single-trial EEG measures describe latent effects of spatial attention during perceptual decision making" at the Shahid Beheshti University

**Authors: Amin Ghaderi-Kangavari, Jamal Amani Rad, Kourosh Parand, & Michael D. Nunez**

### Citation
Ghaderi-Kangavari, A., Rad, J. A., Parand, K., & Nunez, M. D.  (2022). [Neuro-cognitive models of single-trial EEG measures describe latent effects of spatial attention during perceptual decision making](https://www.biorxiv.org/content/10.1101/2022.04.07.487571v1) bioRxiv 2022.04.07.487571; doi: https://doi.org/10.1101/2022.04.07.487571 

## Prerequisites

[Python 3 and Scientific Python libraries](https://www.anaconda.com/products/individual)

[pystan](https://pystan.readthedocs.io)


## Abstract 
Visual perceptual decision-making involves multiple components including visual encoding, attention, accumulation of evidence, and motor execution. Recent research suggests that EEG oscillations can identify the time of encoding and the onset of evidence accumulation during perceptual decision-making. Although scientists show that spatial attention improves participant performance in decision making, little is known about how spatial attention influences the individual cognitive components that gives rise to that improvement in performance. We found evidence in this work that both visual encoding time (VET) before evidence accumulation and other non-decision time process after or during evidence accumulation are influenced by spatial top-down attention, but not evidence accumulation itself. Specifically we used an open-source data set in which participants were informed about the location of a target stimulus in the visual field on some trials during a face-car perceptual decision-making task. Fitting neural drift-diffusion models to response time, accuracy, and single-trial N200 latencies (~ 125 to 225 ms post-stimulus) of EEG allowed us to separate the processes of visual encoding and the decision process from other non-decision time processes such as motor execution. These models were fit in a single step in a hierarchical Bayesian framework. Model selection criteria and comparison to model simulations show that spatial attention manipulates both VET and other non-decision time process.


## See also

[hddm package](https://github.com/hddm-devs/hddm): a package to fit and evaluate hierarchical Drift-Diffusion Models using pyMC. We addapted the Stan functions of DDMs with intrinsic trial-to-trial variability from the likelihood derivations in this package. 

[rlssm package](https://github.com/laurafontanesi/rlssm): a package to fit and evaluate hierarchical Drift-Diffusion Models (and Reinforcement Learning models) using pystan. Some of these functions were used in plotting.py of this repository.

[pyhddmjags package](https://github.com/mdnunez/pyhddmjags): some example scripts to fit and evaluate hierarchical Drift-Diffusion Models using pyjags and pystan.

[brms package](https://github.com/paul-buerkner/brms): a package to fit multiple models, including Drift-Diffusion Models, in Stan using R. We adapted the Stan functions of DDMs with intrinsic trial-to-trial variability from the Stan code generated from brms.
