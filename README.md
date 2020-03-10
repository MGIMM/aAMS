# aAMS

Asymmetric SMC version of Adaptive Multilevel Splitting with efficient variance estimator implemented in `Cython`.


## Installation

Clone the git repository:
```
git clone https://github.com/MGIMM/aAMS.git
```
Install with `pip`:

```
cd aAMS
```
and
```
pip install .
```

## Usage


```python
from aAMS import aAMS
from time import time
from tqdm import tqdm
import numpy as np



# Naive Monte Carlo

N_test = 100000
list_mc = []
for i in tqdm(range(1000)):
    list_mc += [naive_MC(N = np.intc(N_test),
                         x_init = -0.75,
                         y_init = 0.0,
                         beta = 4.1,
                         dt = 0.01)]
print("estimation of rare event:", np.mean(list_mc))
print("naive asymptotic variance estimator:", N_test*np.var(list_mc))

### output:
# 100%|██████████| 1000/1000 [01:00<00:00, 16.43it/s]
# 
# estimation of rare event: 2.9700000000000004e-05
# naive asymptotic variance estimator: 3.0891e-05



# Adaptive Multilevel Splitting

t0=time()
m,v = aAMS(x_init = -0.75,
     y_init = 0.0,
     beta = 4.1,
     dt = 0.01,
     level_star = 1.75,
     N = 10000,       
     K = 1,
     iota=0.1,
     n_max = 100000
     )
print("mean:",m,"\nvar:",v)
print(time()-t0,"seconds used.")

### output:
# mean: 2.9496228881400097e-05 
# var: 3.937245849267889e-08
# 0.8391783237457275 seconds used.


# Comparison with naive variance estimator

N_test = 5000
K_test = 1
n_sim = 500
list_gams = np.zeros(n_sim) 
list_var = np.zeros(n_sim) 
for i in tqdm(range(n_sim)):
    list_gams[i],list_var[i] = aAMS(x_init = -0.75,
                                    y_init = 0.0,
                                    beta = 4.1,
                                    dt = 0.01,
                                    level_star = 1.75,
                                    N = N_test,       
                                    K = K_test,
                                    iota=0.1,
                                    n_max = 100000
                                    )

print("estimation of rare event:", np.mean(list_gams))
print("naive asymptotic variance estimator:", N_test*np.var(list_gams))
print("mean asymptotic variance estimator:", np.mean(list_var))
print("VoV:", np.var(list_var))
print("ideal variance for gAMS:",-np.log(np.mean(list_gams))*np.mean(list_gams)**2)

### output:
# 100%|██████████| 500/500 [03:23<00:00,  2.45it/s]
# estimation of rare event: 2.9022656130470198e-05
# naive asymptotic variance estimator: 3.9672002621255104e-08
# mean asymptotic variance estimator: 3.8650810422399494e-08
# VoV: 7.747465305795918e-17
# ideal variance for gAMS: 8.800025686164234e-09


```

## Remarks

* The purpose of this package is to provide a well optimized implementation of
  gAMS algorithm based on Asymmetric SMC framework that **can run on a personal laptop**, which mainly contributes to the numerical part of my PHD thesis. 

* Since the computation of
  the variance estimator is highly non-trivial, a readable code can be found in `./aAMS/aAMS.pyx`.

* Currently, the code is based on the overdamped Langevin dynamic with a
  three-hole potential. The generalization to the general Markov dynamic in a
  high-dimensional setting is straightforward. The computational costs are linear
  w.r.t. the underlying dimension of the Markov process. Contact me if you have
  any question.



## Reference

Generalized Adaptive Multilevel Splitting. \[[pdf](https://arxiv.org/pdf/1505.02674.pdf)\]

Asymmetric Sequential Monte Carlo. \[[pdf](https://mgimm.github.io/doc/du19.pdf)\]



