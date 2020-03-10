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
"""
Attributes
----------
    x_init: double
        x-coordinate of the initial point.
    y_init: double
        y-coordinate of the initial point.
    beta: double
        Inverse temperature of the associated overdamped Langevin dynamic.
    dt: double
        Time step of the discretization.
    K: (C: np.intc) (py:int)
        Minimum Number of particles to be killed at each iteration.
        Theoretical guarantee is availble when K = 1. In fact, this is the only case where the efficient 
        variance estimator is availble.
    N: (C: long) (py: int)
        Number of particles at each iteration.
    n_max: (C: long) (py: int)
        Upper bound of the number of levels. This is introduced to control the memory allocation. There is no loss of memory 
        efficiency when iota is not set to be 0, since an adaptive procedure of memory allocation will be conducted in order to
        match the number of levels introduced by iota.
    iota: double
        Artificial step size of reaction coordinate. When iota is non-zero, the gAMS enters into Asymmetric SMC framework.
        The asymptotic variance is therefore available as a by-product of the simulation of IPS.
    
Return 
------
double: the estimation of probability.
double: the estimation of asympotic variance.

Remarks
-------
Notice that the asymptotic variance estimator is biased!
The (stochastic) bias is of order O_p(1/N). Hence, when N is small, one may encounter the case where the estimation of the
asymptotic variance is negative! This is totally normal, which indicates that the number of particles N is too low for the current
problem. 
When iota is set to be small, it is also possible to encounter some "bias" in the varianc estimation. The essential problem is that
the consistency of thte variance estimator is guaranteed in the sense that the number of levels is finite, i.e. not too big w.r.t. the
number of particles N. Hence, it is encouraged to use relatively larger iota in order to ensure that the number of levels is not too big
(or even bigger than N). This costs a slightly larger asymptotic variance. 
In fact, the problem mentioned above can be solved by using the unbiased variance estimator, which is also consistent w.r.t. N. However, 
that variance estimator is not efficient. More precisely, the time complexity is O(N*N*n). Therefore, it is not implemented in the current 
version of gAMS.
"""

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



