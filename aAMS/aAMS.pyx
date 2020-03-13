# distutils: language = c++
#from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libcpp.algorithm cimport sort,unique
from libcpp.vector cimport vector
from libc.math cimport ceil, floor
from libc.math cimport pow as pow_C 

from libc.math cimport exp, sqrt, sin, cos, log, M_PI
from libc.stdlib cimport rand, RAND_MAX


# import utils and model setting
# from ._utils cimport my_mean
# from ._utils cimport my_mean2
# from .models cimport _update_state
# from .models cimport naive_M 
# from .models cimport xi

#from joblib import Parallel,delayed
#import multiprocessing

"""
    _      _        ____    ___    ____    ___   _____   _   _   __  __ 
   / \    | |      / ___|  / _ \  |  _ \  |_ _| |_   _| | | | | |  \/  |
  / _ \   | |     | |  _  | | | | | |_) |  | |    | |   | |_| | | |\/| |
 / ___ \  | |___  | |_| | | |_| | |  _ <   | |    | |   |  _  | | |  | |
/_/   \_\ |_____|  \____|  \___/  |_| \_\ |___|   |_|   |_| |_| |_|  |_|
"""
########## ALGORITHM ##########
# naive Monte Carlo 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline double naive_MC(long N,
                             double dt,
                             double beta,
                             double x_init = -0.9,
                             double y_init = 0.0) nogil:
    
    """
    An implementation of Naive Monte Carlo.
    """
    cdef int i
    cdef double N_double = N
    cdef double SumG = 0.0
    cdef double sqrt_inv_temp_dt = sqrt(2.0*dt/beta)
    for i in range(N):
        SumG += naive_M(dt,sqrt_inv_temp_dt,x_init,y_init)
    return SumG/N_double


    
# adaptive algo:

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _M_adaptive(double dt,
                               double sqrt_inv_temp_dt,
                               vector[double] &particle_x,
                               vector[double] &particle_y,
                               vector[double] &_list_level,
                               double iota = 0.1,
                               double rho = 0.2,
                               int max_iter = 1024) nogil: # tested
    
    """
    core mechanism of Markov kernel on the path space.
    """
    cdef double _level,max_level
    cdef double _x = particle_x.front()
    cdef double _y = particle_y.front()
    
    
    cdef int _iter = 0
    cdef int tau = 0 # first hitting time of the max level
    particle_x.clear()
    particle_y.clear()
    _list_level.clear()
    
    # particle_x.reserve(max_iter)
    # particle_y.reserve(max_iter)
    # list_level.reserve(max_iter)
    
    particle_x.push_back(_x)
    particle_y.push_back(_y)
    
    _level = xi(_x,_y,iota)
    _list_level.push_back(_level)
    max_level =  _level
    
    _not_in_AB = True
    #sqrt_inv_temp_dt = sqrt(2.0*dt/beta)
    
    while _not_in_AB and _iter <max_iter:
        _iter += 1
        _x,_y = _update_state(_x = _x,
                              _y = _y,
                              dt = dt,
                              sqrt_inv_temp_dt = sqrt_inv_temp_dt)
        
        _level = xi(_x,_y,iota)
        
        particle_x.push_back(_x)
        particle_y.push_back(_y)
        _list_level.push_back(_level)
        
        if _level > max_level:
            max_level = _level
        #if (_x-A_x)*(_x-A_x) +(_y-A_y)*(_y-A_y) < rho*rho or (_x-B_x)*(_x-B_x)+(_y-B_y)*(_y-B_y) < rho*rho:
        if (_x+1.0)*(_x+1.0) +_y*_y < rho*rho or (_x-1.0)*(_x-1.0)+_y*_y < rho*rho:
            _not_in_AB = False
        else:
            _not_in_AB = True 
        
    tau = 0
    while _list_level[tau] < max_level and tau < max_iter:
        tau += 1
    
    
    # erase the trajectory after the particle (in the path space) reaches its maximum level
    if tau >0:
        particle_x.erase(particle_x.begin()+tau+1, particle_x.end())
        particle_y.erase(particle_y.begin()+tau+1, particle_y.end())
        _list_level.erase(_list_level.begin()+tau+1, _list_level.end())
        
    return max_level 




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double Mutation_adaptive(double target_level,
                                     double dt,
                                     double sqrt_inv_temp_dt,
                                     vector[double] &particle_x,
                                     vector[double] &particle_y,
                                     vector[double] &_list_level,
                                     double iota = 0.1
                                     ) nogil:
    """
    Markov kernel on the path space.
    """
    #cdef int T = list_level.size()
    cdef int i
    cdef int _tau = 0
    cdef double _max_level
    while _list_level[_tau] <= target_level:
        _tau += 1
    if _tau >0:
        particle_x.erase(particle_x.begin(), particle_x.begin()+_tau)
        particle_y.erase(particle_y.begin(), particle_y.begin()+_tau)
        _list_level.erase(_list_level.begin(), _list_level.begin()+_tau)
        
    _max_level = _M_adaptive(dt = dt,
                             sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                             particle_x = particle_x,
                             particle_y = particle_y,
                             _list_level = _list_level,
                             iota = iota)
        
    
    return _max_level
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void lazy_Mutation_adaptive(double target_level,
                                        double dt,
                                        double sqrt_inv_temp_dt,
                                        vector[double] &particle_x,
                                        vector[double] &particle_y,
                                        vector[double] &_list_level
                                        ) nogil:
    """
    lazy kernel for aSMC, which essentially truncates the particle on the path space.
    """
    #cdef int T = list_level.size()
    cdef int i
    cdef int _tau = 0
    cdef double _max_level
    while _list_level[_tau] <= target_level:
        _tau += 1
    if _tau >0:
        particle_x.erase(particle_x.begin(), particle_x.begin()+_tau)
        particle_y.erase(particle_y.begin(), particle_y.begin()+_tau)
        _list_level.erase(_list_level.begin(), _list_level.begin()+_tau)
        
    
    
    










@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline (double,double) _gAMS_asymptotic(double x_init,
                                             double y_init,
                                             double beta,
                                             double dt,
                                             double level_star,
                                             int K,
                                             long N,
                                             long n_max = 100,
                                             double iota = 0.1
                                             ):
    """
    C-version of gAMS algo. The full description can be found in python wrapper.
    """
    # initialization:
    cdef long i
    cdef double N_double = N
    #cdef double[:,:,:] IPS = np.zeros((n_max+1,N,2))
    #cdef double[:,:] SH = np.zeros((n,N))
    
    
    
    
    cdef vector[double] sorted_max_level
    sorted_max_level.reserve(N)
    
    cdef double current_level
    cdef double _level
    
    cdef double sqrt_inv_temp_dt = sqrt(2.0*dt/beta)
    cdef double p_hat = 1.0 
    cdef int N_surviving = 0
    cdef vector[int] I_surviving
    I_surviving.reserve(N)
    
    # trace the levels sequence of each particle at each layer
    cdef vector[vector[double]] list_level
    list_level.reserve(N)
    
    cdef vector[double] level_init
    level_init.reserve(1)
    level_init.push_back(xi(x_init,y_init))
    
    cdef vector[double] list_max_level
    list_max_level.reserve(N)
    # particles at current layer, we do not trace the whole particle system, we only trace the associated potetial values in G_mat, which contains all the info of SH.
    cdef vector[vector[double]] layer_x
    cdef vector[vector[double]] layer_y
    layer_x.reserve(N)
    layer_y.reserve(N)
    cdef int max_iter = 1024*2
    cdef vector[double] vec_x_init,vec_y_init
    vec_x_init.reserve(max_iter)
    vec_y_init.reserve(max_iter)
    vec_x_init.push_back(x_init)
    vec_y_init.push_back(y_init)
    
    # optimize memory allocation for EVE, GENE and G_mat
    cdef long num_level = n_max
    if iota != 0.0:
        num_level = int(ceil((level_star - level_init.front())/iota))
    cdef long _n_max = n_max
    if  num_level <= n_max:
        _n_max = num_level + 1 
        
    cdef long[:,:] EVE = np.zeros((_n_max+1,N), dtype = int)
    cdef long[:,:] GENE = np.zeros((_n_max+1,N), dtype = int)
    cdef double[:,:] G_mat = np.zeros((_n_max+1,N))
    cdef long[:,:] Theta = np.zeros((_n_max+1,N),dtype = int)
    
    cdef long T = 0
    cdef long ParentIndex 
    #cdef double U
    cdef double SumG = 0.0
    
    
    
    
    
    
    
    
    # for var estimation
    cdef double V_ddagger
    cdef vector[double] MeanG, MeanG2
    MeanG.reserve(N)
    MeanG2.reserve(N)
    cdef long p
    cdef double Normalizer = 1.0
    cdef double[:] ArrayEve = np.zeros(N)
    cdef double SumEve = 0.0
    cdef double NUM1 = 1.0
    
    # last layer
    cdef vector[double] _f
    _f.reserve(N)
    
    cdef long CurrentIndex
    # calculate tilde_V_dagger
    cdef double tilde_V_dagger = 0.0
    cdef double[:] MatrixEve = np.zeros(N)
    cdef double[:] SumMatrixEve = np.zeros(N)
    cdef long Index,IndexPrime
    cdef double F
    #cdef double SumEve
    cdef double SumCurrent
    
    
    
    
    # iteration:
    with nogil:
        # initilization of layer (i.e. N particles) and list of levels at current layer.
        for i in range(N):
            layer_x.push_back(vec_x_init)
            layer_y.push_back(vec_y_init)
            list_level.push_back(level_init)
        
        for i in range(N):
            _level = Mutation_adaptive(target_level = -100000.0,
                                       dt = dt,
                                       sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                       particle_x = layer_x[i],
                                       particle_y = layer_y[i],
                                       _list_level = list_level[i]
                                       )
            list_max_level.push_back(_level)
            sorted_max_level.push_back(_level)
            EVE[0,i] = i
        # level calculations
        sort(sorted_max_level.begin(),sorted_max_level.end())
        current_level = sorted_max_level[K-1]
        sorted_max_level.clear()
        # survival test
        for i in range(N):
            if list_max_level[i] > current_level:
                #G_mat[0,i] = 1.0
                #I_surviving.clear()
                I_surviving.push_back(i)
                N_surviving += 1 
            # else:
            #     G_mat[0,i] = 0.0
                
        # iteration:    
        while current_level < level_star and N_surviving > 0 and T < _n_max:
            SumG = 0.0 
            for i in range(N):
                if list_max_level[i]>current_level:
                    G_mat[T,i] = 1.0
                    SumG += 1.0 
                    
                    # mutation
                    ParentIndex = i
                    lazy_Mutation_adaptive(target_level = current_level,
                                           dt = dt,
                                           sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                           particle_x = layer_x[i],
                                           particle_y = layer_y[i],
                                           _list_level = list_level[i]
                                           )
                else:
                    #G_mat[T,i] = 0.0
                    ParentIndex = I_surviving[rucat(N_surviving)]
                    
                    # cloning
                    #layer_x[i].clear()
                    #layer_y[i].clear()
                    #list_level[i].clear()
                    
                    layer_x[i] = layer_x[ParentIndex]
                    layer_y[i] = layer_y[ParentIndex]
                    list_level[i] = list_level[ParentIndex]
                    
                    # mutation
                    _level = Mutation_adaptive(target_level = current_level,
                                               dt = dt,
                                               sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                               particle_x = layer_x[i],
                                               particle_y = layer_y[i],
                                               _list_level = list_level[i],
                                               iota = iota
                                               )
                    list_max_level[i] = _level
                    
                
                # tracing genealogy and survival history (in G_mat)
                EVE[T+1,i] = EVE[T,ParentIndex] 
                GENE[T,i] = ParentIndex
            
            #level calculations
            sorted_max_level = list_max_level
            sort(sorted_max_level.begin(),sorted_max_level.end())
            current_level = sorted_max_level[K-1]
            sorted_max_level.clear()
            
            # update
            I_surviving.clear()
            N_surviving = 0
            
            p_hat *= SumG/N_double            
            for i in range(N):
                if list_max_level[i] > current_level:
                    #G_mat[T+1,i] = 1.0
                    I_surviving.push_back(i)
                    N_surviving += 1
                #else:
                #    G_mat[T+1,i] = 0.0
                
            T += 1
        
        
        # last level
        if current_level >= level_star:
            SumG = 0.0
            for i in range(N):
                _f.push_back(naive_M(dt = dt,
                                     sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                     x_init = layer_x[i].front(),
                                     y_init = layer_y[i].front())) 
                SumG += _f.back()
            p_hat *= SumG/N_double
        else:
            p_hat = 0.0
            
            
            
        
        # Variance Estimation
        
        if p_hat != 0.0:
            ## calculate V_ddagger
            Normalizer = 1.0
            NUM1 = 1.0
            for p in range(T):
                MeanG.push_back(my_mean(G_mat[p,:]))
                MeanG2.push_back(my_mean2(G_mat[p,:]))
                Normalizer *= MeanG.back()
                #Normalizer *= MeanG[p]
            NUM1 = pow_C(N_double/(N_double-1.0), num_level)
            
            SumG = 0.0
            for i in range(N):
                ArrayEve[EVE[T,i]] += _f[i]*Normalizer
                SumG += _f[i]*Normalizer
                
            SumEve = 0.0
            for i in range(N):
                SumEve += ArrayEve[i]*ArrayEve[i]
                
            V_ddagger = p_hat*p_hat-(SumG*SumG - SumEve)*NUM1/(N_double*N_double)
            V_ddagger *= N_double
    
    
            for i in range(N):
                CurrentIndex = i
                for p in range(T):
                    ParentIndex =  GENE[T-p-1,CurrentIndex]
                    Theta[T-p-1,i] = ParentIndex
                    CurrentIndex = ParentIndex
            
            #Normalizer = 1.0
            #for p in range(num_level):
            #    Normalizer *= float(N)/float(N-1)
            Normalizer *= sqrt(NUM1)
            #Normarlizer = sqrt(Normalizer)
            # for p in range(T):
            #     Normalizer *= MeanG[p]
            
            ## calculate tilde_V_dagger
            tilde_V_dagger = 0.0
            for p in range(T):
                for i in range(N):
                    Index = Theta[p,i]
                    IndexPrime = Theta[p+1,i]
                    F = _f[i]*Normalizer/MeanG[p]
                    MatrixEve[i] = G_mat[p,IndexPrime] * sqrt(MeanG2[p]) * F
                for i in range(N):
                    SumMatrixEve[EVE[T,i]] += MatrixEve[i]
                    
                SumEve = 0.0
                for i in range(N):
                    SumEve += SumMatrixEve[i]*SumMatrixEve[i]
                for i in range(N-1):
                    MatrixEve[0] += MatrixEve[i+1]
                SumCurrent = MatrixEve[0]*MatrixEve[0] - SumEve
                tilde_V_dagger += SumCurrent/(N_double*N_double)
                for i in range(N):
                    SumMatrixEve[i] = 0.0
                    MatrixEve[i] = 0.0
            
            
            
    
            return p_hat, V_ddagger+tilde_V_dagger
        else:
            return p_hat,0.0



    
 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline (double,double) _gAMS_non_asymptotic(double x_init,
                                                 double y_init,
                                                 double beta,
                                                 double dt,
                                                 double level_star,
                                                 int K,
                                                 long N,
                                                 long n_max = 100,
                                                 double iota = 0.1
                                                 ):
    """
    C-version of gAMS algo. The full description can be found in python wrapper.
    """
    # initialization:
    cdef long i
    cdef double N_double = N
    #cdef double[:,:,:] IPS = np.zeros((n_max+1,N,2))
    #cdef double[:,:] SH = np.zeros((n,N))
    
    
    
    
    cdef vector[double] sorted_max_level
    sorted_max_level.reserve(N)
    
    cdef double current_level
    cdef double _level
    
    cdef double sqrt_inv_temp_dt = sqrt(2.0*dt/beta)
    cdef double p_hat = 1.0 
    cdef int N_surviving = 0
    cdef vector[int] I_surviving
    I_surviving.reserve(N)
    
    # trace the levels sequence of each particle at each layer
    cdef vector[vector[double]] list_level
    list_level.reserve(N)
    
    cdef vector[double] level_init
    level_init.reserve(1)
    level_init.push_back(xi(x_init,y_init))
    
    cdef vector[double] list_max_level
    list_max_level.reserve(N)
    # particles at current layer, we do not trace the whole particle system, we only trace the associated potetial values in G_mat, which contains all the info of SH.
    cdef vector[vector[double]] layer_x
    cdef vector[vector[double]] layer_y
    layer_x.reserve(N)
    layer_y.reserve(N)
    cdef int max_iter = 1024*2
    cdef vector[double] vec_x_init,vec_y_init
    vec_x_init.reserve(max_iter)
    vec_y_init.reserve(max_iter)
    vec_x_init.push_back(x_init)
    vec_y_init.push_back(y_init)
    
    # optimize memory allocation for EVE, GENE and G_mat
    cdef long num_level = n_max
    if iota != 0.0:
        num_level = int(ceil((level_star - level_init.front())/iota))
    cdef long _n_max = n_max
    if  num_level <= n_max:
        _n_max = num_level + 1 
        
    cdef long[:,:] EVE = np.zeros((_n_max+1,N), dtype = int)
    cdef long[:,:] GENE = np.zeros((_n_max+1,N), dtype = int)
    cdef double[:,:] G_mat = np.zeros((_n_max+1,N))
    cdef long[:,:] Theta = np.zeros((_n_max+1,N),dtype = int)
    
    cdef long T = 0
    cdef long ParentIndex 
    #cdef double U
    cdef double SumG = 0.0
    
    
    # Variance Estimation
    # for var estimation
    cdef vector[double] MeanG, MeanG2, MeanGdot2
    MeanG.reserve(N)
    MeanG2.reserve(N)
    MeanGdot2.reserve(N)
    cdef double ProdCouple
    cdef long p
    cdef long Index1,Index2,ParentIndex1,ParentIndex2
    cdef double NUM1 = 1.0
    cdef long j
    
    # last layer
    cdef vector[double] _f
    _f.reserve(N)
    
    cdef double V_nothing = 0.0
    
    
    
    
    
    
    
    
    
    
    # iteration:
    with nogil:
        # initilization of layer (i.e. N particles) and list of levels at current layer.
        for i in range(N):
            layer_x.push_back(vec_x_init)
            layer_y.push_back(vec_y_init)
            list_level.push_back(level_init)
        
        for i in range(N):
            _level = Mutation_adaptive(target_level = -100000.0,
                                       dt = dt,
                                       sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                       particle_x = layer_x[i],
                                       particle_y = layer_y[i],
                                       _list_level = list_level[i]
                                       )
            list_max_level.push_back(_level)
            sorted_max_level.push_back(_level)
            EVE[0,i] = i
        # level calculations
        sort(sorted_max_level.begin(),sorted_max_level.end())
        current_level = sorted_max_level[K-1]
        sorted_max_level.clear()
        # survival test
        for i in range(N):
            if list_max_level[i] > current_level:
                #G_mat[0,i] = 1.0
                #I_surviving.clear()
                I_surviving.push_back(i)
                N_surviving += 1 
            # else:
            #     G_mat[0,i] = 0.0
                
        # iteration:    
        while current_level < level_star and N_surviving > 0 and T < _n_max:
            SumG = 0.0 
            for i in range(N):
                if list_max_level[i]>current_level:
                    G_mat[T,i] = 1.0
                    SumG += 1.0 
                    
                    # mutation
                    ParentIndex = i
                    lazy_Mutation_adaptive(target_level = current_level,
                                           dt = dt,
                                           sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                           particle_x = layer_x[i],
                                           particle_y = layer_y[i],
                                           _list_level = list_level[i]
                                           )
                else:
                    #G_mat[T,i] = 0.0
                    ParentIndex = I_surviving[rucat(N_surviving)]
                    
                    # cloning
                    #layer_x[i].clear()
                    #layer_y[i].clear()
                    #list_level[i].clear()
                    
                    layer_x[i] = layer_x[ParentIndex]
                    layer_y[i] = layer_y[ParentIndex]
                    list_level[i] = list_level[ParentIndex]
                    
                    # mutation
                    _level = Mutation_adaptive(target_level = current_level,
                                               dt = dt,
                                               sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                               particle_x = layer_x[i],
                                               particle_y = layer_y[i],
                                               _list_level = list_level[i],
                                               iota = iota
                                               )
                    list_max_level[i] = _level
                    
                
                # tracing genealogy and survival history (in G_mat)
                EVE[T+1,i] = EVE[T,ParentIndex] 
                GENE[T,i] = ParentIndex
            
            #level calculations
            sorted_max_level = list_max_level
            sort(sorted_max_level.begin(),sorted_max_level.end())
            current_level = sorted_max_level[K-1]
            sorted_max_level.clear()
            
            # update
            I_surviving.clear()
            N_surviving = 0
            
            p_hat *= SumG/N_double            
            for i in range(N):
                if list_max_level[i] > current_level:
                    #G_mat[T+1,i] = 1.0
                    I_surviving.push_back(i)
                    N_surviving += 1
                #else:
                #    G_mat[T+1,i] = 0.0
                
            T += 1
        
        
        # last level
        if current_level >= level_star:
            SumG = 0.0
            for i in range(N):
                _f.push_back(naive_M(dt = dt,
                                     sqrt_inv_temp_dt = sqrt_inv_temp_dt,
                                     x_init = layer_x[i].front(),
                                     y_init = layer_y[i].front())) 
                SumG += _f.back()
            p_hat *= SumG/N_double
        else:
            p_hat = 0.0
            
            
            
        
        

        if p_hat != 0.0:
            NUM1 = N_double/(N_double-1.0)
            for p in range(T):
                MeanG.push_back(my_mean(G_mat[p,:]))
                MeanG2.push_back(my_mean2(G_mat[p,:]))
                MeanGdot2.push_back((MeanG[p]*MeanG[p] - MeanG2[p]/N_double)*NUM1)
                #Normalizer *= MeanG[p]
            # # construct Theta
            # for i in range(N):
            #     CurrentIndex = i
            #     for p in range(T):
            #         ParentIndex =  GENE[T-p-1,CurrentIndex]
            #         Theta[T-p-1,i] = ParentIndex
            #         CurrentIndex = ParentIndex
            
            for i in range(N-1):
                for j in range(N-i-1): # real index: j+i+1
                    Index1 = i
                    Index2 = i+j+1
                    if EVE[T,Index1] != EVE[T,Index2]:
                        ProdCouple = _f[Index1]*_f[Index2]
                        for p in range(T):
                            ParentIndex1 = GENE[T-p-1,Index1]
                            ParentIndex2 = GENE[T-p-1,Index2]
                            if G_mat[T-p-1,Index1] == 1.0 and G_mat[T-p-1,Index2] == 1.0:
                                ProdCouple *= MeanGdot2[T-p-1]
                            else:
                                ProdCouple *= MeanG[T-p-1]*MeanG[T-p-1]*NUM1
                            Index1 = ParentIndex1
                            Index2 = ParentIndex2

                        V_nothing += ProdCouple 
            V_nothing *= 2.0
            V_nothing /= (N_double-1.0)*N_double


            
            
            
            
            return p_hat, p_hat*p_hat - V_nothing
        else:
            return p_hat, 0.0
    
    
# python wrapper
def aAMS(x_init,
         y_init,
         beta,
         dt,
         level_star,
         K,
         N,
         n_max = 100,
         iota = 0.1,
         var_estimation = "asymptotic"
         ):
    """
    gAMS implementation based on Asymmetric SMC with efficient asymptotic variance estimation.
    
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
        var_estimation: "asymptotic" or "non-asymptotic"
            Whether the efficient (biased) asymptotic variance estimator or the
            non-asymptotic (unbiased but not efficient) variance estimator
            should be implemented. The time complexity are respectively O(TN) and O(TN^2).
        
    Return 
    ------
    double: the estimation of probability.
    double: the estimation of asympotic variance.
    
    Remarks
    -------
    Notice that the asymptotic variance estimator is biased!  The (stochastic)
    bias is of order O_p(1/N). Hence, when N is small, one may encounter the
    case where the estimation of the asymptotic variance is completely
    irrelevant! This indicates that the number of particles N is too low for
    the current problem. In this case, one should use non-asymptotic variance
    estimator.  However, when N is larger (typically larger than 500), it is
    expected that the difference between non-asymptotic variance estimator
    (multiplied by N) and asymptotic variance estimator is very small.
    Therefore, we recomment to use the asymptotic variance estimator, which is
    more efficient by design.  When iota is set to be small, it is also
    possible to encounter some "bias" in the varianc estimation. The essential
    problem is that the consistency of thte variance estimator is guaranteed in
    the sense that the number of levels is finite, i.e. not too big w.r.t. the
    number of particles N. Hence, it is encouraged to use relatively larger
    iota in order to ensure that the number of levels is not too big (or even
    bigger than N). This costs a slightly larger asymptotic variance.  However,
    there will be no problem for non-asymptotic variance estimator, even for
    the choice iota = 0. In fact, the unbiasedness is trivial by replacing the
    martigale structure by local martigale structure (cf. du 2019, aSMC). 
    """
    if var_estimation == "asymptotic":
        return _gAMS_asymptotic(x_init,
                                y_init,
                                beta,
                                dt,
                                level_star,
                                np.intc(K),
                                int(N),
                                int(n_max),
                                iota
                                )
        
    elif var_estimation == "non-asymptotic":
        return _gAMS_non_asymptotic(x_init,
                                    y_init,
                                    beta,
                                    dt,
                                    level_star,
                                    np.intc(K),
                                    int(N),
                                    int(n_max),
                                    iota
                                    )


"""
 _   _   _____   ___   _       ____  
| | | | |_   _| |_ _| | |     / ___| 
| | | |   | |    | |  | |     \___ \ 
| |_| |   | |    | |  | |___   ___) |
 \___/    |_|   |___| |_____| |____/ 
                                     
"""
########## UTILS ########## 
# basic utils 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double runif() nogil:
    return rand()/float(RAND_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cdef inline int rucat(int N) nogil:
    return int(floor(runif()*float(N)))
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (double,double) rnorm2() nogil:
    """
    Gaussian generator based on Box-Muller method.
    In particular, we use both of the Gaussians since the current problem is of dim 2.
    """
    cdef double u1, u2
    u1 = sqrt(-2.*log(runif()))
    u2 = 2.*M_PI*runif()
    cdef double G1, G2
    G1 = u1*cos(u2)
    G2 = u1*sin(u2)
    return G1,G2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double rnorm() nogil:
    """
    Single Gaussian generator based on Box-Muller methods.
    """
    cdef double u1, u2
    u1 = sqrt(-2.*log(runif()))
    u2 = 2.*M_PI*runif()
    cdef double G1
    G1 = u1*cos(u2)
    return G1

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline double starProduct(double[:] x, double[:] y) nogil:
#     return x[1]*y[1] + x[2]*y[3] + x[3]*y[2]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double my_mean(double[:] x) nogil:
    """
    mean for memoryview.
    """
    cdef double sumX = 0.0
    cdef long j
    cdef long N = x.shape[0]
    for j in range(N):
        sumX += x[j]
    return sumX/float(N)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double my_mean2(double[:] x) nogil:
    """
    2-moment for memoryview.
    """
    cdef double sumX = 0.0
    cdef int j
    cdef long N = x.shape[0]
    for j in range(N):
        sumX += x[j]*x[j]
    return sumX/float(N)



"""
 __  __    ___    ____    _____   _       ____  
|  \/  |  / _ \  |  _ \  | ____| | |     / ___| 
| |\/| | | | | | | | | | |  _|   | |     \___ \ 
| |  | | | |_| | | |_| | | |___  | |___   ___) |
|_|  |_|  \___/  |____/  |_____| |_____| |____/ 
"""
########## MODELS ##########
# setting

## metastable states
# cdef double RHO = 0.2
# cdef double A_x = -1.0
# cdef double A_y = 0.0
# cdef double B_x = 1.0
# cdef double B_y = 0.0

## three-hole potential
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double V_func(double x, double y) nogil:
    """
    Potential function. Only used for visualization
    """
    cdef double x2, y2, y13, y35, _V
    x2 = x*x
    y2 = y*y
    #y13 = (y-0.33333333333333333333333333)*(y-0.33333333333333333333333333)
    #y35 = (y-1.66666666666666666666666666)*(y-1.66666666666666666666666666)
    y13 = (y-1./3.)*(y-1./3.)
    y35 = (y-5./3.)*(y-5./3.)
    _V = 0.2*x2*x2 + 0.2*y13*y13 + 3.*exp(-x2 -y13) - 3.*exp(-x2-y35) - 5.*exp(-(x-1.)*(x-1.)-y2) - 5.*exp(-(x+1.)*(x+1.)-y2)
    return _V




# Overdamped Langevin Dynamic

## gradient of three-hole potential

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _Vx(double x,double y) nogil:
    """
    first component of grad V. Division is avoided for acceleration, at the cost of a tiny numerical bias.
    """
    cdef double dVx
    dVx = 0.8*x*x*x\
            -6.*x*exp(-x*x-(y-0.33333333333333333333333333333333)*(y-0.33333333333333333333333333333333))\
            +6.*x*exp(-x*x-(y-1.66666666666666666666666666666666)*(y-1.66666666666666666666666666666666))\
            +10.*(x-1.)*exp(-(x-1.)*(x-1.)-y*y)\
            +10.*(x+1.)*exp(-(x+1.)*(x+1.)-y*y)
    return dVx

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _Vy(double x,double y) nogil:
    """
    second component of grad V.
    """
    cdef double dVy
    dVy = 0.8*(y-0.333333333333333333333333)*(y-0.333333333333333333333333)*(y-0.333333333333333333333333)\
            -(6.*y-2.)*exp(-x*x-(y-0.33333333333333333333333333333333)*(y-0.33333333333333333333333333333333))\
            +(6.*y-10.)*exp(-x*x-(y-1.66666666666666666666666666666666)*(y-1.66666666666666666666666666666666))\
            +10.*y*(exp(-(x-1.)*(x-1.) - y*y)\
            + exp(-(x+1.)*(x+1.)-y*y))
    return dVy


## reaction coordinate

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double xi(double x, double y, double iota = 0.1) nogil:
    """
    double iota:
        is designed to control the discontinuity of the reaction coordinate, i.e. the reaction coordinate is artificially 
        transformed into a stepwise constant function, in order to implement the aSMC version of gAMS. In practice, there is no penalty for
        choosing a little iota if the computational cost is acceptable. 
    
    xi1 is the real continuous version of reaction coordinate.
    """
    cdef double result = xi1(x,y) 
    if iota > 0.0:
        result /= iota
        result = floor(result)*iota
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double xi1(double x, double y) nogil:
    return sqrt((x+1.)*(x+1.)+y*y) 

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double xi2(double x, double y) nogil:
    """
    another choice of reaction coordinate, which works worse than xi1.
    """
    return x 

## Markov update

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (double, double) _update_state(double _x,
                                           double _y, 
                                           double dt, 
                                           double sqrt_inv_temp_dt) nogil:
    """
    underlying Markov dynamics.
    """
    cdef double G1,G2,x_new,y_new
    G1,G2 = rnorm2()
    x_new = _x - _Vx(_x,_y)*dt+sqrt_inv_temp_dt*G1
    y_new = _y - _Vy(_x,_y)*dt+sqrt_inv_temp_dt*G2
    return x_new,y_new

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double naive_M(double dt,
                           double sqrt_inv_temp_dt,
                           double x_init = -0.75,
                           double y_init = 0.0,
                           double rho = 0.2) nogil:
    """
    last step of gAMS.
    """
    
    cdef double _x,_y
    cdef int _iter = 0
    _x = x_init
    _y = y_init
    
    
    _not_in_AB = True
    #sqrt_inv_temp_dt = sqrt(2.0*dt/beta)
    
    while _not_in_AB and _iter <100000:
        _iter += 1
        _x,_y = _update_state(_x,_y,dt, sqrt_inv_temp_dt)
        #if (_x-A_x)*(_x-A_x) +(_y-A_y)*(_y-A_y) < rho*rho or (_x-B_x)*(_x-B_x)+(_y-B_y)*(_y-B_y) < rho*rho:
        if (_x+1.0)*(_x+1.0) +_y*_y < rho*rho or (_x-1.0)*(_x-1.0)+_y*_y < rho*rho:
            _not_in_AB = False
        else:
            _not_in_AB = True 
    if _x>0.0:
        return 1.0
    else:
        return 0.0 
        
        
########## python wrapper ########## 
# py-version of potential function, for viz and test
def V_py(x,y):
    """
    py-version of potential function, for viz and test
    """
    return V_func(x,y)

