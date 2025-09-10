import numpy as np
import matplotlib.pyplot as plt

import Class_site as site
import free_fermion_representation as f
import pickle
import os

from numba import njit



#%% order_parameter_delta_T functions

#function used to compute order parameter for number of floquet cycles = N_cycles 
# at a specific disorder delta and coupling/time term T

#good idea would be to unify method1 and method2

def order_parameter_delta_T_method1(model, fgs, T, delta, N_cycles, edgepar = None):
    orderpar = []
    loop_0 = []
    loop_e = []

    fgs.reset_cov_0_matrix()
    fgs.reset_cov_e_matrix()

    _, _, _, R0 = fgs.floquet_operator_ham(T)
    _, _, _, Re = fgs.floquet_operator_ham(T, anyon=True)
    V0 = f.generate_disorder_term(model, fgs.Cov, delta, edgepar = edgepar)
    Ve = f.generate_disorder_term(model, fgs.Cov, delta, type="Anyon", edgepar = edgepar)
    R_V0 = f.floquet_operator(V0, T, alpha = np.pi/4.)
    R_Ve = f.floquet_operator(Ve, T, alpha = np.pi/4.)

    for _ in range(N_cycles):
        op, value_0, value_e = fgs.order_parameter()
        orderpar.append(op)
        loop_0.append(value_0)
        loop_e.append(value_e)
        fgs.update_cov_0_matrix(R_V0 @ R0) 
        fgs.update_cov_e_matrix(R_Ve @ Re)
        # this approach requires 3*N^3 operations 
        #vs doing 2 consecutive updates update(R0) -> update(R_V0) = 4*N^3 operations!
        #so the way we are doing it is computationally favorable

    op, value_0, value_e = fgs.order_parameter()
    orderpar.append(op)
    loop_0.append(value_0)
    loop_e.append(value_e) 

    return orderpar, loop_0, loop_e


def order_parameter_delta_T_method2(model, fgs, T, delta, N_cycles, edgepar = None):

    fgs.reset_cov_0_matrix()
    fgs.reset_cov_e_matrix()

    if delta != 0:
        V0 = f.generate_disorder_term(model, fgs.Cov, delta, edgepar = edgepar)
        Ve = f.generate_disorder_term(model, fgs.Cov, delta, type = "Anyon", edgepar = edgepar)
    else:
        V0 = 0
        Ve = 0
    #to spare running time!

    Rx = f.floquet_operator(-np.pi/4*(fgs.h0_x - V0), T, alpha = 1)
    Ry = f.floquet_operator(-np.pi/4*(fgs.h0_y - V0), T, alpha = 1)
    Rz = f.floquet_operator(-np.pi/4*(fgs.h0_z - V0), T, alpha = 1)
    R0 = Rz @ Ry @ Rx

    Rx_e = f.floquet_operator(-np.pi/4*(fgs.he_x - Ve), T, alpha = 1)
    Ry_e = f.floquet_operator(-np.pi/4*(fgs.he_y - Ve), T, alpha = 1)
    Rz_e = f.floquet_operator(-np.pi/4*(fgs.he_z - Ve), T, alpha = 1)
    Re = Rz_e @ Ry_e @ Rx_e

    orderpar = []
    loop_0 = []
    loop_e = []
    
    for _ in range(N_cycles):
        op, value_0, value_e = fgs.order_parameter()
        orderpar.append(op)
        loop_0.append(value_0)
        loop_e.append(value_e)
        fgs.update_cov_0_matrix(R0)
        fgs.update_cov_e_matrix(Re)

    op, value_0, value_e = fgs.order_parameter()
    orderpar.append(op)
    loop_0.append(value_0)
    loop_e.append(value_e) 
    return orderpar, loop_0, loop_e



#%% Fourier transform

@njit
def gaussian_window(n, sigma):
    window = np.empty(n)
    for i in range(n):
        window[i] = np.exp(-0.5 * (i / (sigma * n)) ** 2)
    return window

@njit
def cosine_window(n, t_list, nw):
    window = np.empty(n)
    for t in range(n):
        window[t] = np.cos(np.pi * t_list[t] / (2 * t_list[-1])) ** nw
    return window

def fourier_time(t_series, dt, sigma = 0.4, nw=4, gauss = True, normalize = False):
    """ Calculates the FFT of a time series, applying either a Gaussian window function if gauss = True, or a cosine if gauss = False. """

    # Gaussian or cosine window function
    n = len(t_series)
    t_list = np.arange(n)*dt

    if gauss:
        window = gaussian_window(n, sigma)
    else:
        window = cosine_window(n, t_list, nw)

    input_series = window * t_series

    # Fourier transform
    ft = np.fft.fft(input_series) / np.sqrt(n)

    # Take into account the additional minus sign in the time FT
    ft = np.append(ft, ft[0])
    ft = ft[::-1]
    ft = ft[:-1]

    if n != len(ft):
        raise ValueError("Length of Fourier transform is not equal to length of input time series!")

    if normalize:
        ft = ft/np.sqrt(np.sum(np.abs(ft)**2))

    ft_0, ft_pi = ft_0_and_ft_pi(ft)

    return ft, ft_0, ft_pi

def frequencies(n, dt=1):
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi 

    # order frequencies in increasing order
    end = np.argmin(freqs)
    freqs = np.append(freqs[end:], freqs[:end])

    if n % 2 == 1:
        freqs += np.pi*(n-1)/n
    else:
        freqs += np.pi

    return freqs

def ft_0_and_ft_pi(ft):

    ft_0 = ft[0]

    n = len(ft)

    if n % 2 == 1:
        ft_pi = (ft[(n-1)//2]+ft[(n+1)//2])/2 # // even though (n-1)/2 is always integer it will yield n.0, I use // to force it to be integer
    else:
        ft_pi = ft[n//2]

    return ft_0, ft_pi

#%% algorithm to run on cluster to create 2d phase diagram

#algorithm that computes data grid entry for N_shots or not depending on value of (delta,T)
#i.e. calculates N_shots functions (o.p.) per combination (T,delta), 
# and then averages eta(pi) - eta(o) over these N_shots

def compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = '1', edgepar = None, normalize = False):
    """ Computes a single entry in the data grid for given T and delta, averaging over N_shots if necessary. """

    if delta == 0 or N_shots < 2: 
        #no need to average when delta = 0, or N_shots = 1 because no random disorder is added in this case!
        #N_shots = 0 equivalent to N_shots = 1
        #if N_shots = 0 this is just the old algorithm

        if method == '1':
            orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, T, delta,  N_cycles, edgepar = edgepar)
        else: 
            orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, T, delta,  N_cycles, edgepar = edgepar)

        ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1, normalize = normalize)

        result = np.abs(ft_pi) - np.abs(ft_0)

        data_grid_entry = {
            'loop_0' : loop_0, #expectation value of loop with no anyon
            'loop_e' : loop_e, #expectation value of loop with e anyon
            'op_real': orderpar,
            'op_ft' : ft, 
            'result': result
        }

    else:

        loop_0_list = []
        loop_e_list = []
        orderpar_list = []
        op_ft_list = []
        result_list = []

        for _ in tqdm(range(N_shots), desc= f"Shots per delta = {delta:.3f}, T = {T:.3f})", leave = False):
            if method == '1':
                orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, T, delta,  N_cycles, edgepar = edgepar)
            else: 
                orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, T, delta,  N_cycles, edgepar = edgepar)

            ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1, normalize = normalize)

            result_list.append(np.abs(ft_pi) - np.abs(ft_0))
            loop_0_list.append(loop_0)
            loop_e_list.append(loop_e)
            orderpar_list.append(orderpar)
            op_ft_list.append(ft)

        result = np.mean(result_list)
        
        data_grid_entry = {
            'loop_0' : loop_0_list, #expectation value of loop with no anyon
            'loop_e' : loop_e_list, #expectation value of loop with e anyon
            'op_real': orderpar_list,
            'op_ft' : op_ft_list, 
            'result': result
        }
    
    return data_grid_entry


#function to obtain phase diagram without parallelization

def phase_diagram(model, T_list, delta_list, N_shots = 10, N_cycles = 10, method = '1', save_dir = None, general_dir = "phasediagram", edgepar = None, normalize = False):
    """ Computes the phase diagram over a grid of T and delta values, optionally saving results to disk. """

    fgs = f.FermionicGaussianRepresentation(model)
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)

    if save_dir is not None:

        for i, delta in enumerate(tqdm(delta_list, desc="Deltas")):
            for j, T in enumerate(tqdm(T_list, desc=f"T for delta={delta}", leave=False)):
        
                    full_dir = os.path.join(general_dir, save_dir)
                    os.makedirs(full_dir, exist_ok=True)
                    fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
                    fpath = os.path.join(full_dir, fname)

                    if os.path.exists(fpath):
                        with open(fpath, 'rb') as ffile:
                            data_grid[i,j] = pickle.load(ffile)
                    else:
                        data_grid[i,j] = compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = method, edgepar = edgepar, normalize = normalize)
                        # Save to disk if save_dir is provided
                        with open(fpath, 'wb') as ffile:
                            pickle.dump(data_grid[i,j], ffile)

    else:

        for i, delta in enumerate(tqdm(delta_list, desc="Deltas")):
            for j, T in enumerate(tqdm(T_list, desc=f"T for delta={delta}", leave=False)):
                
                data_grid[i,j] = compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = method, edgepar = edgepar, normalize = normalize)


    freqs = frequencies(N_cycles+1)

    return freqs, np.arange(N_cycles+1), data_grid
