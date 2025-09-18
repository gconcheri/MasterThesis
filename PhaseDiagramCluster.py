import numpy as np

import Class_site as site
import free_fermion_representation as f
import pickle
import os

from numba import njit

#%% order_parameter_delta_T functions

#function used to compute order parameter for number of floquet cycles = N_cycles 
# at a specific disorder delta and coupling/time term T


def order_parameter_delta_T(model, fgs, T, delta, N_cycles, edgepar = None, loop_type = 'general', loop_list = None):
    orderpar = []
    loop_0 = []
    loop_e = []

    fgs.reset_cov_0_matrix()
    fgs.reset_cov_e_matrix()

    _, _, _, R0 = fgs.floquet_operator_ham(T)
    _, _, _, Re = fgs.floquet_operator_ham(T, anyon=True)

    if delta != 0:
        V0 = f.generate_disorder_term(model, fgs.Cov, delta, edgepar = edgepar)
        Ve = f.generate_disorder_term(model, fgs.Cov, delta, type="Anyon", edgepar = edgepar)
        R_V0 = f.floquet_operator(V0, T, alpha = np.pi/4.)
        R_Ve = f.floquet_operator(Ve, T, alpha = np.pi/4.)
        R_V0_R0 = R_V0 @ R0
        R_Ve_Re = R_Ve @ Re
    else:
        R_V0_R0 = R0
        R_Ve_Re = Re

    for _ in range(N_cycles):
        op, value_0, value_e = fgs.order_parameter(type=loop_type, plaquette_list=loop_list)
        orderpar.append(op)
        loop_0.append(value_0)
        loop_e.append(value_e)
        fgs.update_cov_0_matrix(R_V0_R0)
        fgs.update_cov_e_matrix(R_Ve_Re)
        # this approach requires 3*N^3 operations
        #vs doing 2 consecutive updates update(R0) -> update(R_V0) = 4*N^3 operations!
        #so the way we are doing it is computationally favorable

    op, value_0, value_e = fgs.order_parameter(type=loop_type, plaquette_list=loop_list)
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

def compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, save_dir, loop_type = 'general', loop_list = None):
    """ Computes a single entry in the data grid for given T and delta, averaging over N_shots if necessary. """
    print("computing entry for: ", f"T = {T}, delta = {delta}, for {N_shots} shots")

    os.makedirs(save_dir, exist_ok=True)
    fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
    fpath = os.path.join(save_dir, fname)

    if os.path.exists(fpath) == False:
        if delta == 0 or N_shots < 2: 
            #no need to average when delta = 0, or N_shots = 1 because no random disorder is added in this case!
            #N_shots = 0 equivalent to N_shots = 1
            #if N_shots = 0 this is just the old algorithm

            orderpar, loop_0, loop_e = order_parameter_delta_T(model, fgs, T, delta,  N_cycles, loop_type=loop_type, loop_list=loop_list)

            ft, _, _ = fourier_time(np.array(orderpar), 1)

            data_grid_entry = {
                'loop_0' : loop_0, #expectation value of loop with no anyon
                'loop_e' : loop_e, #expectation value of loop with e anyon
                'op_real': orderpar,
                'op_ft' : ft
            }

        else:

            loop_0_list = []
            loop_e_list = []
            orderpar_list = []
            op_ft_list = []


            for i in range(N_shots):
                print(f"  shot {i+1} of {N_shots}")
                orderpar, loop_0, loop_e = order_parameter_delta_T(model, fgs, T, delta,  N_cycles, loop_type=loop_type, loop_list=loop_list)

                ft, _, _ = fourier_time(np.array(orderpar), 1)

                loop_0_list.append(loop_0)
                loop_e_list.append(loop_e)
                orderpar_list.append(orderpar)
                op_ft_list.append(ft)


            data_grid_entry = {
                'loop_0' : loop_0_list, #expectation value of loop with no anyon
                'loop_e' : loop_e_list, #expectation value of loop with e anyon
                'op_real': orderpar_list,
                'op_ft': op_ft_list
            }

        # Save to disk if save_dir is provided
        with open(fpath, 'wb') as ffile:
            pickle.dump(data_grid_entry, ffile)
    

def simulation(**kwargs):
    """ Main function to run the simulation and compute the phase diagram data grid. """

    # Extract parameters from kwargs
    T = kwargs.get('T')
    delta = kwargs.get('delta')
    N_cycles = kwargs.get('N_cycles')
    N_shots = kwargs.get('N_shots')
    system_size = kwargs.get('system_size')
    edge = kwargs.get('edge', False)
    save_dir = kwargs.get('save_dir', None)
    loop_type = kwargs.get('loop_type', 'general')
    loop_list = kwargs.get('loop_list', None)

    # Check required parameters
    if None in (T, delta, N_cycles, N_shots, system_size):
        raise ValueError("Missing required simulation parameters.")

    if save_dir is None:
        save_dir = "pd" + f"_size{system_size}" + f"_Nshots{N_shots}" + f"_cycles{N_cycles}" + ("_edge" if edge else "_noedge") + f"_{loop_type}_loop" + f"_{loop_list}"

    model = site.SitesOBC(Npx = system_size, Npy = system_size, edge = edge)
    fgs = f.FermionicGaussianRepresentation(model)

    # Loop over all combinations of T and delta to compute data grid entries
    compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, save_dir, loop_type=loop_type, loop_list=loop_list)