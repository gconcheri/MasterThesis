import numpy as np
import matplotlib.pyplot as plt

import Class_site as site
import free_fermion_representation as f
import pickle
import os

from tqdm import tqdm 
from itertools import product

from numba import njit
from joblib import Parallel, delayed

#%% order_parameter_delta_T functions

#function used to compute order parameter for number of floquet cycles = N_cycles 
# at a specific disorder delta and coupling/time term T

def order_parameter_delta_T_method1(model, fgs, delta, T, N_cycles):
    result = []

    return result

def order_parameter_delta_T_method2(model, fgs, delta, T, N_cycles):

    fgs.reset_cov_0_matrix()
    fgs.reset_cov_e_matrix()

    if delta != 0:
        V0 = f.generate_disorder_term(model, fgs.Cov, delta)
        Ve = f.generate_disorder_term(model, fgs.Cov, delta, type = "Anyon")
    else:
        V0 = 0
        Ve = 0
    #to spare running time!

    Rx = f.floquet_operator(-np.pi/4*fgs.h0_x + V0, T, alpha = 1)
    Ry = f.floquet_operator(-np.pi/4*fgs.h0_y + V0, T, alpha = 1)
    Rz = f.floquet_operator(-np.pi/4*fgs.h0_z + V0, T, alpha = 1)
    R0 = Rz @ Ry @ Rx

    Rx_e = f.floquet_operator(-np.pi/4*fgs.he_x+ Ve, T, alpha = 1)
    Ry_e = f.floquet_operator(-np.pi/4*fgs.he_y + Ve, T, alpha = 1)
    Rz_e = f.floquet_operator(-np.pi/4*fgs.he_z + Ve, T, alpha = 1)
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

def fourier_time(t_series, dt, sigma = 0.4, nw=4, gauss = True):
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

    ft_0 = ft[0]

    if n % 2 == 1:
        ft_pi = (ft[(n-1)//2]+ft[(n+1)//2])/2 # // even though (n-1)/2 is always integer it will yield n.0, I use // to force it to be integer
    else:
        ft_pi = ft[n//2]

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


#%% "slow" algorithm to create 2d phase diagram

#algorithm that calculates 1 function (o.p.) per single T and delta
#can be cancelled probably

def order_parameter(model, T_list, delta_list, N_cycles, method = '1'):
    fgs = f.FermionicGaussianRepresentation(model)
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)

    for i, delta in enumerate(delta_list):
        for j, T in enumerate(T_list):
            print("delta: ", delta, ";   T: ", T)
            if method == '1':
                orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, delta, T, N_cycles)
            else: 
                orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, delta, T, N_cycles)

            ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)

            result = np.abs(ft_pi) - np.abs(ft_0)

            data_grid[i, j] = {
                'loop_0' : loop_0, #expectation value of loop with no anyon
                'loop_e' : loop_e, #expectation value of loop with e anyon
                'op_real': orderpar,
                'op_ft' : ft, 
                'result': result
            }

    freqs = frequencies(N_cycles+1)

    return freqs, np.arange(N_cycles+1), data_grid



#algorithm that calculates N_shots functions (o.p.) per combination (T,delta), 
# and then averages eta(pi) - eta(o) over these N_shots

def order_parameter_shots(model, T_list, delta_list, N_shots = 10, N_cycles = 10, method = '1'):
    fgs = f.FermionicGaussianRepresentation(model)
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)

    for i, delta in enumerate(tqdm(delta_list, desc="Deltas")):
        for j, T in enumerate(tqdm(T_list, desc=f"T for delta={delta}", leave=False)):
            # print("delta: ", delta, ";   T: ", T)
            if delta == 0 or N_shots < 2: 
                #no need to average when delta = 0 because no random disorder is added in this case!
                #if N_shots = 0 this is just the old algorithm

                if method == '1':
                    orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, delta, T, N_cycles)
                else: 
                    orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, delta, T, N_cycles)

                ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)

                result = np.abs(ft_pi) - np.abs(ft_0)
            
                data_grid[i, j] = {
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

                for _ in range(N_shots):
                    print("delta: ", delta, ";   T: ", T)
                    if method == '1':
                        orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, delta, T, N_cycles)
                    else: 
                        orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, delta, T, N_cycles)

                    ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)

                    result_list.append(np.abs(ft_pi) - np.abs(ft_0))
                    loop_0_list.append(loop_0)
                    loop_e_list.append(loop_e)
                    orderpar_list.append(orderpar)
                    op_ft_list.append(ft)
            
                result = np.mean(result_list)
                
                data_grid[i, j] = {
                    'loop_0' : loop_0_list, #expectation value of loop with no anyon
                    'loop_e' : loop_e_list, #expectation value of loop with e anyon
                    'op_real': orderpar_list,
                    'op_ft' : op_ft_list, 
                    'result': result
                }



    freqs = frequencies(N_cycles+1)

    return freqs, np.arange(N_cycles+1), data_grid


#%% FAST algorithm!

def compute_single_shot(model, T, delta, N_cycles, method = '1'):
    # Recreate fgs inside the function (safer for multiprocessing)
    fgs = f.FermionicGaussianRepresentation(model)

    if method == '1':
        orderpar, loop_0, loop_e= order_parameter_delta_T_method1(model, fgs, delta, T, N_cycles)
    else:
        orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, delta, T, N_cycles)

    ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)
    result = np.abs(ft_pi) - np.abs(ft_0)

    return {
        'loop_0' : loop_0, #expectation value of loop with no anyon
        'loop_e' : loop_e, #expectation value of loop with e anyon        
        'op_real': orderpar,
        'op_ft': ft,
        'result': result
    }

def compute_order_param_entry(model, T, delta, N_cycles, N_shots, method='1', save_dir=None):
    if delta == 0 or N_shots < 2:
            #no need to average when delta = 0 because no random disorder is added in this case!
            #if N_shots = 0 this is just the old algorithm
        
        entry = compute_single_shot(model, T, delta, N_cycles, method)

    
    else:
        shot_results = Parallel(n_jobs=-1)(
            delayed(compute_single_shot)(model, T, delta, N_cycles, method)
            for _ in range(N_shots)
        )

        loop_0_list = [r['loop_0'] for r in shot_results]
        loop_e_list = [r['loop_e'] for r in shot_results]
        orderpar_list = [r['op_real'] for r in shot_results]
        op_ft_list = [r['op_ft'] for r in shot_results]
        result_list = [r['result'] for r in shot_results]

        avg_result = np.mean(result_list)

        entry = {
            'loop_0': loop_0_list,
            'loop_e': loop_e_list,
            'op_real': orderpar_list,
            'op_ft': op_ft_list,
            'result': avg_result
        }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
        fpath = os.path.join(save_dir, fname)
        with open(fpath, 'wb') as f:
            pickle.dump(entry, f)

    return delta, T, entry


def order_parameter_parallel(model, T_list, delta_list, N_cycles, N_shots=1, n_jobs=-1, method='1', save_dir=None):
    tasks = []
    desc = f"Computing {(len(delta_list) * len(T_list))} (delta, T) points"
    
    for delta, T in tqdm(product(delta_list, T_list), desc=desc):
        fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
        fpath = os.path.join(save_dir, fname) if save_dir else None

        # Check if already computed
        if save_dir is not None and os.path.exists(fpath):
            # Load immediately and skip compute
            with open(fpath, 'rb') as f:
                entry = pickle.load(f)
            tasks.append((delta, T, entry))
        else:
            # Mark for computation
            tasks.append((delta, T, None))

    # Split tasks
    to_compute = [t for t in tasks if t[2] is None]
    already_done = [t for t in tasks if t[2] is not None]

    # Compute missing entries in parallel
    computed_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_order_param_entry)(model, T, delta, N_cycles, N_shots, method, save_dir)
        for (delta, T, _) in tqdm(to_compute, desc="Computing missing entries")
    )

    # Merge results
    all_results = already_done + computed_results

    # Rebuild grid
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)
    delta_idx = {d: i for i, d in enumerate(delta_list)}
    T_idx = {t: j for j, t in enumerate(T_list)}

    for delta, T, entry in all_results:
        i = delta_idx[delta]
        j = T_idx[T]
        data_grid[i, j] = entry

    freqs = frequencies(N_cycles + 1)
    return freqs, np.arange(N_cycles + 1), data_grid


#%% Alternative fast algorithm
#OTHER WAY OF DOING ALGORITHM, while keeping track of computed and non computed results!
# (I wanted to keep it to learn another way of doing it)

def compute_order_param_entry_otherversion(delta, T, model, N_cycles, N_shots, method='1', save_dir=None):

    # Optional: skip if already computed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                entry = pickle.load(f)
            return delta, T, entry

    # Compute N_shots
    shot_results = Parallel(n_jobs=-1)(
        delayed(compute_single_shot)(model, delta, T, N_cycles, method)
        for _ in range(N_shots)
    )

    #Aggregate results
    loop_0_list = [r['loop_0'] for r in shot_results]
    loop_e_list = [r['loop_e'] for r in shot_results]
    orderpar_list = [r['op_real'] for r in shot_results]
    op_ft_list = [r['op_ft'] for r in shot_results]
    result_list = [r['result'] for r in shot_results]

    avg_result = np.mean(result_list)

    entry = {
        'loop_0': loop_0_list,
        'loop_e': loop_e_list,
        'op_real': orderpar_list,
        'op_ft': op_ft_list,
        'result': avg_result
    }

    # Save to disk if save_dir is provided
    if save_dir is not None:
        with open(fpath, 'wb') as f:
            pickle.dump(entry, f)

    return delta, T, entry

#n_jobs = -1 uses all CPU cores! if I want to limit it I can put e.g. n_jobs = 4
def order_parameter_parallel_otherversion(model, T_list, delta_list, N_cycles, N_shots=1, n_jobs=-1, method='1', save_dir = None):
    
    to_compute = [(delta, T) for delta in delta_list for T in T_list]

    
    # Parallelize across (delta, T) pairs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_order_param_entry_otherversion)(delta, T, model, N_cycles, N_shots, method, save_dir)
        for delta, T in tqdm(to_compute, desc=r"Computing ($\Delta$, T) pairs")
    )

    # Fill grid
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)
    delta_idx = {d: i for i, d in enumerate(delta_list)}
    T_idx = {t: j for j, t in enumerate(T_list)}

    #e.g.
    # delta_list = [0.0, 0.1, 0.2]
    # T_list = [0.1, 0.2, 0.3]

    # delta_idx = {0.0: 0, 0.1: 1, 0.2: 2}
    # T_idx     = {0.1: 0, 0.2: 1, 0.3: 2}


    for delta, T, entry in results:
        i = delta_idx[delta]
        j = T_idx[T]
        data_grid[i, j] = entry

    freqs = frequencies(N_cycles + 1)

    return freqs, np.arange(N_cycles + 1), data_grid


#%% #PLOTS:
#function to plot the 2d phase diagram!
def plot_phase_diagram(data_grid, T_list, delta_list, figsize = (8,6)):
    # Extract the 2D array of 'result'
    Z = np.array([
        [data_grid[i, j]['result'] for j in range(len(T_list))]
        for i in range(len(delta_list))
    ])

    deltaT = (T_list[1]-T_list[0])/2
    deltadelta = (delta_list[1]-delta_list[0])/2
    
    plt.figure(figsize=figsize)
    im = plt.imshow(Z, aspect='auto', origin='lower', 
                    extent=[T_list[0]-deltaT, T_list[-1]+deltaT, delta_list[0]-deltadelta, delta_list[-1]+deltadelta],
                    interpolation='none', 
                    cmap = 'inferno')
    
    plt.colorbar(im, label=r'$|\eta(π)| - |\eta(0)|$')
    plt.xlabel('T')
    plt.ylabel(r'$\Delta$')
    plt.title('Phase Diagram')

    plt.tight_layout()
    plt.show()

# def plot(delta_target, T_target):

#%% #functions to save or load arrays!



def save_file(obj, name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(obj, file)

def load_file(name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj
    


def load_saved_results(delta_list, T_list, save_dir):
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)
    delta_idx = {d: i for i, d in enumerate(delta_list)}
    T_idx = {t: j for j, t in enumerate(T_list)}

    for delta in delta_list:
        for T in T_list:
            fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
            fpath = os.path.join(save_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    entry = pickle.load(f)
                i = delta_idx[delta]
                j = T_idx[T]
                data_grid[i, j] = entry

    return data_grid




#%% profile output of the "order_parameter_delta_T"
import cProfile 
import pstats

#defined function to create the profile output of the "order_parameter_delta_T" function 
# in order to figure out what computations take the most time!
def main():
    model = site.SitesOBC(Npx = 31, Npy = 31, edge = True)
    fgs = f.FermionicGaussianRepresentation(model)
    delta = 0
    T = 0.7
    N_cycles = 10
    result = order_parameter_delta_T_method2(model, fgs, delta=delta, T=T, N_cycles=N_cycles)

if __name__ == "__main__":
    with open("profile_output.txt", "w") as logfile:
        profiler = cProfile.Profile()
        profiler.enable()

        main()  # run your code

        profiler.disable()
        stats = pstats.Stats(profiler, stream=logfile)
        stats.strip_dirs().sort_stats("cumtime").print_stats(50)  # print top 50 entries

# otherwise use:
# python -m cProfile -s cumtime your_script.py 

