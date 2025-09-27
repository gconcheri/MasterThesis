import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import copy


import Class_site as site
import free_fermion_representation as f
import pickle
import os

# from tqdm import tqdm 
from tqdm.auto import tqdm

from itertools import product

from numba import njit
from joblib import Parallel, delayed


#%% order_parameter_delta_T functions

#function used to compute order parameter for number of floquet cycles = N_cycles 
# at a specific disorder delta and coupling/time term T

#good idea would be to unify method1 and method2

def order_parameter_delta_T_method1(model, fgs, T, delta, N_cycles, edgepar = None, loop_type = 'general', loop_list = None):
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


def order_parameter_delta_T_method2(model, fgs, T, delta, N_cycles, edgepar = None, loop_type = 'general', loop_list = None):
    """ Alternative method to compute the order parameter over N_cycles, applying the full Floquet operator each cycle. """

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
        op, value_0, value_e = fgs.order_parameter(type=loop_type, plaquette_list=loop_list)
        orderpar.append(op)
        loop_0.append(value_0)
        loop_e.append(value_e)
        fgs.update_cov_0_matrix(R0)
        fgs.update_cov_e_matrix(Re)

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

#%% "slow" algorithm to create 2d phase diagram

#algorithm that computes data grid entry for N_shots or not depending on value of (delta,T)
#i.e. calculates N_shots functions (o.p.) per combination (T,delta), 
# and then averages eta(pi) - eta(o) over these N_shots

def compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = '1', edgepar = None, normalize = False, loop_type = 'general', loop_list = None):
    """ Computes a single entry in the data grid for given T and delta, averaging over N_shots if necessary. """

    if delta == 0 or N_shots < 2: 
        #no need to average when delta = 0, or N_shots = 1 because no random disorder is added in this case!
        #N_shots = 0 equivalent to N_shots = 1
        #if N_shots = 0 this is just the old algorithm

        if method == '1':
            orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)
        else: 
            orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)

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
                orderpar, loop_0, loop_e = order_parameter_delta_T_method1(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)
            else: 
                orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)

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

def phase_diagram_slow(model, T_list, delta_list, N_shots = 10, N_cycles = 10, method = '1', save_dir = None, general_dir = "phasediagram", edgepar = None, normalize = False, loop_type = 'general', loop_list = None):
    """ Computes the phase diagram over a grid of T and delta values, optionally saving results to disk. """

    fgs = f.FermionicGaussianRepresentation(model)
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)

    if save_dir is not None:

        full_dir = os.path.join(general_dir, save_dir, 'data')
        os.makedirs(full_dir, exist_ok=True)

        for i, delta in enumerate(tqdm(delta_list, desc="Deltas")):
            for j, T in enumerate(tqdm(T_list, desc=f"T for delta={delta}", leave=False)):
        
                    fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
                    fpath = os.path.join(full_dir, fname)

                    if os.path.exists(fpath):
                        with open(fpath, 'rb') as ffile:
                            data_grid[i,j] = pickle.load(ffile)
                    else:
                        data_grid[i,j] = compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = method, edgepar = edgepar, normalize = normalize, loop_type = loop_type, loop_list = loop_list)
                        # Save to disk if save_dir is provided
                        with open(fpath, 'wb') as ffile:
                            pickle.dump(data_grid[i,j], ffile)

    else:

        for i, delta in enumerate(tqdm(delta_list, desc="Deltas")):
            for j, T in enumerate(tqdm(T_list, desc=f"T for delta={delta}", leave=False)):
                
                data_grid[i,j] = compute_data_grid_entry(model, T, delta, fgs, N_shots, N_cycles, method = method, edgepar = edgepar, normalize = normalize, loop_type = loop_type, loop_list = loop_list)


    freqs = frequencies(N_cycles+1)

    return freqs, np.arange(N_cycles+1), data_grid


#%% FAST algorithm!

def compute_single_shot(model, T, delta, N_cycles, method = '1', edgepar = None, normalize = False, loop_type = 'general', loop_list = None):
    """
    Computes a single shot of the phase diagram for given T and delta.
    """
    # Recreate fgs inside the function (safer for multiprocessing)
    fgs = f.FermionicGaussianRepresentation(model)

    if method == '1':
        orderpar, loop_0, loop_e= order_parameter_delta_T_method1(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)
    else:
        orderpar, loop_0, loop_e = order_parameter_delta_T_method2(model, fgs, T, delta,  N_cycles, edgepar = edgepar, loop_type = loop_type, loop_list = loop_list)

    ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1, normalize = normalize)
    result = np.abs(ft_pi) - np.abs(ft_0)

    return {
        'loop_0' : loop_0, #expectation value of loop with no anyon
        'loop_e' : loop_e, #expectation value of loop with e anyon        
        'op_real': orderpar,
        'op_ft': ft,
        'result': result
    }

def compute_order_param_entry(model, T, delta, N_cycles, N_shots, method='1', save_dir=None, general_dir = "phasediagram", edgepar = None, normalize = False, loop_type = 'general', loop_list = None):
    """ Computes a single entry in the data grid for given T and delta, averaging over N_shots if necessary. """
    if delta == 0 or N_shots < 2:
            #no need to average when delta = 0 because no random disorder is added in this case!
            #if N_shots = 0 this is just the old algorithm

        entry = compute_single_shot(model, T, delta, N_cycles, method, edgepar = edgepar, normalize=normalize, loop_type = loop_type, loop_list = loop_list)

    else:
        shot_results = Parallel(n_jobs=-1)(
            delayed(compute_single_shot)(model, T, delta, N_cycles, method, edgepar = edgepar, normalize=normalize, loop_type = loop_type, loop_list = loop_list)
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
        full_dir = os.path.join(general_dir, save_dir)
        os.makedirs(full_dir, exist_ok=True)
        fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
        fpath = os.path.join(full_dir, fname)
        with open(fpath, 'wb') as ffile:
            pickle.dump(entry, ffile)

    return delta, T, entry

def phase_diagram_fast(model, T_list, delta_list, N_cycles, N_shots=1, n_jobs=-1, method='1', save_dir=None, general_dir = "phasediagram", edgepar = None, normalize = False, loop_type = 'general', loop_list = None):
    """ Computes the phase diagram over a grid of T and delta values, optionally saving results to disk. """
    tasks = []
    desc = f"Computing {(len(delta_list) * len(T_list))} (delta, T) points"
    
    for delta, T in tqdm(product(delta_list, T_list), desc=desc):
        fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"
        fpath = os.path.join(general_dir, os.path.join(save_dir, 'data', fname)) if save_dir else None

        # Check if already computed
        if save_dir is not None and os.path.exists(fpath):
            # Load immediately and skip compute
            with open(fpath, 'rb') as ffile:
                entry = pickle.load(ffile)
            tasks.append((delta, T, entry))
        else:
            # Mark for computation
            tasks.append((delta, T, None))

    # Split tasks
    to_compute = [t for t in tasks if t[2] is None]
    already_done = [t for t in tasks if t[2] is not None]

    # Compute missing entries in parallel
    computed_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_order_param_entry)(model, T, delta, N_cycles, N_shots, method, save_dir, general_dir, edgepar, normalize, loop_type, loop_list)
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

def compute_order_param_entry_otherversion(T, delta,  model, N_cycles, N_shots, method='1', save_dir=None):

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
        delayed(compute_single_shot)(model, T, delta,  N_cycles, method)
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
def phase_diagram_fast_otherversion(model, T_list, delta_list, N_cycles, N_shots=1, n_jobs=-1, method='1', save_dir = None):
    
    to_compute = [(delta, T) for delta in delta_list for T in T_list]

    
    # Parallelize across (delta, T) pairs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_order_param_entry_otherversion)(T, delta,  model, N_cycles, N_shots, method, save_dir)
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

#%% Manipulations on data grid

def remove_shots_fromdatagrid(data_grid, T_list, delta_list, result="difference", threshold=100):
    """
    Removes shots from all relevant data grid entry lists if the value for that shot exceeds the threshold.
    Works for both 'difference' and 'ratio' result types.
    """
    data_grid = copy.deepcopy(data_grid) # Avoid modifying the original data_grid

    for i in range(len(delta_list)):
        for j in range(len(T_list)):
            entry = data_grid[i, j]
            if entry is not None:
                op_ft = entry['op_ft']
                # Only process if multi-shot (list)
                if isinstance(op_ft, list):
                    keep_indices = []
                    for idx, ft in enumerate(op_ft):
                        ft_0, ft_pi = ft_0_and_ft_pi(ft)
                        if result == "difference":
                            value = np.abs(ft_pi) - np.abs(ft_0)
                        elif result == "ratio":
                            value = np.abs(ft_pi) / np.abs(ft_0) if np.abs(ft_0) != 0 else np.inf
                        if np.abs(value) <= threshold:
                            keep_indices.append(idx)
                    # Filter all relevant lists
                    entry['op_ft'] = [entry['op_ft'][k] for k in keep_indices]
                    if isinstance(entry['op_real'], list):
                        entry['op_real'] = [entry['op_real'][k] for k in keep_indices]
                    if isinstance(entry['loop_e'], list):
                        entry['loop_e'] = [entry['loop_e'][k] for k in keep_indices]
                    if isinstance(entry['loop_0'], list):
                        entry['loop_0'] = [entry['loop_0'][k] for k in keep_indices]
    return data_grid

def count_remaining_shots(data_grid, T_list, delta_list):
    """
    Returns a grid with the number of remaining shots in entry['op_ft'] for each entry in data_grid.
    """
    shot_count_grid = np.zeros((len(delta_list), len(T_list)), dtype=int)
    for i in range(len(delta_list)):
        for j in range(len(T_list)):
            entry = data_grid[i, j]
            if entry is not None and isinstance(entry['op_ft'], list):
                shot_count_grid[i, j] = len(entry['op_ft'])
            else:
                shot_count_grid[i, j] = 1  # single shot or None
    return shot_count_grid

def get_regularized_data_grid(data_grid, T_list, delta_list, regularization = 1e-15):
    #does not change entry['result'], only entry['op_real'], entry['op_ft'], entry['loop_0']

    data_grid = copy.deepcopy(data_grid) # Avoid modifying the original data_grid

    for i in range(len(delta_list)):
        for j in range(len(T_list)):
            entry = data_grid[i, j]
            if entry is not None:
                arr = np.array(entry['loop_0'])
                arr[arr < regularization] = regularization
                entry['loop_0'] = arr.tolist()
                entry['op_real'] = (np.array(entry['loop_e'])/arr).tolist()

                if isinstance(entry['op_real'][0], list):
                    filtered_op_ft = []
                    for op in entry['op_real']:
                        filtered_op_ft.append(fourier_time(np.array(op), 1)[0])
                    entry['op_ft'] = filtered_op_ft
                else:
                    entry['op_ft'] = fourier_time(np.array(entry['op_real']), 1)[0]

            else:
                raise ValueError(f"Data grid entry at delta index {i}, T index {j} is None.")    
    return data_grid


def get_difference(op_ft):
    if isinstance (op_ft, list):
        differences = []
        for ft in op_ft:
            ft_0, ft_pi = ft_0_and_ft_pi(ft)
            differences.append(np.abs(ft_pi) - np.abs(ft_0))
        return np.mean(differences)
    else:
        ft_0, ft_pi = ft_0_and_ft_pi(op_ft)
        return np.abs(ft_pi) - np.abs(ft_0)

def get_ratio(op_ft, bool_log = False):
    """ Computes the ratio |eta(pi)|/|eta(0)| from the Fourier transform data."""
    # If op_ft is a list (multi-shot), average the ratio over all shots
    if isinstance(op_ft, list):
        ratios = []
        for ft in op_ft:
            ft_0, ft_pi = ft_0_and_ft_pi(ft)
            ratios.append(np.abs(ft_pi) / np.abs(ft_0))
        if bool_log:
            return np.mean(np.log(ratios))
        else:
            return np.mean(ratios)
    else:
        ft_0, ft_pi = ft_0_and_ft_pi(op_ft)
        if bool_log:
            return np.log(np.abs(ft_pi) / np.abs(ft_0))
        else:
            return np.abs(ft_pi) / np.abs(ft_0)

def get_result(op_ft, result = "difference", bool_log = False):
    if result == "difference":
        return get_difference(op_ft)
    elif result == "ratio":
        return get_ratio(op_ft, bool_log)
    else:
        raise ValueError("Invalid result type. Choose 'difference' or 'ratio'.")


def get_data_grid_results(data_grid, T_list, delta_list, result="difference", bool_log = False, regularization = None, threshold=None, count_shots = False, flip_T_axis = False):
    #data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid

    if regularization is not None:
        data_grid = get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)

    if threshold is not None:
        data_grid = remove_shots_fromdatagrid(data_grid, T_list, delta_list, result=result, threshold=threshold)

    new_data_grid = np.empty((len(delta_list), len(T_list)))
    for i in range(len(delta_list)):
        for j in range(len(T_list)):
            entry = data_grid[i, j]
            op_ft = entry['op_ft']

            if result == "difference":
                new_data_grid[i, j] = get_difference(op_ft)
            elif result == "ratio":
                new_data_grid[i, j] = get_ratio(op_ft, bool_log = bool_log)

    if flip_T_axis:
        new_data_grid = new_data_grid[:, ::-1]

    if count_shots:
        data_grid_shot_counts = count_remaining_shots(data_grid, T_list, delta_list)
        if flip_T_axis:
            data_grid_shot_counts = data_grid_shot_counts[:,::-1]

        return new_data_grid, data_grid_shot_counts
    else:
        return new_data_grid

#%% #PLOTS:
#function to plot the 2d phase diagram!
def plot_phase_diagram_fromdatagrid(data_grid, T_list, delta_list, figsize = None, result = "difference", bool_log = False, regularization = None, 
                                    threshold = None, vmax = None, vmin = None, save = False, save_dir_image = None, filename = None, x_digits = None, y_digits = None, flip_T_axis = False):

    if figsize == None:
        figsize = (len(T_list), len(delta_list))

    Z = get_data_grid_results(data_grid, T_list, delta_list, result=result, bool_log = bool_log, regularization = regularization, threshold=threshold, flip_T_axis=flip_T_axis)

    deltaT = np.abs(T_list[1]-T_list[0])/2
    deltadelta = (delta_list[1]-delta_list[0])/2

    if flip_T_axis:
        T_start = T_list[-1]-deltaT
        T_finish = T_list[0]+deltaT
    else:
        T_start = T_list[0]+deltaT
        T_finish = T_list[-1]-deltaT       
    
    
    plt.figure(figsize=figsize)
    im = plt.imshow(Z, aspect='auto', origin='lower', 
                    extent=[T_start, T_finish, delta_list[0]-deltadelta, delta_list[-1]+deltadelta],
                    interpolation='none', 
                    vmin = vmin,
                    vmax = vmax
                    #cmap = 'inferno'
                    )
    
    if flip_T_axis:
        plt.xticks(np.array(T_list[::-1]))
    else:
        plt.xticks(np.array(T_list))
    plt.yticks(np.array(delta_list))

    if x_digits is not None:
        str_x = '%.' + f'{x_digits}' + 'f'
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter(str_x))
    if y_digits is not None:
        str_y = '%.' + f'{y_digits}' + 'f'
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(str_y))

    if result == "difference":
        plt.colorbar(im, label=r'$|\eta(π)| - |\eta(0)|$')
    elif result == "ratio":
        if bool_log:
            plt.colorbar(im, label=r'$\log(\frac{|\eta(π)|}{|\eta(0)|})$')
        else:
            plt.colorbar(im, label=r'$\frac{|\eta(π)|}{|\eta(0)|}$')

    plt.xlabel('T')
    plt.ylabel(r'$\Delta$')
    plt.title('Phase Diagram')

    plt.tight_layout()

    if save:
        if save_dir_image is None:
            save_dir_image = "figures_phasediagram"
        os.makedirs(save_dir_image, exist_ok=True)
        if filename is None:
            filename = f"phase_diagram_{result}.png"   
        plt.savefig(os.path.join(save_dir_image, filename))
    
    plt.show()

def plot_phase_diagram(T_list, delta_list, save_dir = None, general_dir = "phasediagram", figsize = None, result = "difference", bool_log = False, 
                       regularization = None, threshold = None, vmin = None, vmax = None, save = False, save_dir_image = None, filename = None, suffix = ".svg",
                       x_digits = None, y_digits = None, flip_T_axis = False):
    
    data_grid = load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)

    if save:
        if filename is None:
            filename = f"result_{result}"
            if bool_log:
                filename = filename + "_log"
            if vmin is not None and vmax is not None:
                filename = filename + f"_vmin{vmin}_vmax{vmax}"
            if threshold is not None:
                filename = filename + f"_thresh{threshold}"
            if regularization is not None:
                filename = filename + f"_reg{regularization}"
            if flip_T_axis:
                filename = filename + "flipped_T"
            filename = filename + suffix
    
        if save_dir_image is None:
            save_dir_image = "figures/phasediagram"
            save_dir_image = os.path.join(general_dir, save_dir, save_dir_image)
            os.makedirs(save_dir_image, exist_ok=True)

    plot_phase_diagram_fromdatagrid(data_grid[:len(delta_list), :len(T_list)], T_list, delta_list, figsize = figsize, result = result, bool_log = bool_log,
                                    regularization = regularization, threshold = threshold, vmin = vmin, vmax = vmax,
                                    save = save, save_dir_image = save_dir_image, filename = filename, x_digits=x_digits, y_digits = y_digits, flip_T_axis=flip_T_axis)

def plot_line_phase_diagram_fromdatagrids(data_grid_list, delta, T_list, delta_list, loop_size_list, result = "difference", bool_log = False, regularization = None, 
                                    threshold = None, save = False, save_dir_image = None, filename = None, return_results = True):


    delta_idx = delta_list.index(delta)   
    results = {}

    
    for idx, data_grid in enumerate(data_grid_list):
        data_grid = get_data_grid_results(data_grid, T_list, delta_list, result=result, bool_log = bool_log, regularization = regularization, threshold=threshold)
        data = data_grid[delta_idx,:]
        plt.plot(T_list, data, label = f'loop size: {loop_size_list[idx]}')
        
        if return_results:
            results[idx] = data


    plt.xlabel('T')
    if result == "difference":
        plt.ylabel(r'$|\eta(π)| - |\eta(0)|$')
    elif result == "ratio":
        if bool_log:
            plt.ylabel(r'$\log(\frac{|\eta(π)|}{|\eta(0)|})$')
        else:
            plt.ylabel(r'$\frac{|\eta(π)|}{|\eta(0)|}$') 

    plt.legend()

    plt.title('phase diagram along ' + r'$\Delta$ = ' + f'{delta}') 

    plt.tight_layout()

    if save:
        if save_dir_image is None:
            save_dir_image = "figures_phasediagram"
        os.makedirs(save_dir_image, exist_ok=True)
        if filename is None:
            filename = f"finite_size_scaling_{result}"
            filename += f"_{delta}"
            filename += ".png"   
        plt.savefig(os.path.join(save_dir_image, filename))
    
    plt.show()

    if return_results:
        return results

def plot_single_entry_fromdatagrid(
    data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles, name="op_real",
    figsize=(12, 5), shot_idx=None, regularization = None, threshold=None, log_list=None, layout="row", color_list=None,
    save=False, save_dir_image=None, filename=None
): #color_list could be plt.cm.tab10.colors or similar
    """
    Plots a single entry from the data grid for given delta and T indices.
    If name is a list, creates subplots for each specified quantity.
    If log_list is a list of booleans, it specifies whether to plot the log for each corresponding quantity in name.
    layout: "row", "col", or (nrows, ncols)
    If log_bool = True for the loop_e, then we place absolute value before taking log to avoid issues with negative values.
    """
    #data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid
    #I don't need it because I put this line inside the get_regularized_data_grid and remove_shots_fromdatagrid functions

    if log_list is not None:
        if not isinstance(log_list, list) or len(log_list) != len(name):
            raise ValueError("log_list must be a list of booleans with the same length as name.")

    if regularization is not None:
        data_grid = get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)

    if threshold is not None:
        data_grid = remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)
    
    entry = data_grid[delta_idx, T_idx]
    floquet_cycles = np.arange(N_cycles + 1)
    freqs = frequencies(N_cycles + 1)

    if entry is None:
        print(f"No data available for Δ = {delta_list[delta_idx]}, T = {T_list[T_idx]}")
        return

    def plot_shots(data, x, title, ylabel, axx=None, log_bool=False, color_list=None):
        if axx is None:
            raise ValueError("axx must be provided for plotting shots.")
        if shot_idx is not None:
            if 0 <= shot_idx < len(data):
                axx.plot(x, data[shot_idx])
                axx.legend()
                axx.set_xlabel('Floquet Cycles' if title != 'Fourier Transform' else 'Frequency')
                axx.set_ylabel(ylabel)
                if log_bool:
                    axx.set_ylabel("Log " + ylabel)
                    axx.set_yscale("log")
                axx.set_title(title + f' (Shot {shot_idx})')
            else:
                print(f"shot_idx {shot_idx} out of range (max {len(data)-1})")
                return
        else:
            nshots = len(data)
            if color_list is None:
                color_list = [plt.cm.viridis(i / nshots) for i in range(nshots)]

            for i, shot in enumerate(data):

                axx.plot(x, shot, label=f'Shot {i+1}', color=color_list[i])
                axx.legend()
                axx.set_xlabel('Floquet Cycles' if title != 'Fourier Transform' else 'Frequency')
                axx.set_ylabel(ylabel)
                if log_bool:
                    axx.set_ylabel("Log " + ylabel)
                    axx.set_yscale("log")
                axx.set_title(title)


    # Make name a list if it's not already
    if not isinstance(name, list):
        name = [name]

    nplots = len(name)
    # Determine layout
    if isinstance(layout, str):
        if layout == "row":
            nrows, ncols = 1, nplots
        elif layout == "col":
            nrows, ncols = nplots, 1
        else:
            raise ValueError("layout must be 'row', 'col', or a tuple (nrows, ncols)")
    elif isinstance(layout, tuple) and len(layout) == 2:
        nrows, ncols = layout
        if nrows * ncols < nplots:
            raise ValueError("layout grid too small for number of plots")
    else:
        raise ValueError("layout must be 'row', 'col', or a tuple (nrows, ncols)")

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows) if nplots > 1 else figsize)
    axes = np.array(axes).reshape(-1) if nplots > 1 else [axes]

    for idx, n in enumerate(name):
        log_bool = log_list[idx] if log_list is not None else False
        ax = axes[idx]
        if n in ["op_real", "loop_0", "loop_e"]:
            title = {
                "op_real": "Order Parameter",
                "loop_0": "Loop Expectation Value (no anyon)",
                "loop_e": "Loop Expectation Value (e anyon)"
            }[n]
            ylabel = {
                "op_real": r"$\eta$",
                "loop_0": r"$\langle O_0 \rangle$",
                "loop_e": r"$\langle O_e \rangle$",
            }[n]
            data = entry[n]
            x = floquet_cycles

            if delta_idx == 0 or not isinstance(data, list):
                ax.plot(x, np.abs(data) if log_bool else data)
                ax.set_xlabel('Floquet Cycles')
                ax.set_ylabel(ylabel)
                if log_bool:
                    ax.set_ylabel("Log " + ylabel)
                    ax.set_yscale("log")
                ax.set_title(title)
            else:
                if log_bool:
                    data = [np.abs(shot) for shot in data]
                plot_shots(data, x, title, ylabel, axx=ax, log_bool=log_bool, color_list=color_list)

        elif n == "op_ft":
            data = entry[n]
            ylabel = r"$FT ({\eta})$"
            title = "Fourier Transform"
            x = freqs
            if delta_idx == 0 or not isinstance(data, list):
                ax.plot(x, np.abs(data))
                ax.set_xlabel('Frequency')
                ax.set_ylabel(ylabel)
                if log_bool:
                    ax.set_ylabel("Log " + ylabel)
                    ax.set_yscale("log")
                ax.set_title(title)
            else:
                plot_shots([np.abs(shot) for shot in data], x, title, ylabel, axx=ax, log_bool=log_bool, color_list=color_list)
        else:
            ax.set_title(f"Invalid name '{n}'")
            ax.axis('off')
        
        # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        fig.suptitle(f'Δ = {delta_list[delta_idx]}, T = {T_list[T_idx]}', fontsize=16)

    plt.tight_layout()

    # Optionally save the figure
    if save:
        # Expect save_dir_image to be an absolute or project-relative path prepared by the caller
        if save_dir_image is None:
            # Fallback to a sensible default in current working directory if not provided
            save_dir_image = os.path.join("figures_phasediagram", "single_entries")
        os.makedirs(save_dir_image, exist_ok=True)

        # Build a default filename if not provided
        if filename is None:
            # Ensure name text is readable
            name_part = name if isinstance(name, str) else "-".join(name)
            filename = f"single_entry_delta_{delta_list[delta_idx]:.3f}_T_{T_list[T_idx]:.3f}_{name_part}.svg"

        plt.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
        # bbox_inches='tight' is useful to make your saved figures look cleaner by removing extra whitespace.

    plt.show()

def plot_single_entry(
    delta, T, T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10,
    name = "op_real", figsize = (12,5), shot_idx = None, regularization = None, threshold=None,
    log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename: str | None = None, suffix: str = ".svg"
):
    """
    Plot a single (delta, T) entry loaded from saved results.

    If save is True, the figure is saved under a subfolder inside general_dir.
    - save_subdir: path relative to general_dir where to save the image
      (e.g., "figures/single_entries"). If None, a default will be used.
    - filename: image file name. If None, a descriptive default name will be generated.
    """

    data_grid = load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    if delta in delta_list and T in T_list:
        delta_idx = delta_list.index(delta)
        T_idx = T_list.index(T)

        # Prepare save directory inside general_dir if requested
        save_dir_image = None
        if save:
            if save_subdir is None:
                # Default subfolder inside general_dir
                save_subdir = "single_entries"
                # if log_list is not None and any(log_list):
                #     save_subdir = save_subdir + "_log"
                # if threshold is not None:
                #     save_subdir = save_subdir + f"_thresh{threshold}"
                # if regularization is not None:
                #     save_subdir = save_subdir + f"_reg{regularization}"
                # if shot_idx is not None:
                #     save_subdir = save_subdir + f"_shot{shot_idx}"
                save_subdir = save_subdir + create_suffix(log = log_list, thresh = threshold, reg = regularization, shot = shot_idx)
                save_subdir = os.path.join("figures", save_subdir)

            save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
            os.makedirs(save_dir_image, exist_ok=True)

            # Build default filename if not provided
            if filename is None:
                name_part = name if isinstance(name, str) else "-".join(name)
                filename = f"delta_{delta:.3f}_T_{T:.3f}_{name_part}"
                filename = filename + suffix

        plot_single_entry_fromdatagrid(
            data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
            name = name, figsize = figsize, shot_idx = shot_idx,
            regularization = regularization, threshold=threshold,
            log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename=filename
        )
    else:
        print("Provided delta or T not in the respective list.")
        return 
    
#plot all T for a fixed delta
def plot_all_T_fixed_delta_fromdatagrid(
    data_grid, delta_idx, T_list, delta_list, N_cycles, name = "op_real", figsize = (12,5),
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    #data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid

    if regularization is not None:
        data_grid = get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)

    if threshold is not None:
        data_grid = remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    for T_idx in range(len(T_list)):
        # Build per-plot filename when saving
        fn = None
        if save:
            name_part = name if isinstance(name, str) else "-".join(name)
            prefix = filename_prefix or f"delta_{delta_list[delta_idx]:.3f}"
            fn = f"{prefix}_T_{T_list[T_idx]:.3f}_{name_part}" + suffix

        plot_single_entry_fromdatagrid(
            data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
            name = name, figsize = figsize, log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename=fn
        )

def plot_all_T_fixed_delta(
    delta, T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10,
    name = "op_real", figsize = (12,5), regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    data_grid = load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    if delta in delta_list:
        delta_idx = delta_list.index(delta)
        save_dir_image = None
        if save:
            if save_subdir is None:
                save_subdir = "single_entries"
                save_subdir = save_subdir + create_suffix(log = log_list, thresh = threshold, reg = regularization)
                save_subdir = os.path.join("figures", save_subdir)
            save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
            os.makedirs(save_dir_image, exist_ok=True)

        plot_all_T_fixed_delta_fromdatagrid(
            data_grid, delta_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize,
            log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix
        )
    else:
        print("Provided delta not in the respective list.")
        return

def plot_all_delta_fixed_T_fromdatagrid(
    data_grid, T_idx, T_list, delta_list, N_cycles, name = "op_real", figsize = (12,5),
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid
    if regularization is not None:
        data_grid = get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)

    if threshold is not None:
        data_grid = remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    for delta_idx in range(len(delta_list)):
        fn = None
        if save:
            name_part = name if isinstance(name, str) else "-".join(name)
            prefix = filename_prefix or f"T_{T_list[T_idx]:.3f}"
            fn = f"{prefix}_delta_{delta_list[delta_idx]:.3f}_{name_part}" + suffix
        plot_single_entry_fromdatagrid(
            data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
            name = name, figsize = figsize, log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename=fn
        )

def plot_all_delta_fixed_T(
    T, T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10, name = "op_real", figsize = (12,5), 
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    data_grid = load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    if T in T_list:
        T_idx = T_list.index(T)
        save_dir_image = None
        if save:
            if save_subdir is None:
                save_subdir = "single_entries"
                save_subdir = save_subdir + create_suffix(log = log_list, thresh = threshold, reg = regularization)
                save_subdir = os.path.join("figures", save_subdir)

            save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
            os.makedirs(save_dir_image, exist_ok=True)

        plot_all_delta_fixed_T_fromdatagrid(
            data_grid, T_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize, 
            regularization = regularization, threshold=threshold, log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix
        )
    else:
        print("Provided T not in the respective list.")
        return

def plot_all_entries_fromdatagrid(
    data_grid, T_list, delta_list, N_cycles = 10, name = "op_real", figsize = (12,5), 
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid

    if regularization is not None:
        data_grid = get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)

    if threshold is not None:
        data_grid = remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    for delta_idx in range(len(delta_list)):
        for T_idx in range(len(T_list)):
            fn = None
            if save:
                name_part = name if isinstance(name, str) else "-".join(name)
                prefix = filename_prefix or "all_entries"
                fn = f"{prefix}_delta_{delta_list[delta_idx]:.3f}_T_{T_list[T_idx]:.3f}_{name_part}" + suffix
            plot_single_entry_fromdatagrid(
                data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize, 
                log_list=log_list, layout=layout, color_list=color_list,
                save=save, save_dir_image=save_dir_image, filename=fn
            )

def plot_all_entries(
    T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10, name = "op_real", figsize = (12,5), 
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg"
):
    data_grid = load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    save_dir_image = None
    if save:
        if save_subdir is None:
            save_subdir = os.path.join("figures", "single_entries")
        save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
        os.makedirs(save_dir_image, exist_ok=True)

    plot_all_entries_fromdatagrid(
        data_grid, T_list, delta_list, N_cycles, name = name, figsize = figsize, 
        regularization = regularization, threshold=threshold, log_list=log_list, layout=layout, color_list=color_list,
        save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix
    )

def create_suffix(**kwargs):
    """
    Build a suffix string for folder or filename based on provided keyword arguments.
    - If a value is a list and contains True, adds just the key (e.g., 'log').
    - If a value is not None and not False, adds key+value (e.g., 'thresh2').
    - Skips None and False values.
    """
    parts = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        if isinstance(value, list):
            # Special case for lists of bools (like log_list)
            if any(value):
                parts.append(str(key))
        else:
            parts.append(f"{key}{value}")
    return ("_" + "_".join(parts)) if parts else ""

#%% functions to save or load arrays!


def save(obj, name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(obj, file)

def load(name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj
    


def load_saved_results(T_list, delta_list, save_dir = None, general_dir = "phasediagram", data_dir = "data"):
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)
    delta_idx = {d: i for i, d in enumerate(delta_list)}
    T_idx = {t: j for j, t in enumerate(T_list)}

    for delta in delta_list:
        for T in T_list:
            fname = f"delta_{delta:.5f}_T_{T:.5f}.pkl"

            full_dir = os.path.join(general_dir, save_dir, data_dir)

            if os.path.exists(full_dir):
                fpath = os.path.join(full_dir, fname)
            else:
                print(f"Directory {full_dir} does not exist. Please check the path.")
                return None

            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    entry = pickle.load(f)
                i = delta_idx[delta]
                j = T_idx[T]
                data_grid[i, j] = entry
            else: 
                print(f"File {fpath} does not exist. Skipping. ")

    return data_grid



#%% profile output of the "order_parameter_delta_T"
""" import cProfile 
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
"""

#%% Debug usage of the manipulation functions
"""
T_list = np.linspace(1,0.1,10).tolist()

delta_list = [0,0.1,0.2,0.3]
N_cycles = 10

data_grid = load_saved_results(T_list, delta_list, save_dir = "pd_1_size31_noedge", general_dir = "phasediagram")

#remove_shots_fromdatagrid(data_grid, T_list, delta_list, result="difference", threshold=100)
get_regularized_data_grid(data_grid, T_list, delta_list, regularization = 1e-15)

"""