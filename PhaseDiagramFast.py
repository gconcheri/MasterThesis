import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed


import Class_site as site
import free_fermion_representation as f

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

def freqs(n, dt=1):
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi 

    # order frequencies in increasing order
    end = np.argmin(freqs)
    freqs = np.append(freqs[end:], freqs[:end])

    if n % 2 == 1:
        freqs += np.pi*(n-1)/n
    else:
        freqs += np.pi

    return freqs


def order_parameter_delta_T(model, fgs, delta, T, N_cycles):

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

    result = []
    
    for _ in range(N_cycles):
        result.append(fgs.order_parameter())
        fgs.update_cov_0_matrix(R0)
        fgs.update_cov_e_matrix(Re)
    result.append(fgs.order_parameter())
 
    return result
    
def order_parameter(model, T_list, delta_list, N_cycles):
    fgs = f.FermionicGaussianRepresentation(model)
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)

    for i, delta in enumerate(delta_list):
        for j, T in enumerate(T_list):
            print("delta: ", delta, ";   T: ", T)
            
            orderpar = order_parameter_delta_T(model, fgs, delta, T, N_cycles)
            freqs, ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)


            result = np.abs(ft_pi) - np.abs(ft_0)

            data_grid[i, j] = {
                'op_real': orderpar,
                'op_ft' : ft, 
                'result': result
            }
    return freqs, np.arange(N_cycles), data_grid


def compute_order_param_entry(delta, T, model, N_cycles):
    # Recreate fgs inside the function (safer for multiprocessing)
    fgs = f.FermionicGaussianRepresentation(model)

    orderpar = order_parameter_delta_T(model, fgs, delta, T, N_cycles)
    ft, ft_0, ft_pi = fourier_time(np.array(orderpar), 1)
    result = np.abs(ft_pi) - np.abs(ft_0)

    return delta, T, {
        'op_real': orderpar,
        'op_ft': ft,
        'result': result
    }

#n_jobs = -1 uses all CPU cores! if I want to limit it I can put e.g. n_jobs = 4
def order_parameter_parallel(model, T_list, delta_list, N_cycles, n_jobs=-1): 
    # Launch parallel computation across all (delta, T) pairs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_order_param_entry)(delta, T, model, N_cycles)
        for delta in delta_list
        for T in T_list
    )

    # Allocate result grid
    data_grid = np.empty((len(delta_list), len(T_list)), dtype=object)
    delta_idx = {d: i for i, d in enumerate(delta_list)}
    T_idx = {t: j for j, t in enumerate(T_list)}

    # delta_list = [0.0, 0.1, 0.2]
    # T_list = [0.1, 0.2, 0.3]

    # delta_idx = {0.0: 0, 0.1: 1, 0.2: 2}
    # T_idx     = {0.1: 0, 0.2: 1, 0.3: 2}

    for delta, T, entry in results:
        i = delta_idx[delta]
        j = T_idx[T]
        data_grid[i, j] = entry

    frequencies = freqs(N_cycles+1, 1)


    return frequencies, np.arange(N_cycles), data_grid




def plot_order_parameter_results(data_grid, delta_list, T_list, figsize = (8,6)):
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

