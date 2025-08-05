import numpy as np
import matplotlib.pyplot as plt

import Class_site as site
import free_fermion_representation as f
import pickle




def fourier_time(t_series, dt, sigma = 0.4, nw=4, gauss = True):
    """ Calculates the FFT of a time series, applying either a Gaussian window function if gauss = True, or a cosine if gauss = False. """

    # Gaussian or cosine window function
    n = len(t_series)
    t_list = np.arange(n)*dt

    if gauss == True:
        gauss = [np.exp(-1/2.*(i/(sigma * n))**2) for i in np.arange(n)]
        input_series = gauss * t_series
    else:
        Wfunlist = [np.cos(np.pi*t_list[t]/(2*t_list[-1]))**nw  for t in range(n)]
        input_series = Wfunlist * t_series

    # Fourier transform
    ft = np.fft.fft(input_series) / np.sqrt(n)
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi 

    # order frequencies in increasing order
    end = np.argmin(freqs)
    freqs = np.append(freqs[end:], freqs[:end])


    # Take into account the additional minus sign in the time FT
    ft = np.append(ft, ft[0])
    ft = ft[::-1]
    ft = ft[:-1]

    ft_0 = ft[0]

    if n % 2 == 1:
        freqs += np.pi*(n-1)/n
        ft_pi = (ft[(n-1)//2]+ft[(n+1)//2])/2 # // even though (n-1)/2 is always integer it will yield n.0, I use // to force it to be integer
    else:
        freqs += np.pi
        ft_pi = ft[n//2]

    return freqs, ft, ft_0, ft_pi



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




def plot_order_parameter_results(data_grid, T_list, delta_list, figsize = (8,6)):
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


def save_file(obj, name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(obj, file)

def load_file(name, doit = True):
    if doit:
        with open(f'{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
    return obj



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
    result = order_parameter_delta_T(model, fgs, delta=delta, T=T, N_cycles=N_cycles)

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
