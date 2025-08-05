import numpy as np


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
