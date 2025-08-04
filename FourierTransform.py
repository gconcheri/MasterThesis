import numpy as np

#Markus Drescher genius FT ahah

def fourier_time(t_series, dt, sigma = 0.4, nw=4, gauss = True):
    """ Calculates the FFT of a time series, applying either a Gaussian window function if gauss = True, or a cosine if gauss = False. """

    # Gaussian or cosine window function
    n = len(t_series)
    print(n)
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

    # shift results accordingly
    ftShifted = np.append(ft[end:], ft[:end])

    # Take into account the additional minus sign in the time FT
    if len(ftShifted)%2 == 0:
        ftShifted = np.append(ftShifted, ftShifted[0])
        ftShifted = ftShifted[::-1]
        ftShifted = ftShifted[:-1]

    else:
        ftShifted = ftShifted[::-1]


    return freqs, ftShifted


#better to have number of cycles = even! in this way I can exactly compute n(pi)!
# otherwise I would have to do mean between 2 values 
#then I shift frequencies exactly to 0, and then the other steps can be done 