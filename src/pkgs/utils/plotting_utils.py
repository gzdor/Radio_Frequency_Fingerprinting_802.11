#!/usr/bin/env python3
"""
#++++++++++++++++++++++++++++++++++++++++++++++

    Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

    Totality of this code is non-proprietary and may be used at will. 

#++++++++++++++++++++++++++++++++++++++++++++++


Description: 

@brief a module defining a variety of IQ signal data plotting utils.

@author: Greg Zdor (gzdor@icloud.com)

@date Date_Of_Creation: 3/15/2023 

@date Last_Modification 3/15/2023 

No Copyright - use at will

"""

import pywt
import copy
import numpy as np 
import matplotlib.pyplot as plt 


# Define spectrogram plotting function 
def create_spectrogram(iq: np.ndarray, fs: float, fc: float) -> None:
    """
    @brief plots spectrogram of input 

    @type iq np.ndarray
    @param iq np.complex128 1D array to plot

    @type fs float 
    @param fs the sampling rate of the input sequence, in hertz 

    @type fc float 
    @param fc the center frequency of the input sequence, in hertz
    """
    
    # Call spectrogram function 
    plt.figure(figsize=(12,6)) 
    plt.specgram(
            iq, 
            Fs = fs, 
            Fc = fc, 
            mode = 'psd', 
            scale = 'dB') 
    plt.colorbar(label = 'dB')
    plt.xlabel('Time (seconds)', fontsize=24)
    plt.ylabel('Frequency (hertz', fontsize=24)
    plt.title('Spectrogram', fontsize=24)
    print(f'\n\nCreating spectrogram plot.\n\n')
    plt.show()


# Define power spectral density (psd) plotting function 
def create_dft(iq: np.ndarray, fft_size: int, fs: float) -> None:
    """
    @brief plots discrete Fourier transform of input sequence 

    @type iq np.ndarray
    @param iq np.complex128 1D array to plot

    @type fft_size int
    @param fft_size number of samples to use in DFT 

    @type fs float 
    @param fs the sampling rate of the input sequence, in hertz 
    """

    # Compute DFT 
    X = np.fft.fftshift(np.fft.fft(iq,fft_size))
    f = np.arange(start = -fs/2, stop = fs/2, step = fs/fft_size)

    # Compute power in dB  - frequency domain 
    X_power_db = 20*np.log10((1/np.sqrt(2))*np.abs(X)/fft_size)

    # DFT plot 
    plt.figure(figsize=(12,6))  
    plt.plot(f, X_power_db, label = 'FFT power')
    plt.xlabel('Frequency (hertz)', fontsize=24)
    plt.ylabel('Power (dB)', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('DFT Magnitude', fontsize=24)
    plt.legend(fontsize=16)
    print(f'\n\nCreating frequency domain plot.\n\n')
    plt.show()


# Define time domain plotting function
def create_time_domain_plot(iq: np.ndarray) -> None:
    """ 
    @brief plots input sequence across time 

    @type iq np.ndarray
    @param iq np.complex128 1D array to plot
    """

    # Compute power in dB - time domain 
    x_power_db = 20*np.log10(np.abs(iq))

    # Time plot 
    plt.figure(figsize=(12,6))  
    plt.plot(x_power_db, label = 'Time Power')
    plt.xlabel('Time (samples)', fontsize=24)
    plt.ylabel('Power (dB)', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Time Magnitude', fontsize=24)
    plt.legend(fontsize=16)
    print(f'\n\nCreating time domain plot.\n\n')
    plt.show()


# Define wavelets transform plotting function
def create_wavelets_plot(iq: np.ndarray) -> None:
    """
    @brief plots discrete wavelet transform of input sequence

    For theory on wavelet transforms, see: 
        - https://www.mathworks.com/discovery/wavelet-transforms.html 
        - https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html 
        - https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34 

    The wavelet transform is based on taking the first derivative of a Gaussian distribution
    function and convolve the signal with this wavelet, and do this with a set of wavelets 
    at a variety of scales. 

    @type iq np.ndarray
    @param iq np.complex128 1D array to plot
    """

    # Make explicit copy of iq 
    iq = copy.copy(iq)

    # Compute discrete wavelet transform 
    wavelet_transform = pywt.dwt(iq, 'db1') #TODO this needs confirming this syntax is right - check PyWavelets docs 

    # Create plot 
    plt.figure(figsize=(12,12)) 
    plt.plot(wavelet_transform, label = 'wavelet transform')
    plt.xlabel('Wavelet transform', fontsize=24)
    plt.ylabel('Samples', fontsize=24)
    plt.title('Discrete wavelet transform', fontsize=24)
    print(f'\n\nCreating discrete wavelet transform plot.\n\n')
    plt.show() 


# Define constellation plotting function
def create_constellation_plot(iq: np.ndarray) -> None:
    """
    @brief plots a constellation diagram of the input, 
    input must be complex-valued.

    @type iq np.ndarray
    @param iq np.complex128 1D array to plot
    """

    # Check input is complex-valued 

    # Get I and Q components 
    imaginary =  np.imag(iq)
    real = np.real(iq)

    # Create plot 
    plt.figure(figsize=(12,12)) 
    plt.scatter(real, imaginary, label = 'IQ points')
    plt.xlabel('In-phase (real)', fontsize=24)
    plt.ylabel('Quadrature (imaginary)', fontsize=24)
    plt.title('Constellation diagram', fontsize=24)
    print(f'\n\nCreating constellation plot.\n\n')
    plt.show()