import numpy as np

def linear_convolution_fft(x, h):
    """
    Perform linear convolution using FFT.
    
    Parameters:
    x : list or array
        The input signal.
    h : list or array
        The impulse response.
    
    Returns:
    y : array
        The result of the convolution.
    """
    # Length of the result
    len_y = len(x) + len(h) - 1
    
    # Perform FFT on both signals with zero-padding to match the length of the result
    X = np.fft.fft(x, len_y)
    H = np.fft.fft(h, len_y)
    
    # Multiply in the frequency domain
    Y = X * H
    
    # Perform inverse FFT to get the result back in the time domain
    y = np.fft.ifft(Y)
    
    # Return the real part of the result (imaginary part should be negligible)
    return np.real(y)

# Example usage
x = [1, 2, 3]  # Input sequence
h = [0, 1, 0.5]  # Impulse response
result = linear_convolution_fft(x, h)

print("Input sequence (x):", x)
print("Impulse response (h):", h)
print("Linear Convolution Result (y):", result)

