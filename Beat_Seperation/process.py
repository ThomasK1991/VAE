from helper_code import *
from scipy import signal

data = load_recording("C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VAE\Beat_Seperation\Test_Data\JS00001.mat")
hdr = load_header("C:\Users\Thomas Kaprielian\Documents\Master's Thesis\VAE\Beat_Seperation\Test_Data\JS00001.hea")
# expand to 12 lead setup if original signal has less channels
data = np.nan_to_num(data)
fs = get_frequency(hdr)
# resample to 500hz
if fs == float(1000):
    data = signal.resample_poly(data, up=1, down=2, axis=-1)  # to 500Hz
    fs = 500
elif fs == float(500):
    pass
else:
    data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
    fs = 500

data = signal.filtfilt(self.b, self.a, data)

if self.sample:
    fs = int(fs)
    # random sample signal if len > 8192 samples
    if data.shape[-1] >= 8192:
        idx = data.shape[-1] - 8192-1
        idx = np.random.randint(idx)
        data = data[:, idx:idx + 8192]

mu = np.nanmean(data, axis=-1, keepdims=True)
std = np.nanstd(data, axis=-1, keepdims=True)
#std = np.nanstd(data.flatten())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = (data - mu) / std
data = np.nan_to_num(data)
