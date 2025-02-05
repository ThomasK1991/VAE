from helper_code import *
from scipy import signal
from tqdm import tqdm
import warnings
import pandas as pd

class dataset:
    classes = ['164889003','164890007','6374002','426627000','733534002',
               '713427006','270492004','713426002','39732003','445118002',
               '164947007','251146004','111975006','698252002','426783006',
               '284470004','10370003','365413008','427172004','164917005',
               '47665007','427393009','426177001','427084000','164934002',
               '59931005']
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'],
                          ['284470004', '63593006'],
                          ['427172004', '17338001'],
                          ['733534002','164909002']]
    def __init__(self,header_files):
        self.files = []
        self.sample = True
        self.num_leads = None
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h
            tmp['record'] = h.replace('.hea','.mat')
            hdr = load_header(h)
            tmp['nsamp'] = get_nsamp(hdr)
            tmp['leads'] = get_leads(hdr)
            tmp['age'] = get_age(hdr)
            tmp['sex'] = get_sex(hdr)
            tmp['dx'] = get_labels(hdr)
            tmp['fs'] = get_frequency(hdr)
            tmp['target'] = np.zeros((26,))
            #print(f"Original dx for file {h}: {tmp['dx']}")
            tmp['dx'] = replace_equivalent_classes(tmp['dx'], dataset.equivalent_classes)
            #print(f"Replaced dx for file {h}: {tmp['dx']}")

            for dx in tmp['dx']:
                # in SNOMED code is in scored classes
                if dx in dataset.classes:
                    idx = dataset.classes.index(dx)
                    tmp['target'][idx] = 1
            self.files.append(tmp)

        # set filter parameters
        self.b, self.a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

        self.files = pd.DataFrame(self.files)

def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0]  # Use the first class as the representative class.
    return classes

def get_nsamp(header):
    #print("Header content:", header)
    return int(header.split('\n')[0].split(' ')[3])


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
