import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def extract_features(f):
    y, _ = librosa.load(f)
    signal = y[0:int(0.9 * _)] #take the first 0.9 seconds
    # get Mel-spectrogram
    S = librosa.feature.melspectrogram(signal)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB)
    plt.show()
    return np.ndarray.flatten(S_DB)
  
coughs = ['folder name']
for cough in coughs:
    path = glob.glob('path+'/*.wav')
    for f in path:
        features = extract_features(f)
