import scipy
import numpy
import os
from torch.utils.data import Dataset, DataLoader
import pandas
import librosa

class NMEDDataSet(Dataset):
    def __init__(self, EEG, Music, label):
        self.x = EEG
        self.y = (Music, label)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def loadHindi(path):

    # Load Behaviour Data

    # Load EEG Dataset
    files = os.listdir(path)
    for file in files:
        fname = os.path.splitext(file)
        trigger = fname.strip("_")[0]
        type = fname.strip("_")[1]
        rawData = scipy.io.loadmat(os.join.path("./NMED-H", fname))
        data = pandas.DataFrame(rawData[trigger.replace("song", "data") + "_" + type])
        # fs = rawData["fs"][0][0]
        sub = numpy.array([])
        sub_temp = rawData["subs"+str(trigger)+"_"+type][0]
        for key in sub_temp:
            sub.append(key)

    # librosa, music feature extraction
    stimulus = []
    trig1 = librosa.load(".\\trigger\\Salim Merchant - Ainvayi Ainvayi.mp3")
    stimulus.append(trig1[23814:11761542])
    trig2 = librosa.load(".\\trigger\\Benni Dayal,Shalmali Kholgade - Daaru Desi (From ＂Cocktail＂).mp3", sr = 44100)
    stimulus.append(trig2[43394:11754291])
    trig3 = librosa.load(".\\trigger\\Salim-Sulaiman - Haule Haule.mp3", sr = 44100)
    stimulus.append(trig3[3528:11602710])
    trig4 = librosa.load(".\\trigger\\Shilpa Rao,Siddharth Mahadevan - Malang.mp3")
    stimulus.append(trig4[15579:12004236])

def loadTempo(files):

    for file in files:

        = scipy.io.loadmat()

def buildDataSet():
    