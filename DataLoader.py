import scipy.io.wavfile
import numpy
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas
import h5py
import librosa

class EEGDataset(Dataset):
    def __init__(self, EEG, Music, label):
        self.x = EEG
        self.y_Music = Music
        self.y_Label = label

    def __getitem__(self, index):
        return self.x[index], (self.y_Music[index], self.y_Label[index])

    def __len__(self):
        return len(self.x)

class NMEDDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y_Music, self.y_Label = y
    def __getitem__(self, index):
        return self.x[index], (self.y_Music[index], self.y_Label[index])
    def __len__(self):
        return len(self.x)

def f(x):
    return numpy.array(x)

def loadHindi(path):
    # librosa, music feature extraction
    stimulus = []
    trig1 = scipy.io.wavfile.read("./trigger/Salim Merchant - Ainvayi Ainvayi.wav")
    trig1 = numpy.array(trig1[1])
    trig1 = trig1.swapaxes(0, 1)
    trig1 = (trig1[0]+trig1[1])/2
    stimulus.append(trig1[23814:11761542])
    trig2 = scipy.io.wavfile.read("./trigger/Benni Dayal,Shalmali Kholgade - Daaru Desi (From ＂Cocktail＂).wav")
    trig2 = numpy.array(trig2[1])   
    trig2 = trig2.swapaxes(0, 1)
    trig2 = (trig2[0]+trig2[1])/2
    stimulus.append(trig2[43394:11754291])
    trig3 = scipy.io.wavfile.read("./trigger/Salim-Sulaiman - Haule Haule.wav")
    trig3 = numpy.array(trig3[1])       
    trig3 = trig3.swapaxes(0, 1)
    trig3 = (trig3[0]+trig3[1])/2
    stimulus.append(trig3[3528:11602710])
    trig4 = scipy.io.wavfile.read("./trigger/Shilpa Rao,Siddharth Mahadevan - Malang.wav")
    trig4 = numpy.array(trig4[1])   
    trig4 = trig4.swapaxes(0, 1)
    trig4 = (trig4[0]+trig4[1])/2
    stimulus.append(trig4[15579:12004236])

    a = scipy.io.loadmat("./behaveStruct_a.mat")
    # print(a)
    b = scipy.io.loadmat("./behaveStruct_b.mat")
    # print(b)
    behave = {"a": a["behaveStruct"], "b": b["behaveStruct"]}

    files = os.listdir(path)
    trial = []
    audio = []
    label = []
    # Get EEG File
    for file in files:
        fname = os.path.splitext(file)
        fname = fname[0].split("_")
        trigger = fname[0].replace("song", "data")
        type = fname[1]
        # print(file)
        rawData = scipy.io.loadmat(os.path.join("./NMED-H", file))
        sub = rawData[str(trigger)+"_"+type]
        sub = sub.swapaxes(0, 2)
        sub = sub.swapaxes(1, 2)
        # print(sub.shape)
        for i in (0, 11):
            trial.append(torch.Tensor(sub[i]))# .reshape(125, sub[i].shape[0] // 125))) # .reshape(125, sub[i].shape[0] // 125))
            print(sub[i].shape)
            # audio.append(stimulus[int(trigger.strip("data")) % 10 - 1])
            audio.append(librosa.feature.mfcc(stimulus[int(trigger.strip("data")) % 10 - 1]))
            print(librosa.feature.mfcc(stimulus[int(trigger.strip("data")) % 10 - 1]).shape)
            # print(type)
            # print(behave[type])
            label.append(behave[type][int(trigger.strip("data")) % 10 - 1][0][i])
            # print(behave[type][int(trigger.strip("data")) % 10 - 1][0][i].shape)
    # print((trial, (audio, label)))
    return (trial, (audio, label))

def loadTempo(path):

    stimulus = []
    trig1 = scipy.io.wavfile.read("./trigger/Bonobo - First Fires.wav")
    trig1 = numpy.array(trig1[1])
    trig1 = trig1.swapaxes(0, 1)
    trig1 = (trig1[0]+trig1[1])/2
    stimulus.append(trig1)
    trig2 = scipy.io.wavfile.read("./trigger/LA Priest - Oino.wav")
    trig2 = numpy.array(trig2[1])   
    trig2 = trig2.swapaxes(0, 1)
    trig2 = (trig2[0]+trig2[1])/2
    stimulus.append(trig2)
    trig3 = scipy.io.wavfile.read("./trigger/Daedelus - Tiptoes.wav")
    trig3 = numpy.array(trig3[1])       
    trig3 = trig3.swapaxes(0, 1)
    trig3 = (trig3[0]+trig3[1])/2
    stimulus.append(trig3)
    trig4 = scipy.io.wavfile.read("./trigger/Croquet Club - Careless Love.wav")
    trig4 = numpy.array(trig4[1])   
    trig4 = trig4.swapaxes(0, 1)
    trig4 = (trig4[0]+trig4[1])/2
    stimulus.append(trig4)
    trig5 = scipy.io.wavfile.read("./trigger/Thievery Corporation - Lebanese Blonde.wav")
    trig5 = numpy.array(trig5[1])   
    trig5 = trig5.swapaxes(0, 1)
    trig5 = (trig5[0]+trig5[1])/2
    stimulus.append(trig5)
    trig6 = scipy.io.wavfile.read("./trigger/Polo & Pan - Canopée.wav")
    trig6 = numpy.array(trig6[1])   
    trig6 = trig6.swapaxes(0, 1)
    trig6 = (trig6[0]+trig6[1])/2
    stimulus.append(trig6)
    trig7 = scipy.io.wavfile.read("./trigger/Kazy Lambist - Doing Yoga.wav")
    trig7 = numpy.array(trig7[1])   
    trig7 = trig7.swapaxes(0, 1)
    trig7 = (trig7[0]+trig7[1])/2
    stimulus.append(trig7)
    trig8 = scipy.io.wavfile.read("./trigger/RÜFÜS DU SOL - Until the Sun Needs to Rise.wav")
    trig8 = numpy.array(trig8[1])   
    trig8 = trig8.swapaxes(0, 1)
    trig8 = (trig8[0]+trig8[1])/2
    stimulus.append(trig8)
    trig9 = scipy.io.wavfile.read("./trigger/The Knife - Silent Shout.wav")
    trig9 = numpy.array(trig9[1])   
    trig9 = trig9.swapaxes(0, 1)
    trig9 = (trig9[0]+trig9[1])/2
    stimulus.append(trig9)
    trig10 = scipy.io.wavfile.read("./trigger/David Bowie - The Last Thing You Should Do (2021 Remaster).wav")
    trig10 = numpy.array(trig10[1])   
    trig10 = trig10.swapaxes(0, 1)
    trig10 = (trig10[0]+trig10[1])/2
    stimulus.append(trig10)
    
    
    files = os.listdir(path)
    trial = []
    audio = []
    label = []

    f = h5py.File("behavioralRatings.mat")
    a_group_key = list(f.keys())[0]
    behave = f[a_group_key][()] 
    behave = behave.swapaxes(0, 1)
    behave = behave.swapaxes(1, 2)
    # (behave.shape)
    for file in files:
        fname = os.path.splitext(file)
        rawData = scipy.io.loadmat(os.path.join("./NMED-T", file))
        trigger = fname[0].split("_")[0]
        trigger = trigger.replace("song", "data")
        sub = rawData[str(trigger)]
        sub = sub.swapaxes(0, 2)
        sub = sub.swapaxes(1, 2)  
        for i in (0, 19):
            trial.append(torch.Tensor(sub[i]))
            audio.append(librosa.feature.mfcc(stimulus[int(trigger.strip("data")) % 10 - 1]))
            label.append(behave[int(trigger.strip("data")) - 21][i])

    return (trial, (audio, label))
