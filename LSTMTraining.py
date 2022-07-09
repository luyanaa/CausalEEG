from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy
import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils import weight_norm
from TCN import TCN

num_epochs = 100

def set_device(model, Tensor, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model =model.to(device)
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    return model, Tensor

class Net1037(nn.Module):

    # Default Parameters
    nhid = 25
    levels = 8
    hidden_size = 64
    mfcc_length = 20
    TLength=4
    HLength=2

    def __init__(self, channels=125, sample_frequency=512, time_basis=1):
        super(Net1037, self).__init__()
        # self.tcn = TCN(input_size=channels, output_size=self.hidden_size, num_channels=[self.nhid] * self.levels)
        self.lstm = nn.LSTM(input_size = channels, hidden_size = self.hidden_size, num_layers = 3, batch_first=True)
        self.fc1 = nn.Linear(in_features = self.hidden_size,  out_features = self.mfcc_length)
        self.fc2 = nn.Linear(in_features = self.hidden_size, out_features=self.TLength)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid0 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features = self.hidden_size, out_features=self.HLength)
    def forward(self, x, type, hidden=None):
        # EEG Input, 512Hz 
        # x = self.tcn(x)
        x, (hidden,cell) = self.lstm(x.transpose(0,1), hidden)
        # mfcc
        mfcc = self.fc1(x)
        mfcc = self.sigmoid0(mfcc) * 1000.0
        mfcc = mfcc.transpose(0, 1)
        # label
        if type == "NMED-T":
            labelT = self.fc2(x)
            labelT = self.sigmoid1(labelT) * 10.0
            return mfcc, labelT
        elif type == "NMED-H":
            labelH = self.fc3(x)
            labelH = self.sigmoid2(labelH) * 10.0
            return mfcc, labelH
        else: 
            Exception("Not Implemented!")

class Runner(object):
    def __init__(self):
        self.model = Net1037()
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-8
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.2, patience=5, verbose=True)
        self.device = [1, ]
        self.criterion = torch.nn.MSELoss()

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, DataLoaderT, DataLoaderH, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()
        # epoch_loss = torch.zeros(15).cuda()
        all_prediction_T_tag, all_prediction_T_mfcc, all_prediction_H_tag, all_prediction_H_mfcc = [], [], [], []
        all_label_T_tag, all_label_T_mfcc, all_label_H_tag, all_label_H_mfcc = [], [], [], []
        self.model = self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        Tensor = torch.FloatTensor

        hidden_train = None
        epoch_tag_loss, epoch_mfcc_loss = 0.0, 0.0
        # Enumerate Data from NMED-T
        for batch, (EEG, (FFT, label)) in enumerate(DataLoaderT):
            EEG = torch.Tensor(EEG)
            # Label (Enjoyment, Similarity)
            FFT_1 = FFT
            FFT = torch.Tensor(FFT)
            label = torch.Tensor(label)
            FFT = FFT.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            EEG = EEG.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            label = label.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            # Inference
            mfcc, tag = self.model(EEG, "NMED-T")
            tag = torch.mean(tag, axis = 0)
            FFT = torch.nn.functional.interpolate(FFT.unsqueeze(0), size=mfcc.shape[1]).squeeze(0)
            # mfcc1.resize(20, FFT_1.shape[1])
            # mfcc = mfcc.squeeze(1)
            # Tag Loss
            TagLoss = self.criterion(label, tag)

            # mfcc Loss
            MFCCLoss = self.criterion(mfcc, FFT)
            
            loss = TagLoss+ MFCCLoss
            all_prediction_T_tag.extend(tag.cpu().detach().numpy())
            all_label_T_tag.extend(label.cpu().detach().numpy())
            all_prediction_T_mfcc.extend(mfcc.cpu().detach().numpy())
            all_label_T_mfcc.extend(FFT.cpu().detach().numpy())

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_tag_loss += TagLoss.item()
            epoch_mfcc_loss += MFCCLoss.item()
        
        avg_loss_T_tag = epoch_tag_loss/len(DataLoaderT)
        avg_loss_T_MFCC = epoch_mfcc_loss / len(DataLoaderT)
        
        epoch_tag_loss, epoch_mfcc_loss = 0.0, 0.0
        # Enumerate Data from NMED-H
        for batch, (EEG, (FFT, label)) in enumerate(DataLoaderH):
            EEG = torch.Tensor(EEG)
            FFT = torch.Tensor(FFT)
            label = torch.Tensor(label)
            FFT = FFT.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            EEG = EEG.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            label = label.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            # Inference
            mfcc, tag = self.model(EEG, "NMED-H")
            tag = torch.mean(tag, axis = 0)
            FFT = torch.nn.functional.interpolate(FFT.unsqueeze(0), size=mfcc.shape[1]).squeeze(0)
            # Tag Loss
            TagLoss = self.criterion(label, tag)

            # mfcc Loss
            MFCCLoss = self.criterion(mfcc, FFT)

            all_prediction_H_tag.extend(tag.cpu().detach().numpy())
            all_label_H_tag.extend(label.cpu().detach().numpy())
            all_prediction_H_mfcc.extend(mfcc.cpu().detach().numpy())
            all_label_H_mfcc.extend(FFT.cpu().detach().numpy())

            loss = TagLoss + MFCCLoss
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_tag_loss += TagLoss.item()
            epoch_mfcc_loss += MFCCLoss.item()
        
        avg_loss_H_tag = epoch_tag_loss/len(DataLoaderH)
        avg_loss_H_MFCC = epoch_mfcc_loss / len(DataLoaderH)

        return (all_prediction_T_tag, all_prediction_T_mfcc, all_prediction_H_tag, all_prediction_H_mfcc), (all_label_T_tag, all_label_T_mfcc, all_label_H_tag, all_label_H_mfcc), (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC)

    # Early stopping function for given validation loss
    # def early_stop(self, loss, epoch):
    #     self.scheduler.step(loss, epoch)
    #     self.learning_rate = self.optimizer.param_groups[0]['lr']
    #     stop = self.learning_rate < self.stopping_rate
    #     return stop

if __name__ == "__main__":
    x_Hindi, y_Hindi = DataLoader.loadHindi("./NMED-H")
    x_Tempo, y_Tempo = DataLoader.loadTempo("./NMED-T")
    x_train_Hindi, x_valid_Hindi = x_Hindi, x_Hindi
    x_train_Tempo, x_valid_Tempo = x_Tempo, x_Tempo
    y_train_Hindi, y_valid_Hindi = y_Hindi, y_Hindi
    y_train_Tempo, y_valid_Tempo = y_Tempo, y_Tempo
    # x_train_Hindi, x_valid_Hindi, y_train_Hindi, y_valid_Hindi = train_test_split(x_Hindi, y_Hindi, train_size=0.9)
    # x_train_Tempo, x_valid_Tempo, y_train_Tempo, y_valid_Tempo = train_test_split(x_Tempo, y_Tempo, train_size=0.9)
    train_set_Hindi = DataLoader.NMEDDataSet(x_train_Hindi, y_train_Hindi)
    vaild_set_Hindi = DataLoader.NMEDDataSet(x_valid_Hindi, y_valid_Hindi)
    train_set_Tempo = DataLoader.NMEDDataSet(x_train_Tempo, y_train_Tempo)
    vaild_set_Tempo = DataLoader.NMEDDataSet(x_valid_Tempo, y_valid_Tempo)
    runner = Runner()

    from sklearn.metrics import r2_score
    for epoch in range(num_epochs):
        train_y_pred, train_y_truth, train_loss = runner.run(train_set_Hindi, train_set_Tempo, 'train')
        valid_y_pred, valid_y_truth, valid_loss = runner.run(vaild_set_Hindi, vaild_set_Tempo,  'eval')
        # Label R2
        (pred_T_tag, pred_T_mfcc, pred_H_tag, pred_H_mfcc) = train_y_pred 
        (label_T_tag, label_T_mfcc, label_H_tag, label_H_mfcc) = train_y_truth 
        (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC) = train_loss
        # trainLabelError = r2_score(numpy.concatenate(pred_T_tag,pred_H_tag), numpy.concatenate(label_T_tag, label_H_tag))
        # trainMFCCError = r2_score(numpy.concatenate(pred_T_mfcc,pred_H_mfcc), numpy.concatenate(label_T_mfcc, label_H_mfcc))
        print(train_loss)

        (pred_T_tag, pred_T_mfcc, pred_H_tag, pred_H_mfcc) = valid_y_pred 
        (label_T_tag, label_T_mfcc, label_H_tag, label_H_mfcc) = valid_y_truth 
        (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC) = valid_loss
        # validLabelError = r2_score(numpy.concatenate(pred_T_tag,pred_H_tag), numpy.concatenate(label_T_tag, label_H_tag))
        # validMFCCError = r2_score(numpy.concatenate(pred_T_mfcc,pred_H_mfcc), numpy.concatenate(label_T_mfcc, label_H_mfcc))
        print(valid_loss)

        if epoch % 10 == 0:
            torch.save(runner.model, "model"+str(epoch)+".pth")
        # if runner.early_stop(valid_loss, epoch + 1):
        #     torch.save(runner.model, "model"+str(epoch)+".pth")
        #     break
    
    print("Training Finished")