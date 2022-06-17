from turtle import forward
import torch
import torch.nn as nn
import scipy
import numpy
import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils import weight_norm
from TCN import TemporalConvNet

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
    def __init__(self, channels, sample_frequency=512, time_basis=1):
        super(Net1037, self).__init__()
        channel_shape = numpy.ones(sample_frequency*time_basis)*channels
        self.tcn = TemporalConvNet(num_inputs=sample_frequency*time_basis, num_channels=channel_shape)
        self.lstm1 = nn.LSTM()
        self.lstm2 = nn.LSTM()
        self.lstm3 = nn.LSTM()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
    def forward(self, x, type):
        # EEG Input, 512Hz 
        x = self.tcn(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)

        # mfcc
        mfcc = self.fc1(x)
        mfcc = nn.Sigmoid(mfcc)

        # label
        if type == "NMED-T":
            labelT = self.fc2(x)
            labelT = nn.Sigmoid(labelT) * 10.0
            return mfcc, labelT
        elif type == "NMED-H":
            labelH = self.fc3(x)
            labelH = nn.Sigmoid(labelH) * 10.0
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
        self.model.train() if mode is 'train' else self.model.eval()
        # epoch_loss = torch.zeros(15).cuda()
        all_prediction_T_tag, all_prediction_T_mfcc, all_prediction_H_tag, all_prediction_H_mfcc = []
        all_label_T_tag, all_label_T_mfcc, all_label_H_tag, all_label_H_mfcc = []
        self.model = self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        Tensor = torch.FloatTensor

        epoch_tag_loss, epoch_mfcc_loss = 0.0, 0.0
        # Enumerate Data from NMED-T
        for batch, (x, y) in enumerate(DataLoaderT):
            EEG = x.type(Tensor)
            # Label (Enjoyment, Similarity)
            FFT, label = y
            FFT = FFT.type(Tensor)
            label = label.type(Tensor)
            
            EEG, label = EEG.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), label.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            # Inference
            mfcc, tag = self.model(EEG, "NMED-T")

            # Tag Loss
            TagLoss = self.criterion(label, tag)

            # mfcc Loss
            MFCCLoss = self.criterion(mfcc, FFT)
            
            all_prediction_T_tag.extend(tag.cpu().detach().numpy())
            all_label_T_tag.extend(label.cpu().detach().numpy())
            all_prediction_T_mfcc.extend(mfcc.cpu().detach().numpy())
            all_label_T_mfcc.extend(FFT.cpu().detach().numpy())

            if mode is 'train':
                self.optimizer.zero_grad()
                TagLoss.backward()
                MFCCLoss.backward()
                self.optimizer.step()
            
            epoch_tag_loss += TagLoss.item()
            epoch_mfcc_loss += MFCCLoss.item()
            # print(global_grad_saver)
        
        avg_loss_T_tag = epoch_tag_loss/len(DataLoaderT.dataset)
        avg_loss_T_MFCC = epoch_mfcc_loss / len(DataLoaderT.dataset)
        
        epoch_tag_loss, epoch_mfcc_loss = 0.0, 0.0
        # Enumerate Data from NMED-H
        for batch, (x, y) in enumerate(DataLoaderH):
            EEG = x.type(Tensor)
            # Label (Enjoyment, Similarity)
            FFT, label = y
            FFT = FFT.type(Tensor)
            label = label.type(Tensor)

            EEG, label = EEG.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), label.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            # Inference
            mfcc, tag = self.model(EEG, "NMED-T")

            # Tag Loss
            TagLoss = self.criterion(label, tag)
            # mfcc Loss
            MFCCLoss = self.criterion(mfcc, FFT)
            
            all_prediction_H_tag.extend(tag.cpu().detach().numpy())
            all_label_H_tag.extend(label.cpu().detach().numpy())
            all_prediction_H_mfcc.extend(mfcc.cpu().detach().numpy())
            all_label_H_mfcc.extend(FFT.cpu().detach().numpy())

            if mode is 'train':
                self.optimizer.zero_grad()
                TagLoss.backward()
                MFCCLoss.backward()
                self.optimizer.step()
            
            epoch_tag_loss += TagLoss.item()
            epoch_mfcc_loss += MFCCLoss.item()
            # print(global_grad_saver)
        
        avg_loss_H_tag = epoch_tag_loss/len(DataLoaderT.dataset)
        avg_loss_H_MFCC = epoch_mfcc_loss / len(DataLoaderT.dataset)

        return (all_prediction_T_tag, all_prediction_T_mfcc, all_prediction_H_tag, all_prediction_H_mfcc), (all_label_T_tag, all_label_T_mfcc, all_label_H_tag, all_label_H_mfcc), (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC)

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop




if __name__ == "__main__":
    x_Hindi, y_Hindi = DataLoader.loadHindi("./NMED-H")
    x_Tempo, y_Tempo = DataLoader.loadTempo("./NMED-T")

    x_train_Hindi, x_valid_Hindi, y_train_Hindi, y_valid_Hindi = train_test_split(x_Hindi, y_Hindi, train_size=0.9)
    x_train_Tempo, x_valid_Tempo, y_train_Tempo, y_valid_Tempo = train_test_split(x_Tempo, y_Tempo, train_size=0.9)
    train_set_Hindi = DataLoader.NMEDDataSet(x_train_Hindi, y_train_Hindi)
    vaild_set_Hindi = DataLoader.NMEDDataSet(x_valid_Hindi, y_valid_Hindi)
    train_set_Tempo = DataLoader.NMEDDataSet(x_train_Tempo, y_train_Tempo)
    vaild_set_Tempo = DataLoader.NMEDDataSet(x_valid_Tempo, y_valid_Tempo)
    runner = Runner()

    from sklearn.metrics import r2_score, accuracy_score
    for epoch in range(num_epochs):
        train_y_pred, train_y_truth, train_loss = runner.run(train_set_Hindi, train_set_Tempo, 'train')
        valid_y_pred, valid_y_truth, valid_loss = runner.run(vaild_set_Hindi, vaild_set_Tempo,  'eval')
        # Label R2
        (pred_T_tag, pred_T_mfcc, pred_H_tag, pred_H_mfcc) = train_y_pred 
        (label_T_tag, label_T_mfcc, label_H_tag, label_H_mfcc) = train_y_truth 
        (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC) = train_loss
        trainLabelError = r2_score(numpy.concatenate(pred_T_tag,pred_H_tag), numpy.concatenate(label_T_tag, label_H_tag))
        trainMFCCError = r2_score(numpy.concatenate(pred_T_mfcc,pred_H_mfcc), numpy.concatenate(label_T_mfcc, label_H_mfcc))
        print(trainLabelError, trainMFCCError, train_loss)

        (pred_T_tag, pred_T_mfcc, pred_H_tag, pred_H_mfcc) = valid_y_pred 
        (label_T_tag, label_T_mfcc, label_H_tag, label_H_mfcc) = valid_y_truth 
        (avg_loss_T_tag, avg_loss_T_MFCC, avg_loss_H_tag, avg_loss_H_MFCC) = valid_loss
        validLabelError = r2_score(numpy.concatenate(pred_T_tag,pred_H_tag), numpy.concatenate(label_T_tag, label_H_tag))
        validMFCCError = r2_score(numpy.concatenate(pred_T_mfcc,pred_H_mfcc), numpy.concatenate(label_T_mfcc, label_H_mfcc))
        print(validLabelError, validMFCCError, valid_loss)

        if epoch % 10 == 0:
            torch.save(runner.model, "model"+str(epoch)+".pth")
        if runner.early_stop(valid_loss, epoch + 1):
            torch.save(runner.model, "model"+str(epoch)+".pth")
            break
    
    print("Training Finished")