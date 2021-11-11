import torch
from torch import nn
from torch.autograd import Variable

class SequentialLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers):
        super(SequentialLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size / num features
        self.hidden_size = hidden_size  # hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm


    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        return hn