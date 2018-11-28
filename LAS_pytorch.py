import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pack_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from wordmap import wordMap
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

data = np.load("train.npy", encoding="latin1")
label = np.load("train_label_preprocessed.npy", encoding='ASCII')
# data = np.load("dev.npy", encoding="latin1")
# label = np.load("dev_label_preprocessed.npy", encoding='ASCII')

batch_size = 16
attention_dim = 256
hidden_dim = 512

class WSJDataset(Dataset):
    def __init__(self, wsj_data, wsj_label):
        self.data = wsj_data
        lens = [len(i) for i in self.data]
        lens = [(lens[i]//8)*8 for i in range(len(lens))]
        self.data = [self.data[i][:lens[i]] for i in range(len(self.data))]
        self.label = wsj_label
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])
    def __len__(self):
        return len(self.data)

def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i].to(DEVICE) for i in seq_order]
    targets = [targets[i].to(DEVICE) for i in seq_order]

    labels = torch.IntTensor(batch_size, max(target.size(0) for target in targets)).zero_()
    targets_lens = torch.tensor([tensor.size(0) for tensor in targets])
    for i in range(len(targets)):
        labels[i][:targets[i].size(0)] = targets[i]
    targets = labels.long().to(DEVICE)
    inputs_lens = torch.tensor([tensor.size(0) for tensor in inputs])

    return inputs, targets, inputs_lens, targets_lens

class Listen(nn.Module):
    def __init__(self, input_size=40):
        super(Listen,self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, bidirectional=True)
        #output_size = 1024
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, bidirectional=True)
        #output_size = 1024
        self.lstm3 = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, bidirectional=True)
        
        self.key = nn.Linear(1024, attention_dim)
        self.value = nn.Linear(1024, attention_dim)
        
    def forward(self, seq_list):
        """
        seq_list should be packed/padded shape(max_seq_len*batch*40)
        """
        #input: max_seq_len*batch*(256*2), output: max_seq_len/2*batch*(256*4)
        output1, _ = self.lstm1(pack_sequence(seq_list), None)
        output1, lens = pad_packed_sequence(output1) 
        output1 = output1.permute(1,0,2)
        output1 = output1.contiguous().view(output1.size(0), (int)(output1.size(1)/2), output1.size(2)*2)
        output1 = output1.permute(1,0,2)
        lens = lens/2
        #input: max_seq_len/2*batch*(256*4), output: max_seq_len/2*batch*(256*2)
        output2, _ = self.lstm2(pack_padded_sequence(output1, lens), None)
        #input: max_seq_len/2*batch*(256*2), output: max_seq_len/4*batch*(256*4)
        output2, lens = pad_packed_sequence(output2)
        output2 = output2.permute(1,0,2)
        output2 = output2.contiguous().view(output2.size(0), (int)(output2.size(1)/2), output2.size(2)*2)
        output2 = output2.permute(1,0,2)
        lens = lens/2
        #input: max_seq_len/4*batch*(256*4), output: max_seq_len/4*batch*(256*2)
        output3, _ = self.lstm3(pack_padded_sequence(output2, lens), None)
        #input: max_seq_len/4*batch*(256*2), output: max_seq_len/8*batch*(256*4)
        output3, lens = pad_packed_sequence(output3)
        output3 = output3.permute(1,0,2)
        output3 = output3.contiguous().view(output3.size(0), (int)(output3.size(1)/2), output3.size(2)*2)
        output3 = output3.permute(1,0,2)
        lens = lens/2
        key = self.key(output3)
        value = self.value(output3)       
        return key, value, lens

class Spell(nn.Module):
    def __init__(self):
        super(Spell,self).__init__()      
        self.embedding = nn.Embedding(num_embeddings=34, embedding_dim=attention_dim)
        
        self.LSTM1 = nn.LSTMCell(input_size=attention_dim*2, hidden_size=hidden_dim)
        self.LSTM2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=attention_dim)
        self.outputEmbedding = nn.Linear(2 * attention_dim,34)
        # self.transcriptsEmbedding = nn.Linear(34,attention_dim)

        self.relu = nn.ReLU()
        self.energySoftmax = nn.Softmax(dim=-1)

        self.hidden1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.cell1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.hidden2 = nn.Parameter(torch.zeros(batch_size, attention_dim))
        self.cell2 = nn.Parameter(torch.zeros(batch_size, attention_dim))

    def forward(self, key, value, lens, transcripts, transcripts_lens):
        """
        key: [maxLength x batch x featureSize]  
        value: [maxLength x batch x featureSize]
        lens: [batch] 
        transcripts: [batch x maxLength] 
        """
        #trainscripts_lens is fixed and we will mask output
        totalResult = []
        attentionScore = []

        maxTranscriptLength = transcripts[0].size(0)
        transcripts = torch.stack(transcripts, 0)

        context = Variable(torch.FloatTensor(batch_size, attention_dim).zero_()).cuda()
        key = key.transpose(1,0).transpose(1,2)
        value = value.transpose(1,0)
        finalOut = None
        hidden1, hidden2, cell1, cell2 = self.hidden1, self.hidden2, self.cell1, self.cell2


        for timeStamp in range(maxTranscriptLength):
            randomGenerate = np.random.randint(20, size=1)
            if randomGenerate[0] < 1 and type(finalOut) != type(None):   
                temp = torch.argmax(finalOut, dim=1).cuda()
                timeInputs = self.embedding(temp)
            else:
                feedTranscripts = transcripts[:,timeStamp]    
                timeInputs = self.embedding(feedTranscripts)

            # prev step context
            catImputs = torch.cat((timeInputs,context), dim=-1)
            # feed in
            hidden1, cell1 = self.LSTM1(catImputs, (hidden1,cell1))
            hidden2, cell2 = self.LSTM2(hidden1, (hidden2,cell2))
            # hidden2: [batch*1*attention_dim] [batch*attention_dim*seq]
            energy = torch.bmm(hidden2.unsqueeze(1), key)
            energy = energy.squeeze()
            # need to mask the energy
            mask = torch.zeros(batch_size, lens[0])
            for i in range(batch_size):
                mask[i][:lens[i]] = torch.ones(lens[i])
            mask = mask.to(DEVICE)
            energy = energy*mask
            alpha = self.energySoftmax(energy)
            # print(alpha)
            # print(mask)
            alpha = alpha * mask
            # alpha: [batch*seq]
            alpha = alpha/torch.sum(alpha, dim=-1, keepdim=True)
            attentionScore.append(alpha)
            context = torch.bmm(alpha.unsqueeze(1), value)
            context = context.squeeze()
            finalOut = self.outputEmbedding(torch.cat((hidden2,context), dim=-1))
            totalResult += [finalOut]
            # print(finalOut)
            # print(alpha)
            # print('--------------------------')
        totalResult = torch.stack(totalResult)
        attentionScore = torch.stack(attentionScore)

        return totalResult, attentionScore         

class LAS(nn.Module):
    def __init__(self):
        super(LAS,self).__init__() 
        self.listen = Listen()
        self.spell = Spell()
        # self.spell.load_state_dict(torch.load("SpellPreTrained2layer512featuresTeachForce"))
    def forward(self, inputs, inputs_lens, targets, targets_lens):
        key, value, lens = self.listen(inputs)        
        return self.spell(key, value, lens, targets, targets_lens)    
    def predict(self, inputs, inputs_lens):
        key, value, lens = self.listen(inputs, inputs_lens)        


wsjdataset = WSJDataset(data, label)
train_loader = DataLoader(wsjdataset, batch_size=batch_size, collate_fn=collate_lines, shuffle=True, drop_last=True)
las = LAS()
las = las.to(DEVICE)
# las.load_state_dict(torch.load("ModelSaturaday2th"))
optimizer = torch.optim.Adam(las.parameters(), lr=5e-4)
loss = nn.CrossEntropyLoss()
loss = loss.cuda()
EPOCH = 40

for epoch in range(EPOCH):
    lossVariable = 0
    for ith, (inputs, targets, inputs_lens, targets_lens) in enumerate(train_loader):
        inputTarget = [target[:-1] for target in targets]
        expectedTarget = [target[1:] for target in targets]
        targets_lens = targets_lens-1
        result, attentionScore = las(inputs, inputs_lens, inputTarget, targets_lens)
        # print(result)
        if ith%400 == 0:
            attentionScore = attentionScore.transpose(1,0)
            np.save("{}epoch{}thbatch.npy".format(epoch+1,ith+1), attentionScore[0].detach().cpu().numpy())
        result = result.permute(1,0,2)
        totalResult = torch.cat([result[i][:targets_lens[i]] for i in range(targets_lens.size(0))])
        expectedTarget = torch.cat([expectedTarget[i][:targets_lens[i]] for i in range(targets_lens.size(0))])
        
        cost = loss(totalResult, expectedTarget)
        lossVariable += cost.item()
        print('{}th batch loss:{} epoch loss{}'.format(ith+1, cost.item(), lossVariable/(ith+1)))
        cost.backward()
        torch.nn.utils.clip_grad_norm_(las.parameters(), 0.25)
        optimizer.step()
    print('<<<<<<{}th Epoch>>>>>>'.format(epoch))
    torch.save(las.state_dict(), "ModelMonday{}th".format(epoch+1))

