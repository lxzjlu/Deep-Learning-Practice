import torch
from torch import nn 
from torch.nn import functional as F
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator
import matplotlib.pyplot as plt
#超参数定义
batch_size = 32
learning_rate = 0.001
epochs = 5
dropout = 0.5

embedding_dim = 128
hidden_dim = 128
num_layers = 1

#网络选择
# rnn_type = 'lstm'
rnn_type = 'GRU'
#GPU选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

###########################
#数据处理

#定义单词处理方法
tokenize = lambda x: x.split()
text_field = Field(sequential=True, tokenize=tokenize, lower=True, batch_first = True)
label_field = LabelField(dtype = torch.float, batch_first = True)
#数据集导入
train_data, test_data = IMDB.splits(text_field, label_field, train='train', test='test', root='.data')
print("train data length: ", len(train_data))
print("test data length: ", len(test_data))
#构建字典
text_field.build_vocab(train_data)
label_field.build_vocab(train_data)
#输出训练集常见词汇
top_20_words = text_field.vocab.freqs.most_common(20)
print("Top 20 words:")
for word, freq in top_20_words:
    print('词汇:{}, 频率:{}'.format(word,freq))
#输出训练集总词汇个数
vocab_size = len(text_field.vocab)
print('训练集总词汇个数{}'.format(len(text_field.vocab)))
#构建iter迭代器
train_iter, test_iter = BucketIterator.splits(
                        (train_data, test_data),
                        batch_size=batch_size,
                        sort_within_batch=True,
                        sort_key=lambda x: len(x.text),  
                        device=device
)
#网络选择
if rnn_type == 'lstm':
    bidirectional = True
    rnn_layer = nn.LSTM(input_size=embedding_dim, 
                                   hidden_size=hidden_dim, 
                                   num_layers=num_layers, 
                                   bidirectional=bidirectional, 
                                   dropout=dropout, 
                                   batch_first=True)
    rnn_linear_layer = nn.Linear(hidden_dim * 2 , 1)
else:
    bidirectional = False
    rnn_layer = nn.GRU(input_size=embedding_dim, 
                                  hidden_size=hidden_dim, 
                                  num_layers=num_layers, 
                                  bidirectional=bidirectional, 
                                  dropout=dropout, 
                                  batch_first=True)  
    rnn_linear_layer = nn.Linear(hidden_dim , 1)
 
###############################    
#网络定义

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = rnn_layer
        self.linear = rnn_linear_layer 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        if self.rnn.bidirectional:
            x = torch.cat((x[:, :self.rnn.hidden_size], x[:, self.rnn.hidden_size:]), dim=1)
        x = self.linear(x)
        y = torch.sigmoid(x)
        return y


##############################
#模型训练
net = RNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_acc = 0.0
#存储损失与精度
loss_history = []
accuracy_history = []
loss_times_history = []
loss_history_test = []
accuracy_history_test = []
loss_times_history_test = []
for epoch in range(epochs):
    training_loss = 0.0
    training_acc = 0.0 
    net.train()
    for iter in train_iter:
        #获取输入
        inputs= iter.text
        labels = iter.label
        inputs = inputs.to(device)
        labels = labels.to(device)
        #梯度置零
        optimizer.zero_grad()
        #前向计算
        outputs = net(inputs).squeeze(1)
        #获取损失
        loss = criterion(outputs, labels)
        #计算梯度
        loss.backward()
        optimizer.step()
        #累计损失
        training_loss += loss.item()
        loss_times_history.append(loss.item())
        # 计算准确率
        training_acc += torch.sum(torch.round(outputs) == labels).item()
    #计算各epoch中的平均损失值和准确率
    avg_loss = training_loss / len(train_data)
    avg_accuracy = 100.0 * training_acc / len(train_data)   
    print('Epoch: {}     training loss: {:.4f}      training accuracy: {:.2f}'.format((epoch + 1), avg_loss, avg_accuracy))
    #存储平均损失值和准确率
    loss_history.append(avg_loss)
    accuracy_history.append(avg_accuracy)
    
    #测试精度
    test_accuary = 0
    test_loss = 0.0
    with torch.no_grad():
        for testdata in test_iter:
            #获取输入
            inputs= testdata.text
            labels = testdata.label
            inputs = inputs.to(device)
            labels = labels.to(device)
            #前向计算
            outputs = net(inputs).squeeze(1)
            #获取损失
            loss = criterion(outputs, labels)
            #累计损失
            test_loss += loss.item()
            loss_times_history_test.append(loss.item())
            #计算准确率
            test_accuary += torch.sum(torch.round(outputs) == labels).item()
            
    avg_test_accuracy = 100 * test_accuary/ len(test_data)
    avg_test_loss = test_loss / len(test_data)
    
    print('Epoch: {}     test     loss: {:.4f}      testing  accuracy: {:.2f}'.format((epoch + 1), avg_test_loss, avg_test_accuracy))    
    #存储平均损失值和准确率
    loss_history_test.append(avg_test_loss)
    accuracy_history_test.append(avg_test_accuracy)
    #存储模型
    if avg_test_accuracy > best_acc:
        best_acc= avg_test_accuracy
        torch.save(net.state_dict(), '/home/lxz/code_py/Deep-Learning-Practice_20230718/Practice3.pth')

print('Finished Training')


#画图
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), loss_history)
plt.plot(range(epochs), loss_history_test)
# plt.legend(labels=["train","test"],loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), accuracy_history)
plt.plot(range(epochs), accuracy_history_test)
# plt.legend(label=["train","test"],loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()


#各类别分类对应精度
positive_correct = 0
positive_total = 0
negative_correct = 0
negative_total = 0
#禁用grad
with torch.no_grad():
    for testdata in test_iter:
        #数据输入
        inputs= testdata.text
        labels = testdata.label
        inputs = inputs.to(device)
        labels = labels.to(device)
        #正向计算
        outputs = net(inputs).squeeze(1)
        #计算各类正确数 和 各类总数
        for label, output in zip(labels, outputs):
            if int(label.item())==0 and int(torch.round(output))==0:
                positive_correct += 1
            if int(label.item())==1 and int(torch.round(output))==1:
                negative_correct += 1

        positive_total += labels.eq(0).sum().item()
        negative_total += labels.eq(1).sum().item()

#输出各类精度
accuracy_positive = 100 * positive_correct / positive_total
accuracy_negative = 100 * negative_correct / negative_total

print('Accuracy for positive is {:.2f} %'.format(accuracy_positive))
print('Accuracy for negative is {:.2f} %'.format(accuracy_negative))






















