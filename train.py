import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet #神經網路模型
from nltk_utils import tokenize, stem, bag_of_words #語言處理工具

with open('intents.json','r') as f:
    intents_box = json.load(f)

all_words = []  #儲存所有文字
tags =[]    #儲存標籤
xy = []     #標籤和座標

for intent in intents_box['intents_list']:
    tag = intent['tag']
    tags.append(tag)

    for sentence in intent['patterns']:
        w = tokenize(sentence)
        all_words.extend(w)
        xy.append((w,tag)) #分類對應的斷詞和標籤

ignore_words = ['?','!','.',',','~','&',':',';',"%"]  #停用詞
all_words = [stem(w) for w in all_words if w not in ignore_words]

#分類&排序
all_words = sorted(set(all_words)) #sorted:按照大小寫排序  set:去掉重複的斷詞
tags = sorted(set(tags))


#-----Training Data-----#
X_train = [] #輸入: 詞袋向量
y_train = [] #輸出: 儲存tag(分類)

for (sentence, tag) in xy:
    bag = bag_of_words(sentence,all_words)  #bag就是詞袋向量
    X_train.append(bag)
    
    label = tags.index(tag)  #index:序號  lable: 產出tags的序號
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
#-----Training Data-----#

#--- Pytorch神經網路設定區域 ---#
#Hyperparameter(超參數)
batch_size = 8
input_size = len(X_train[0])
hidden_size = 6
output_size = len(tags)
learning_rate = 0.0005
num_epochs = 5000

#創建pytorch數據集
class ChatDataset(Dataset):
    #初始化函式
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    #用序號取得資料
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    #取得training set大小
    def __len__(self):
        return self.n_samples


#--- Pytorch神經網路訓練區域 ---#
def main():
    # 模型、數據集、硬體整合
    dataset = ChatDataset()
    train_loader  = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        for (sentence, tag) in train_loader:
            # 梯度歸零
            optimizer.zero_grad()
            
            sentence = sentence.to(device)
            tag = tag.to(dtype=torch.long).to(device)
            
            # 前向傳播(forward propagation)
            outputs = model(sentence)
            loss = criterion(outputs, tag)
            
            # 反向傳播(backward propagation)
            loss.backward()
            
            # 更新所有參數
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            
    print(f'final loss, loss={loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
        }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. File saved to {FILE}')
    #--- Pytorch神經網路訓練區域 ---#

if __name__=="__main__":
    main()