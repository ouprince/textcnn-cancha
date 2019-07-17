# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        
        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze = not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None

        self.convs1 = nn.ModuleList(
                          [nn.Sequential(nn.Conv2d(chanel_num, int(filter_num/2), (size, embedding_dimension)),nn.BatchNorm2d(int(filter_num/2))) for size in filter_sizes])
        
        self.convs2 = nn.ModuleList(
                          [nn.Sequential(nn.Conv2d(int(filter_num/2), filter_num, (size, 1), padding = (int((size - 1)/2),0)),nn.BatchNorm2d(filter_num)) for size in filter_sizes])

        self.downsamples = nn.ModuleList(
                          [nn.Sequential(nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)),nn.BatchNorm2d(filter_num)) for size in filter_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)


    def forward(self,x):# 输入是 [batch,seq] 形式的数据
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1) # 这样就变成了 4 维 [batchsize,2,seq,embedding]
        else:
            x = self.embedding(x) # [batchsize,seq,embedding]
            x = x.unsqueeze(1)  # 增加一维，变成 [batchsize,1,seq,embedding]
        ''' 这里的卷积根本不同于 TensorFlow 的卷积，这里的卷积是以第二维为深度
            然后第三四维为长和宽'''
        downsamples = [conv(x) for conv in self.downsamples]
        x = [nn.ReLU(inplace = True)(conv(x)) for conv in self.convs1] # 每个元素为[batchsize,filter_num,(seq - filter_num + 1), 1]
        x = [nn.ReLU(inplace = True)(conv(i) + downsample).squeeze(3) for conv,downsample,i in zip(self.convs2,downsamples,x)] # 每个元素为[batchsize,filter_num,(seq - filter_size + 1) * 1]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x] # 每个元素为[batchsize,filter_num * 1]
        x = torch.cat(x, 1) # [batchsize,filter_num * len(filter_sizes) * 1]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
