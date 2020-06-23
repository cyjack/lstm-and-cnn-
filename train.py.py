# coding: utf-8

from __future__ import print_function
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import os

import numpy as np

from model import TextRNN, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = '../data'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')


def train():
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, 600)  # 获取训练数据每个字的id和对应标签的one-hot形式
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, 600)
    # 使用LSTM或者CNN
    model = TextRNN()
    # model = TextCNN()
    # 选择损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    # Loss = nn.BCELoss()
    # Loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0
    for epoch in range(1000):
        batch_train = batch_iter(x_train, y_train, 100)
        for x_batch, y_batch in batch_train:
            x = np.array(x_batch)
            y = np.array(y_batch)
            x = torch.LongTensor(x)
            y = torch.Tensor(y)
            # y = torch.LongTensor(y)
            x = Variable(x)
            y = Variable(y)
            out = model(x)
            loss = Loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
        # 对模型进行验证
        if (epoch + 1) % 20 == 0:
            batch_val = batch_iter(x_val, y_val, 100)
            for x_batch, y_batch in batch_train:
                x = np.array(x_batch)
                y = np.array(y_batch)
                x = torch.LongTensor(x)
                y = torch.Tensor(y)
                # y = torch.LongTensor(y)
                x = Variable(x)
                y = Variable(y)
                out = model(x)
                loss = Loss(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
                if accracy > best_val_acc:
                    torch.save(model.state_dict(), 'model_params.pkl')
                    best_val_acc = accracy
                print(accracy)


if __name__ == '__main__':
    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()
    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_dir)
    # 获取字数
    vocab_size = len(words)
    train()
