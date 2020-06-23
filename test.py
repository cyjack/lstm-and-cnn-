# coding: utf-8

from __future__ import print_function

import os
import tensorflow.contrib.keras as kr
import torch
from torch import nn
from cnews_loader import read_category, read_vocab
from model import TextRNN
from torch.autograd import Variable
import numpy as np

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Conv1d(64, 256, 5)
        self.f1 = nn.Sequential(nn.Linear(152576, 128),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 10),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x = x.detach().numpy()
        x = np.transpose(x, [0, 2, 1])
        x = torch.Tensor(x)
        x = Variable(x)
        x = self.conv(x)
        x = x.view(-1, 152576)
        x = self.f1(x)
        return self.f2(x)


class CnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextCNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


class RnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextRNN()
        self.model.load_state_dict(torch.load('model_rnn_params.pkl'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


if __name__ == '__main__':
    model = CnnModel()
    # model = RnnModel()
    test_demo = [
        '湖人助教力助科比恢复手感 他也是阿泰的精神导师新浪体育讯记者戴高乐报道  上赛季，科比的右手食指遭遇重创，他的投篮手感也因此大受影响。不过很快科比就调整了自己的投篮手型，并通过这一方式让自己的投篮命中率回升。而在这科比背后，有一位特别助教对科比帮助很大，他就是查克·珀森。珀森上赛季担任湖人的特别助教，除了帮助科比调整投篮手型之外，他的另一个重要任务就是担任阿泰的精神导师。来到湖人队之后，阿泰收敛起了暴躁的脾气，成为湖人夺冠路上不可或缺的一员，珀森的“心灵按摩”功不可没。经历了上赛季的成功之后，珀森本赛季被“升职”成为湖人队的全职助教，每场比赛，他都会坐在球场边，帮助禅师杰克逊一起指挥湖人球员在场上拼杀。对于珀森的工作，禅师非常欣赏，“查克非常善于分析问题，”菲尔·杰克逊说，“他总是在寻找问题的答案，同时也在找造成这一问题的原因，这是我们都非常乐于看到的。我会在平时把防守中出现的一些问题交给他，然后他会通过组织球员练习找到解决的办法。他在球员时代曾是一名很好的外线投手，不过现在他与内线球员的配合也相当不错。',
        '弗老大被裁美国媒体看热闹“特权”在中国像蠢蛋弗老大要走了。虽然他只在首钢男篮效力了13天，而且表现毫无亮点，大大地让球迷和俱乐部失望了，但就像中国人常说的“好聚好散”，队友还是友好地与他告别，俱乐部与他和平分手，球迷还请他留下了在北京的最后一次签名。相比之下，弗老大的同胞美国人却没那么“宽容”。他们嘲讽这位NBA前巨星的英雄迟暮，批评他在CBA的业余表现，还惊讶于中国人的“大方”。今天，北京首钢俱乐部将与弗朗西斯继续商讨解约一事。从昨日的进展来看，双方可以做到“买卖不成人意在”，但回到美国后，恐怕等待弗朗西斯的就没有这么轻松的环境了。进展@北京昨日与队友告别  最后一次为球迷签名弗朗西斯在13天里为首钢队打了4场比赛，3场的得分为0，只有一场得了2分。昨天是他来到北京的第14天，虽然他与首钢还未正式解约，但双方都明白“缘分已尽”。下午，弗朗西斯来到首钢俱乐部与队友们告别。弗朗西斯走到队友身边，依次与他们握手拥抱。“你们都对我很好，安排的条件也很好，我很喜欢这支球队，想融入你们，但我现在真的很不适应。希望你们']
    for i in test_demo:
        print(i, ":", model.predict(i))