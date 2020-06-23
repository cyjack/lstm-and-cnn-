'''
对翻译得到的结果进行评价
'''
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# 删除英文中bpe格式
import os
import jieba
import nltk

def cut_word(line):
   x_cut = []
   for x in line:
      cut_out = jieba.lcut(x)  # 每一行文本进行分词
      x_ = " ".join(cut_out)
      print(x_)
      x_cut.append(x_)
   return x_cut


def cut_word_e(line):
   x_cut = []
   for x in line:
      cut_out = nltk.word_tokenize(x)  # 每一行文本进行分词
      x_ = " ".join(cut_out)
      print(x_)
      x_cut.append(x_)
   return x_cut


def eng_delete_bpe(sentence):
   dlbpe = []
   for s in sentence:
      dlbpe.append(s.replace('@@ ', ''))
   return dlbpe

#判断是不是中文
def is_Chinese(word):  # 判断是不是中文
   for ch in word:
      if '\u4e00' <= ch <= '\u9fff':
         return True
   return False

# 删除分字以及bpe格式
def zh_delete_bpe(sentence):
   dlbpe = []
   for s in sentence:
      s = s.replace('@@ ', '')
      dlbpe.append(s)
   return dlbpe

def read_file(src,tgt):
   f1 = open(src,'r',encoding='utf-8')
   f2 = open(tgt, 'r', encoding='utf-8')
   src =[]
   tgt = []
   for i in f1:
      src.append(i)
   for i in f2:
      tgt.append(i)
   return src,tgt

def bleu_score(src,tgt):
   smooth = SmoothingFunction()
   score_baidu = sentence_bleu(src, tgt,weights=(0.5, 0.5, 0, 0),smoothing_function=smooth.method1)
   return score_baidu

def accuracy(src,tgt):
   src_l = src.strip().split(' ')
   s = set()
   for i in src_l:
      s.add(i)
   l = len(s)
   tgt_l = tgt.split(' ')
   a = 0
   for i in tgt_l:
      if i in s:
         a+=1

   acc = float(a/l)
   # print(acc)
   return acc

#对bleu数据进行预处理处理成['I','love','you'] 是数据变为列表形式
def bleu_pre(src,tgt):
   src_c = []
   src = src.strip().split(' ')
   for i in src:
      src_c.append(i)
   tgt_c = []
   tgt = tgt.strip().split(' ')
   for i in tgt:
      tgt_c.append(i)
   return  src_c,tgt_c

# src = '她 的 工作 是 在 幼儿园 里 照看 儿童 。 '
# tgt = '她 的 工作 是 在 幼儿园 里 照看 儿童 。 '
# print(accuracy(src,tgt))
dir = 'D:\pytorch_test_01\\venv\OpenNMT-py-master\data\\test\\1500_c'
src = 'eng_test_bpe.txt'
tgt = 'pred_30w_c_zh2eng.txt'
#输出翻译出来的结果的准确率是由翻译出的内容除以标准答案中的内容。
def add_acc(dir,src,tgt):
   src, tgt = read_file(os.path.join(dir,src), os.path.join(dir,tgt))
   if is_Chinese(src):
      src = zh_delete_bpe(src)
      tgt = zh_delete_bpe(tgt)
      print('我是中文')
   else:
      src = eng_delete_bpe(src)
      tgt = eng_delete_bpe(tgt)
   a = 0
   for i, j in zip(src, tgt):
      # print(i)
      # print(j)
      a += accuracy(i, j)

   return(float(a / len(src)))

#输出翻译出来的结果的bleu值。
def bleu(dir,src,tgt):
   src, tgt = read_file(os.path.join(dir,src), os.path.join(dir,tgt))
   if is_Chinese(src):
      src = zh_delete_bpe(src)
      tgt = zh_delete_bpe(tgt)
   else:
      src = eng_delete_bpe(src)
      tgt = eng_delete_bpe(tgt)
   a = 0
   for i,j in zip(src,tgt):
      # print(i)
      i,j = bleu_pre(i,j)
      # print(i)
      # print(j)
      a += bleu_score([i],j)

   return float(a / len(src))


# print(add_acc(dir,src,tgt))
# print(bleu(dir,src,tgt))
