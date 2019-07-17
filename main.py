# -*- coding:utf-8 -*-
import sys,os
import argparse
import torch
import torchtext.data as data
from torchtext.vocab import Vectors

import model
import train
import dataset
from logger import logger_console

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.005]')
parser.add_argument('-epochs', type=int, default=2000, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=256, help='batch size for training [default: 256]')
parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=200,
                        help='how many steps to wait before testing [default: 200]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# model
# parser.add_argument('-fix-length',type=int,default=200,help='the length of text')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,5,7',
                        help='comma-separated filter sizes to use for convolution')
parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
                        help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors

def load_dataset(label_field, text_field, args, **kwargs):
    train_dataset, dev_dataset = dataset.get_dataset('data', label_field, text_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_dataset, dev_dataset),
                                batch_sizes=(args.batch_size, args.batch_size),
                                sort_key=lambda x: len(x.title+x.content),
                                **kwargs)
    return train_iter, dev_iter

logger_console.info('Loading data...')
text_field = data.Field(lower=True,batch_first  = True,stop_words = None) # 可根据需要设置 stop_words
label_field = data.Field(sequential=False)
''' for x in train_iter  x 是 128 个batch, [batchsize,seq] ''' 
train_iter, dev_iter = load_dataset(label_field, text_field, args, repeat=False, shuffle=True)
args.vocabulary_size = len(text_field.vocab)

# save text_field,label_field.vocab.itos
torch.save(text_field,os.path.join(args.save_dir,'textfield.pkl'))
torch.save(label_field.vocab.itos,os.path.join(args.save_dir,'labelfield.pkl'))

if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
logger_console.info("All labels is \033[5;33m[{}]\033[0m".format(" ".join(label_field.vocab.itos)))

args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

logger_console.info('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue# vectors 就不要显示了
    logger_console.info('\t{} = {}'.format(attr.upper(), value))

text_cnn = model.TextCNN(args)
if args.snapshot and os.path.exists(args.snapshot):
    logger_console.info('\nLoading model from {}...\n'.format(args.snapshot))
    text_cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()

# 保存 args
torch.save(args, os.path.join(args.save_dir,'args.pkl'))

try:
    train.train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    logger_console.info('Exiting from training early')
