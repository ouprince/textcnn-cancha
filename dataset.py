# -*- coding:utf-8 -*-
import re
from torchtext import data
import jieba
import jieba.posseg as pseg
import logging
jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

def word_cut(text):
    text = regex.sub(' ', text)
    return [o.word for o in pseg.cut(text) if o.word.strip() and not o.flag.startswith('x')]

def get_dataset(path,label_field, text_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='shuku.dev.data', validation='dev.data',
        fields=[
                ('label', label_field),
                ('title', text_field),
                ('content', text_field)]
        )
    return train, dev
