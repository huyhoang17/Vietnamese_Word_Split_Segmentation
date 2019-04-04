import itertools

import numpy as np

import configs as cf
from utils import preprocessing_text


class PaddingSequenceGenerator:
    def __init__(self, iter_seqs, path, count):
        self.iter_seqs = iter_seqs
        self.path = path
        self.count = count

    def __iter__(self):
        for sample in self.iter_seqs:
            yield sample

    def __len__(self):
        # count = get_count(self.path)
        return self.count


def input_pad_sequences(phrases):
    for phrases in phrases:
        encode = "1" + "0" * (cf.MAXLEN - 1)
        yield encode


def label_pad_sequences(phrases, text=False):
    for phrase in phrases:
        encode = "".join(
            list(
                map(lambda x: "1" if x == " " else "0", (" " + phrase.strip()))
            )
        )
        encode = [
            int(en) for en in list("".join(
                ["1" + e[1:] for e in encode.split('1') if e])
            )
        ]
        encode = encode + list(
            [int(z) for z in "0" * (cf.MAXLEN - len(encode))])
        encode = encode[:cf.MAXLEN]
        if text:
            yield phrase.replace(" ", ""), encode
        else:
            yield encode


def gen_lines(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield line


def gen_phrases(path=None, non_space=False, label=False):
    common_phrases = (
        p.lower().strip()
        for p in itertools.chain.from_iterable(
            preprocessing_text(line) for line in open(path)
        )
        if len(p.lower().strip().split()) > 1
    )
    if non_space:
        return (
            (p.replace(" ", ""), label)
            if label else p.replace(" ", "")
            for p in common_phrases
        )
    else:
        return common_phrases


def gen_no_space_phrases(phrases, label=True):
    for phrase in phrases:
        # gen X, y
        if label:
            yield phrase.replace(" ", ""), phrase
        else:
            yield phrase.replace(" ", "")


def gen_tokenizer(tokenizer, path=None):
    for sample in tokenizer.texts_to_sequences(gen_phrases(path, True, False)):
        yield sample


def get_count(path):
    train = gen_tokenizer(path)
    count = 0
    for sample in train:
        count += 1
    return count


def custom_pad_sequences(sequences, maxlen):
    """https://github.com/keras-team/keras/issues/7894
    """
    for s in sequences:
        if len(s) == 0:
            yield np.zeros((1, maxlen), dtype='int32')
        else:
            s = np.array(s, dtype='int32')
            yield np.expand_dims(
                np.pad(s[:maxlen], (0, max(0, maxlen - len(s))), 'constant'),
                axis=0
            )[0]


def make_samples(X, y, bs=128):
    X_samples, y_samples = [], []
    while True:
        for X_sample, y_sample in zip(X, y):
            X_samples.append(X_sample)
            y_samples.append(y_sample)
            if len(X_samples) == bs:
                yield np.array(X_samples), np.array(y_samples)
                X_samples, y_samples = [], []
        if X_samples:
            yield np.array(X_samples), np.array(y_samples)
