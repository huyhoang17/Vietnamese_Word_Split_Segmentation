from itertools import tee
import pickle

import configs as cf
from generators import (
    PaddingSequenceGenerator,
    custom_pad_sequences,
    label_pad_sequences,
    gen_tokenizer,
    get_count,
    gen_phrases,
    make_samples
)
from models import make_baseline_model


def main():
    BATCH_SIZE = 512
    EPOCHS = 10
    # train_generator = make_samples(X_train, y_train, bs=BATCH_SIZE)
    # validation_generator = make_samples(X_test, y_test, bs=BATCH_SIZE)

    # get no samples
    len_train = get_count("../models/train.txt")
    len_test = get_count("../models/test.txt")

    # make data generator
    # its took me few days thinking how to avoid OOM when loading huge dataset
    # :D
    train_padding = PaddingSequenceGenerator(
        gen_tokenizer("../data/train.txt"), "../data/train.txt", len_train)
    train_data = custom_pad_sequences(train_padding, cf.MAXLEN)
    train_label = label_pad_sequences(gen_phrases("train.txt"))
    test_padding = PaddingSequenceGenerator(
        gen_tokenizer("../data/test.txt"), "../data/test.txt", len_test)
    test_data = custom_pad_sequences(test_padding, cf.MAXLEN)
    test_label = label_pad_sequences(gen_phrases("test.txt"))

    # make generator copy
    train_data, train_sub = tee(train_data)
    test_data, test_sub = tee(test_data)

    for i in range(EPOCHS):
        train_generator = make_samples(train_sub, train_label)
        validation_generator = make_samples(test_sub, test_label)
        hist = model.fit_generator(
            train_generator,
            samples_per_epoch=len_train // BATCH_SIZE,
            nb_epoch=1,
            validation_data=validation_generator,
            nb_val_samples=len_test // BATCH_SIZE
        )

    with open("../models/hist.pk", "pk") as f:
        pickle.dump(hist, f)

    model.save_weights("../models/word_seg_vnese_{}epochs.h5".format(EPOCHS))
    with open("../models/config.json", "w") as f:
        f.write(model.to_json())


if __name__ == '__main__':
    model = make_baseline_model()
    main()
