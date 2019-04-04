from keras.layers import (
    Dense, Input, Embedding, CuDNNGRU
)
from keras.layers import Bidirectional
from keras.models import Model
from keras import optimizers

from attentions import Attention
import configs as cf
from optimizers import AdaBound


def make_baseline_model():
    input_layer = Input(shape=(cf.MAXLEN,))
    embedding_layer = Embedding(
        len(cf.ALPHABETS) - 1, 128, trainable=True, input_length=cf.MAXLEN
    )(input_layer)
    rnn = Bidirectional(CuDNNGRU(128, return_sequences=False))(embedding_layer)
    output_layer = Dense(cf.MAXLEN, activation="sigmoid")(rnn)

    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam_optimizer, metrics=['accuracy'])
    return model


def make_attention_model():
    input_layer = Input(shape=(cf.MAXLEN,))
    embedding_layer = Embedding(
        len(cf.ALPHABETS) - 1, 128, trainable=True, input_length=cf.MAXLEN
    )(input_layer)
    rnn = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedding_layer)
    rnn = Attention(cf.MAXLEN)(rnn)
    output_layer = Dense(cf.MAXLEN, activation="sigmoid")(rnn)

    model = Model(inputs=input_layer, outputs=output_layer)

    # adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    optimizer = AdaBound(lr=1e-03,
                         final_lr=0.1,
                         gamma=1e-03,
                         weight_decay=0.,
                         amsbound=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = make_baseline_model()
    print(model.summary())
