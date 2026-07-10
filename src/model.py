import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    LSTM,
    Dense
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from src.config import *

def build_model():

    inputs = Input(shape=(TIME_STEPS, INPUT_FEATURES))
    x = Conv1D(
        filters=CNN_FILTERS_1,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(POOL_SIZE)(x)

    x = Dropout(DROPOUT_RATE)(x)

    x = Conv1D(
        filters=CNN_FILTERS_2,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(POOL_SIZE)(x)

    x = Dropout(DROPOUT_RATE)(x)

    x = Bidirectional(
        LSTM(
            LSTM_UNITS
        )
    )(x)

    x = Dropout(0.4)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model