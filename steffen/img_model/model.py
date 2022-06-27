import numpy as np
import tensorflow as tf
from utils_model import fix_layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    VGG16, MobileNetV2, NASNetMobile, ResNet50
)
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    concatenate,
    Input,
    BatchNormalization,
    ReLU,
    LeakyReLU
)


def get_model(
    input_size: tuple = (224, 224, 3),
    hidden_size: int = 64,
    dropout: float = 0.5,
    num_classes: int = 4,
    trainable_layers: int = 1,
    mc_dropout: bool = False,
    **kwargs
):
    act_fn = tf.nn.sigmoid

    baseModel = VGG16(
        input_shape = input_size,
        include_top = False,
        weights = 'imagenet'
    )

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(hidden_size)(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = ReLU()(headModel)
    headModel = (
        Dropout(dropout)(headModel, training=True)
        if mc_dropout else Dropout(dropout)(headModel)
    )
    headModel = Dense(num_classes, activation=act_fn)(headModel)

    # place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    model = fix_layers(model, num_flex_layers=trainable_layers + 8)

    return model


        
