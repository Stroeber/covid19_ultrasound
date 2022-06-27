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
    Concatenate,
    Input,
    BatchNormalization,
    ReLU,
    LeakyReLU
)



class MultimodalModel(tf.keras.Model):

    def __init__(self, num_classes = 4, dropout=0.2, img_feat_size = 16, input_size = (224, 224, 3)):
        super(MultimodalModel, self).__init__()

        self.CNNModel = VGG16(
            input_shape = input_size,
            include_top = False,
            weights = 'imagenet'
        )
        self.CNNModel.trainable = False
        # self.avg_pool_2d = AveragePooling2D(pool_size=(4, 4))
        self.flatten = Flatten()
        self.dense_block1 = [
            Dense(img_feat_size),
            BatchNormalization(),
            LeakyReLU()
        ]
        self.dropout1 = Dropout(dropout)
        self.sym_dense = Dense(img_feat_size)
        self.concat = Concatenate()
        self.dense_block2 = [
            Dense(img_feat_size),
            BatchNormalization(),
            LeakyReLU()
        ]
        self.dropout2 = Dropout(dropout)
        self.output_layer = Dense(num_classes, activation=tf.nn.sigmoid)

    def call(self, input):
        x = self.CNNModel(input['images'])
        x = self.flatten(x)
        for layer in self.dense_block1:                     #Ist ausschreiben schneller als for loop?
            x = layer(x)
        x = self.dropout1(x)
        symptoms = self.sym_dense(input['symptoms'])
        x = self.concat([x, symptoms])

        for layer in self.dense_block2:
            x = layer(x)
        x = self.dropout2(x)
        model_output = self.output_layer(x)
        return model_output