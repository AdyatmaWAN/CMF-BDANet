import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Dropout, Subtract, Concatenate, Activation, Layer, Conv2D, MaxPooling2D, BatchNormalization, Subtract, Activation
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.backend import abs
import tensorflow.keras.backend as K


tensorflow.random.set_seed(1234)

class snn:
    def __init__(self, num_of_class, residual, dropout, dense, num_of_layer, input_shape, substraction, shared):
        self.n_class = num_of_class
        self.is_residual = residual
        self.is_dropout = dropout
        self.is_dense = dense
        self.num_of_layer = num_of_layer
        self.input_shape = input_shape
        self.substraction = substraction
        self.shared = shared

    def __build_siamese_model(self):
        inputs =Input(self.input_shape)
        input = inputs
        feats = []
        convs = [32, 64, 128]

        for conv in convs:
            x = input

            for i in range(0, self.num_of_layer):
                x = Conv2D(conv, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)
            if self.is_dropout:
                x = Dropout(0.5)(x)
            if self.is_residual:
                feats.append(Flatten()(x))
            input = x

        x = Flatten()(input)

        if self.is_dense:
            for i in range(0, self.num_of_layer):
                x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            if self.is_dropout:
                x = Dropout(0.5)(x)

        if self.is_residual:
            output = Concatenate()(feats)
            if self.is_dense:
                output = Concatenate()([output, x])
        else:
            output = x
        outputs = output
        return Model(inputs, outputs)

    def get_model(self):
        img_a = Input(self.input_shape)
        img_b = Input(self.input_shape)

        if self.shared:
            model = self.__build_siamese_model()
            feat_a = model(img_a)
            feat_b = model(img_b)
        else:
            model_a = self.__build_siamese_model()
            feat_a = model_a(img_a)
            model_b = self.__build_siamese_model()
            feat_b = model_b(img_b)

        if self.substraction:
            distance = Subtract()([feat_a, feat_b])
            distance = K.abs(distance)
            # distance = Lambda(lambda x: K.abs(x))(distance)

        else:
            distance = Concatenate()([feat_a, feat_b])

        if(self.n_class == 2):
            actv = "sigmoid"
            self.n_class = 1
        else:
            actv = "softmax"

        outputs = Dense(self.n_class, activation=actv, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(distance)
        model = Model(inputs=[img_a, img_b], outputs=outputs)

        return model

if __name__ == "__main__":
    snn1 = snn(5, True, True, True, 3, (16, 16, 1), False, False)
    model = snn1.get_model()
    model.summary()
    snn2 = snn(5, True, True, True, 3, (16, 16, 1), False, True)
    model2 = snn2.get_model()
    model2.summary()
