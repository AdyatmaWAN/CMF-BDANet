import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Subtract, Concatenate, Activation, Layer, Conv2D, MaxPooling2D, BatchNormalization, Subtract, Activation
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.backend import abs
import tensorflow.keras.backend as K

tensorflow.random.set_seed(1234)

class snn:
    def __init__(self, num_of_class, input_shape):
        self.n_class = num_of_class
        self.input_shape = input_shape


    def __build_siamese_model(self):
        inputs =Input(self.input_shape)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)

        return Model(inputs, x)

    def get_model(self):
        img_a = Input(self.input_shape)
        img_b = Input(self.input_shape)

        model_a = self.__build_siamese_model()
        feat_a = model_a(img_a)
        model_b = self.__build_siamese_model()
        feat_b = model_b(img_b)

        merged = Concatenate()([feat_a, feat_b])

        if(self.n_class == 2):
            actv = "sigmoid"
            self.n_class = 1
        else:
            actv = "softmax"

        fc = Dense(128, activation='relu')(merged)
        fc = Dropout(0.5)(fc)  # Added dropout layer
        outputs = Dense(self.n_class, activation=actv)(fc)
        model = Model(inputs=[img_a, img_b], outputs=outputs)

        return model

if __name__ == "__main__":
    snn = snn(2, True, True, True, 3, (16, 16, 1), False, False)
    model = snn.get_model()
    model.summary()