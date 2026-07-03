import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Subtract, Concatenate, Activation, Layer, Conv2D, MaxPooling2D, BatchNormalization, Subtract, Activation
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.backend import abs
import tensorflow.keras.backend as K

from models.ordinal import add_coral_head

tensorflow.random.set_seed(1234)

class snn:
    def __init__(self, num_of_class, input_shape, ordinal=False):
        self.n_class = num_of_class
        self.input_shape = input_shape
        self.ordinal = ordinal


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

        fc = Dense(128, activation='relu')(merged)
        fc = Dropout(0.5)(fc)  # Added dropout layer

        if self.ordinal:
            # CORAL ordinal head (see models/ordinal.py) - requires n_class >= 3.
            outputs = add_coral_head(fc, self.n_class)
        else:
            # Binary classification is represented as either num_of_class == 1
            # (this repo's convention) or == 2 (the original Fujita convention,
            # i.e. two classes rather than one output unit) - both must produce
            # a single sigmoid unit, never a softmax over 1 or 2 dense units.
            if self.n_class in (1, 2):
                actv = "sigmoid"
                units = 1
            else:
                actv = "softmax"
                units = self.n_class
            outputs = Dense(units, activation=actv)(fc)

        model = Model(inputs=[img_a, img_b], outputs=outputs)

        return model


if __name__ == "__main__":
    for num_of_class in (1, 2, 5):
        model = snn(num_of_class, (16, 16, 1)).get_model()
        units = model.output_shape[-1]
        actv = model.layers[-1].activation.__name__
        expected_units, expected_actv = (
            (1, "sigmoid") if num_of_class in (1, 2) else (num_of_class, "softmax")
        )
        assert (units, actv) == (expected_units, expected_actv), (num_of_class, units, actv)
    print("fujita.snn self-check OK: num_of_class in {1, 2} -> sigmoid(1), else -> softmax(n)")

    ordinal_model = snn(5, (16, 16, 1), ordinal=True).get_model()
    assert ordinal_model.output_shape[-1] == 4, ordinal_model.output_shape  # K-1 thresholds
    assert ordinal_model.layers[-1].__class__.__name__ == "CoralBiases", ordinal_model.layers[-1]
    print("fujita.snn ordinal self-check OK: num_of_class=5, ordinal=True -> CoralBiases(4 thresholds)")
