from tensorflow import keras
from time import time
import keras as keras
from keras import layers
from keras.callbacks import TensorBoard, ReduceLROnPlateau


class AlexNet:
    def __init__(self, BATCH_SIZE, EPOCHS, ACTIVATION_FUNCTION, CLASS_COUNT):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION
        self.CLASS_COUNT = CLASS_COUNT
        self.model = self.__init_model()

    def __init_model(self):
        model = keras.Sequential()

        model.add(layers.Conv2D(96, (11, 11), input_shape=(32, 32, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        model.add(layers.Conv2D(256, (5, 5), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        model.add(layers.ZeroPadding2D((1, 1)))
        model.add(layers.Conv2D(512, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        model.add(layers.ZeroPadding2D((1, 1)))
        model.add(layers.Conv2D(1024, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))

        # Layer 5
        model.add(layers.ZeroPadding2D((1, 1)))
        model.add(layers.Conv2D(1024, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        model.add(layers.Flatten())
        model.add(layers.Dense(4096))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.Dropout(0.5))

        # Layer 7
        model.add(layers.Dense(4096))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.ACTIVATION_FUNCTION))
        model.add(layers.Dropout(0.5))

        # Layer 8
        model.add(layers.Dense(self.CLASS_COUNT))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, X_validation, y_validation):
        # Set a learning rate annealer
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                      patience=3,
                                      verbose=1,
                                      factor=0.2,
                                      min_lr=1e-6)

        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_validation, y_validation),
                                 epochs=self.EPOCHS,
                                 batch_size=self.BATCH_SIZE,
                                 callbacks=[reduce_lr])

        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']

        self.model.save(
            f"../models/model=AlexNet,epoch={self.EPOCHS},batch={self.BATCH_SIZE},activation={self.ACTIVATION_FUNCTION}.h5")

        return training_loss, validation_loss, training_accuracy, validation_accuracy

    def test(self, X_test, y_test):
        Y_test = self.model.predict(X_test)
        return Y_test

    def summary(self):
        print(self.model.summary())
