from tensorflow import keras
from time import time
import keras as keras
from keras import layers
from keras.callbacks import TensorBoard, ReduceLROnPlateau


class LeNet5:
    def __init__(self, BATCH_SIZE, EPOCHS, ACTIVATION_FUNCTION):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION
        self.model = self.__init_model()

    def __init_model(self):
        model = keras.Sequential()
        model.add(
            layers.Conv2D(filters=6, kernel_size=(3, 3), activation=self.ACTIVATION_FUNCTION, input_shape=(32, 32, 1)))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation=self.ACTIVATION_FUNCTION))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation=self.ACTIVATION_FUNCTION))
        model.add(layers.Dense(units=84, activation=self.ACTIVATION_FUNCTION))
        model.add(layers.Dense(units=12, activation='softmax'))
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
            f"../models/model=LeNet-5,epoch={self.EPOCHS},batch={self.BATCH_SIZE},activation={self.ACTIVATION_FUNCTION}.h5")

        return training_loss, validation_loss, training_accuracy, validation_accuracy

    def test(self, X_test, y_test):
        Y_test = self.model.predict(X_test)
        return Y_test

    def summary(self):
        print(self.model.summary())