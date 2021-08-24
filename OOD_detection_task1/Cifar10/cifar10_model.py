import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD

from load_data import load_cifar10
import global_variables as gv


class CifarModel(object):
    def __init__(self, train_class_number, train_data, train_label, test_data, test_label, model_save_path,
                 validation_data=None, validation_label=None,):
        self.train_class_number = train_class_number
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.model_save_path = model_save_path
        self.validation_data = validation_data
        self.validation_label = validation_label

    def create_model(self):
        model = tf.keras.Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.train_data.shape[1:], kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(10, activation=None))
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train(self):
        model = self.create_model()
        # model.fit(self.train_data, self.train_label, epochs=10, verbose=2, shuffle=True, validation_split=0.2)
        model.fit(self.train_data, self.train_label, epochs=120, verbose=1,
                  validation_data=(self.test_data, self.test_label))
        model.save(self.model_save_path)
        print("save path:", self.model_save_path)

    def show_model(self):
        model = tf.keras.models.load_model(self.model_save_path)
        model.summary()
        print("train dataset:")
        print(self.train_data.shape, self.train_label.shape)
        model.evaluate(self.train_data, self.train_label, verbose=2)

        print("test dataset:")
        print(self.test_data.shape, self.test_label.shape)
        model.evaluate(self.test_data, self.test_label, verbose=2)


def train_models():
    if not gv.cifar10_data_is_loaded:
        gv.x_train, gv.y_train_category, gv.y_train_one_hot,\
          gv.x_test, gv.y_test_category, gv.y_test_one_hot = load_cifar10()
        gv.cifar10_data_is_loaded = True

    cifar10_model = CifarModel(train_class_number=gv.category_classified_of_model,
                               train_data=gv.x_train, train_label=gv.y_train_one_hot,
                               test_data=gv.x_test, test_label=gv.y_test_one_hot, model_save_path=gv.model_path)

    if os.path.exists(gv.model_path):
        print('{} is existed!'.format(gv.model_path))
        cifar10_model.show_model()
    else:
        cifar10_model.train()


if __name__ == "__main__":
    train_models()

