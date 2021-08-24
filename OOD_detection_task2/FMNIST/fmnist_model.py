import tensorflow as tf
import os

from load_data import load_fmnist_of_given_category
import global_variables as gv


class FMnistModel(object):
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

        model.add(tf.keras.layers.Convolution2D(40, (5, 5), strides=(1, 1),
                                                input_shape=self.train_data.shape[1:], activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Convolution2D(20, (5, 5), strides=(1, 1), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(320, activation='relu'))
        model.add(tf.keras.layers.Dense(160, activation='relu'))
        model.add(tf.keras.layers.Dense(80, activation='relu'))
        model.add(tf.keras.layers.Dense(40, activation='relu'))
        model.add(tf.keras.layers.Dense(self.train_class_number, activation='softmax'))

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self):
        model = self.create_model()
        # model.fit(self.train_data, self.train_label, epochs=10, verbose=2, shuffle=True, validation_split=0.2)
        model.fit(self.train_data, self.train_label, epochs=20, verbose=2)
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


def train_models(category_of_train):

    gv.x_train, gv.y_train_category, gv.y_train_one_hot,\
      gv.x_test, gv.y_test_category, gv.y_test_one_hot = \
      load_fmnist_of_given_category(category_of_train)

    mnist_model = FMnistModel(train_class_number=gv.category_classified_of_model,
                              train_data=gv.x_train, train_label=gv.y_train_one_hot,
                              test_data=gv.x_test, test_label=gv.y_test_one_hot, model_save_path=gv.model_path)

    if os.path.exists(gv.model_path):
        print('{} is existed!'.format(gv.model_path))
        mnist_model.show_model()
    else:
        mnist_model.train()


if __name__ == "__main__":
    for category_number in range(2, 10):
        gv.category_classified_of_model = category_number
        gv.model_path = 'models/fmnist_model_0-{}.h5'.format(category_number-1)
        train_models(category_number)
