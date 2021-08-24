
model_path = 'models/mnist_model.h5'
detector_path = 'detector/'

category_classified_of_model = 10

mnist_data_is_loaded = False
x_train = None
y_train_category = None
y_train_one_hot = None
x_test = None
y_test_category = None
y_test_one_hot = None

fully_connected_layers = [-10, -8, -6, -4]
fully_connected_layers_number = 4
neuron_number_of_fully_connected_layers = [320, 160, 80, 40]

selective_rate = 0.20
