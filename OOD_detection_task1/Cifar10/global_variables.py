
model_path = 'models/cifar10_model.h5'
detector_path = 'detector/'

category_classified_of_model = 10

cifar10_data_is_loaded = False
x_train = None
y_train_category = None
y_train_one_hot = None
x_test = None
y_test_category = None
y_test_one_hot = None

fully_connected_layers = [-5]
fully_connected_layers_number = 1
neuron_number_of_fully_connected_layers = [512]

selective_rate = 0.20
