import argparse

import pandas as pd
from kerastuner import HyperModel, BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, \
    AveragePooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam


class HyperDetectionModel(HyperModel):
    def __init__(self):
        super().__init__()

    def build(self, hp):
        model = Sequential()

        model.add(
            Conv1D(filters=hp.Int('conv_filters_0', min_value=2, max_value=90),
                   kernel_size=hp.Int('kernel_0', min_value=1, max_value=24),
                   strides=hp.Int('strides_0', min_value=1, max_value=4),
                   padding=hp.Choice('padding_0', ['valid', 'same']),
                   activation='relu', input_shape=(24, 1), name='conv1d_0'))

        # hyperparameter for whether to include or not a Pooling layer
        if hp.Boolean('pooling_0', default=True):
            # check the input shape of this layer to decide the limits of
            # the hyperparameters
            input_size_pooling0 = model.get_layer('conv1d_0').output.shape[1]
            # hyperparameter for choosing the type of pooling (max or average)
            if hp.Boolean('max_pooling_0', default=True):
                model.add(MaxPooling1D(pool_size=hp.Int('max_pool_size_0',
                                                        min_value=min(2,
                                                                      input_size_pooling0),
                                                        max_value=min(5,
                                                                      input_size_pooling0),
                                                        parent_name='max_pooling_0',
                                                        parent_values=[True]),
                                       strides=hp.Int('max_pool_strides_0',
                                                      min_value=1,
                                                      max_value=min(2,
                                                                    input_size_pooling0),
                                                      parent_name='max_pooling_0',
                                                      parent_values=[True]),
                                       padding=hp.Choice('max_pool_padding_0',
                                                         ['valid', 'same'],
                                                         parent_name='max_pooling_0',
                                                         parent_values=[True]),
                                       name='max_pooling1d_0'))
            else:
                model.add(AveragePooling1D(pool_size=hp.Int('avg_pool_size_0',
                                                            min_value=min(2,
                                                                          input_size_pooling0),
                                                            max_value=min(5,
                                                                          input_size_pooling0),
                                                            parent_name='max_pooling_0',
                                                            parent_values=[
                                                                False]),
                                           strides=hp.Int('avg_pool_strides_0',
                                                          min_value=1,
                                                          max_value=min(2,
                                                                        input_size_pooling0),
                                                          parent_name='max_pooling_0',
                                                          parent_values=[
                                                              False]),
                                           padding=hp.Choice(
                                               'avg_pool_padding_0',
                                               ['valid', 'same'],
                                               parent_name='max_pooling_0',
                                               parent_values=[False]),
                                           name='avg_pooling1d_0'))

        # hyperparameter for whether to add a second Conv layer
        if hp.Boolean('second_conv_layer', default=True):
            # check which is the previous layer in order to get the input
            # size for the current layer
            if hp.get('pooling_0'):
                if hp.get('max_pooling_0'):
                    input_size_conv1 = \
                        model.get_layer('max_pooling1d_0').output.shape[1]
                else:
                    input_size_conv1 = \
                        model.get_layer('avg_pooling1d_0').output.shape[1]
            else:
                input_size_conv1 = model.get_layer('conv1d_0').output.shape[1]

            model.add(Conv1D(
                filters=hp.Int('conv_filters_1', min_value=2, max_value=90,
                               parent_name='second_conv_layer',
                               parent_values=[True]),
                kernel_size=hp.Int('kernel_1', min_value=1,
                                   max_value=min(8, input_size_conv1),
                                   parent_name='second_conv_layer',
                                   parent_values=[True]),
                strides=hp.Int('strides_1', min_value=1,
                               max_value=min(4, input_size_conv1),
                               parent_name='second_conv_layer',
                               parent_values=[True]),
                padding=hp.Choice('padding_1', ['valid', 'same'],
                                  parent_name='second_conv_layer',
                                  parent_values=[True]), activation='relu',
                name='conv1d_1'))

        model.add(GlobalMaxPooling1D())

        # hypertune the fully-connected layers
        for l_fc in range(hp.Int('fc_layers_num', min_value=0, max_value=12)):
            model.add(Dense(units=hp.Int('fc_units_' + str(l_fc), min_value=4,
                                         max_value=50), activation='relu'))
            # hypertuner for whether to implement dropout for the current layer
            if hp.Boolean('dropout_' + str(l_fc), default=False):
                model.add(Dropout(hp.Choice('dropout_rate_' + str(l_fc),
                                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                            parent_name='dropout_' + str(l_fc),
                                            parent_values=[True])))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        return model


def prepare_data(data_path, frac=0.7, seed=4):
    """Prepares the data from a CSV-formatted file for training and
    evaluating a model

    The function implements splitting the data into training, validation and
    test sets, feature scaling (namely min-max scaling) and processing each
    set in the format accepted by the model (as defined in
    HyperDetectionModel class)

    :param data_path: file containing the data
    :type data_path: str
    :param frac: fraction of the data to be kept for training
    :type frac: float
    :param seed: seed for random splitting of the data
    :type seed: int
    :return: a dictionary of processed input data and labels for training,
    validation and testing sets
    :rtype: dict
    """
    columns = list(map(str, range(24))) + ['label']
    price_data = pd.read_table(data_path, sep=',', names=columns)

    train = price_data.sample(frac=frac, random_state=seed)
    train_label = train.pop('label')
    df = price_data.drop(train.index)
    validation = df.sample(frac=0.5, random_state=seed)
    validation_label = validation.pop('label')
    test = df.drop(validation.index)
    test_label = test.pop('label')

    # scale based on the training set
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_validation = scaler.transform(validation)
    scaled_test = scaler.transform(test)

    # reshape data to the input shape accepted by the model (as defined in
    # HyperDetectionModel class in this module)
    processed_data = {'train': (
        scaled_train.reshape(scaled_train.shape[0], 24, 1),
        train_label.to_numpy()), 'val': (
        scaled_validation.reshape(scaled_validation.shape[0], 24, 1),
        validation_label.to_numpy()), 'test': (
        scaled_test.reshape(scaled_test.shape[0], 24, 1),
        test_label.to_numpy())}

    return processed_data


def get_hypertuner(settings=None):
    """Creates a BayesianOptimization tuner based on a dictionary of
    parameter settings

    The default dictionary is:
    settings = {'objective': 'val_accuracy', 'max_trials': 40,
                'executions_per_trial': 7, 'seed': 123,
                'name': 'hypertuner'}

    The keys correspond to the arguments with the same name passed to a
    Keras Tuner. Check the documentation for Keras Tuners for more details
    (https://keras-team.github.io/keras-tuner/documentation/tuners/)

    :param settings: parameters for the BayesianOptimization Tuner
    :type settings: dict
    :return: hyperparameter tuner based on Bayesian optimisation
    :rtype: BayesianOptimization
    """
    if settings is None:
        settings = {'objective': 'val_accuracy', 'max_trials': 40,
                    'executions_per_trial': 7, 'seed': 123,
                    'name': 'hypertuner'}

    objective = settings['objective']
    max_trials = settings['max_trials']
    executions_per_trial = settings['executions_per_trial']
    seed = settings['seed']
    name = settings['name']

    hypermodel = HyperDetectionModel()
    tuner = BayesianOptimization(hypermodel, objective=objective,
                                 max_trials=max_trials,
                                 executions_per_trial=executions_per_trial,
                                 seed=seed, project_name=name)

    return tuner


def search_hyperparameters(tuner, train, validation, epochs=150,
                           model_name='model'):
    """Finds the best model given a Keras Tuner and the training and
    validation data

    The function saves and returns the best-performing model found through
    hyperparameter tuning.

    :param model_name: name of the best model to be saved (as path)
    :type model_name: str
    :param tuner: tuner used in hyperparameters search
    :type tuner: Tuner
    :param train: tuple of input data and labels for training
    :type train: tuple
    :param validation: tuple of input data and labels for validation
    :type validation: tuple
    :param epochs: maximum number of training epochs (default 150)
    :type epochs: int
    :return: best model found by the Tuner
    :rtype: Sequential
    """
    train_data, train_label = train
    validation_data, validation_label = validation

    tuner.search(train_data, train_label, epochs=epochs,
                 validation_data=(validation_data, validation_label),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

    # only saves and returns the best model
    best_model = tuner.get_best_models()[0]
    best_model.save(model_name)

    return best_model


def classify_price(model, prices):
    """Returns a list of binary classes (0 or 1) predicted by a model for
    the given pricing curves

    The input pricing data has to be of shape (samples, 24, 1),
    where 'samples' denotes the number of samples in the input.
    The returned array is of shape (samples, 1).

    :param model: model used for classification
    :type model: keras.Model
    :param prices: pricing curves to be classified
    :type prices: numpy.ndarray
    :return: predicted labels
    :rtype: numpy.ndarray
    """
    # use a 50% threshold for deciding the label
    results = (model.predict(prices) > 0.5).astype("int8")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str,
                        help='file containing pricing data')
    args = parser.parse_args()

    # preprocess data
    data_dict = prepare_data(args.data)
    # generate Tuner
    tuner = get_hypertuner()
    # tune the hyperparameters and get the best model
    best_model = search_hyperparameters(tuner, data_dict['train'],
                                        data_dict['val'])
