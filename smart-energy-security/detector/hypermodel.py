from kerastuner import HyperModel, BayesianOptimization
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

        if hp.Boolean('pooling_0', default=True):
            input_size_pooling0 = model.get_layer('conv1d_0').output.shape[1]
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

        if hp.Boolean('second_conv_layer', default=True):
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

        for l_fc in range(hp.Int('fc_layers_num', min_value=0, max_value=12)):
            model.add(Dense(units=hp.Int('fc_units_' + str(l_fc), min_value=4,
                                         max_value=50), activation='relu'))
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


def get_hypertuner(settings=None):
    """

    :param settings:
    :type settings: dict
    :return: hyperparameter tuner based on Bayesian optimisation
    :rtype: BayesianOptimization
    """
    if settings is None:
        settings = {'obj': 'val_accuracy', 'max_trials': 40,
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


def search_hyperparameters(tuner, train, validation, epochs=150):
    """

    :param tuner: tuner used in hyperparameters search
    :type tuner: Tuner
    :param train: tuple of input data and labels for training
    :type train: tuple
    :param validation: tuple of input data and labels for validation
    :type validation: tuple
    :param epochs: maximum number of training epochs (default 150)
    :type epochs: int
    """
    train_data, train_label = train
    validation_data, validation_label = validation

    tuner.search(train_data, train_label, epochs=epochs,
                 validation_data=(validation_data, validation_label),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
