import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

import detector.hypermodel as hm
import scheduler.scheduler as sd


def main():
    # create relative path to resources directory
    resources_path = os.path.normpath(
        os.path.join(os.path.dirname(Path(__file__)), '../resources'))

    parser = argparse.ArgumentParser(
        epilog='All output files are saved in a new directory named "out" ('
               'in the same directory as the "resources" and the main '
               'application package)')
    detect_args = parser.add_argument_group('Detection arguments',
                                            description='Detection of '
                                                        'abnormal pricing '
                                                        'curve')
    schedule_args = parser.add_argument_group('Scheduling arguments',
                                              description='Task scheduling '
                                                          'based on the '
                                                          'guideline price '
                                                          'and scheduling '
                                                          'requirements ('
                                                          'i.e. ready time, '
                                                          'deadline, maximum '
                                                          'scheduled energy '
                                                          'per hour, energy '
                                                          'demand)')

    # Detection arguments starting here
    detect_args.add_argument('-D', '--detect', action='store_true',
                             required='--schedule' not in sys.argv and '-S'
                                      not in sys.argv,
                             help='select detection mode')
    detect_args.add_argument('-d', '--data', type=str,
                             required='--detect' in sys.argv or '-D' in
                                      sys.argv,
                             help='file containing pricing data to be '
                                  'classified')
    detect_args.add_argument('-s', '--scaler', type=str,
                             help='pickle file containing the scaler for the '
                                  'data (by default, the scaler stored in '
                                  'resources is used)',
                             default=os.path.join(resources_path,
                                                  'scaler.pkl'))
    detect_args.add_argument('-m', '--model', type=str,
                             help='model to be used for classification (by '
                                  'default, the model stored in resources is '
                                  'used)',
                             default=os.path.join(resources_path, 'model'))

    # Scheduling arguments starting here
    schedule_args.add_argument('-S', '--schedule', action='store_true',
                               required='--detect' not in sys.argv and '-D'
                                        not in sys.argv,
                               help='select scheduling mode')
    schedule_args.add_argument('-r', '--requirements', type=str,
                               required='--schedule' in sys.argv or '-S' in
                                        sys.argv,
                               help='Excel file containing scheduling '
                                    'requirements that has to be satisfied')
    schedule_args.add_argument('-p', '--pricing', type=str,
                               required=('--schedule' in sys.argv or '-S' in
                                         sys.argv) and not ('--detect' in
                                                            sys.argv or '-D'
                                                            in sys.argv),
                               help='csv file containing pricing data')
    schedule_args.add_argument('-l', '--label', type=str,
                               choices=['none', 'normal', 'abnormal', 'all'],
                               required=False, help='labels to be used in '
                                                    'scheduling; none '
                                                    '-- the pricing '
                                                    'curves do not '
                                                    'contain any '
                                                    'label; normal -- '
                                                    'only pricing curves '
                                                    'labelled as normal are '
                                                    'considered; abnormal -- '
                                                    'only pricing curves '
                                                    'labelled as abnormal '
                                                    'are considered; all -- '
                                                    'pricing curves are '
                                                    'labelled and all are '
                                                    'considered (the default '
                                                    'option is "all")',
                               default='all')
    schedule_args.add_argument('--solutions', action='store_true',
                               required=False, help='option for saving '
                                                    'scheduling results in '
                                                    '.csv files (separate '
                                                    'files for each pricing '
                                                    'curve in the input data)')

    args = parser.parse_args()

    # create directory for output files
    os.mkdir(os.path.join(os.path.dirname(Path(__file__)), '../out'))
    out_path = os.path.normpath(
        os.path.join(os.path.dirname(Path(__file__)), '../out'))

    # check if detection mode is selected
    if args.detect:
        model = tf.keras.models.load_model(args.model)
        scaler = pickle.load(open(args.scaler, 'rb'))
        price_data = pd.read_table(args.data, sep=',',
                                   names=list(map(str, range(24))))
        # scale input data
        scaled_data = scaler.transform(price_data)
        # reshape the data to be classified into the shape accepted by the
        # model (i.e. (samples_number, 24, 1))
        prediction = hm.classify_price(model, scaled_data.reshape(
            scaled_data.shape[0], 24, 1)).flatten()
        # add label column
        predicted_data = pd.concat(
            [price_data, pd.Series(prediction, name='label')], axis=1)
        predicted_data.to_csv(os.path.join(out_path, 'TestingResults.txt'),
                              header=False, index=False)

    # check if scheduling mode is selected
    if args.schedule:
        os.mkdir(os.path.join(out_path, 'figures'))
        if args.solutions:
            os.mkdir(os.path.join(out_path, 'scheduling'))
        if args.pricing:
            columns = list(map(str, range(24)))
            # use label column if a pricing option other than 'none' is
            # mentioned
            if args.label != 'none':
                columns.append('label')
            cost = pd.read_table(args.pricing, sep=',', names=columns)
        # use classified guideline prices if detection is also performed
        elif args.detect:
            cost = predicted_data
        # keep only 0-labelled pricing curves for 'normal' mode
        if args.label == 'normal':
            cost.drop(cost.loc[cost['label'] != 0].index, inplace=True)
        # keep only 1-labelled pricing curves for 'normal' mode
        elif args.label == 'abnormal':
            cost.drop(cost.loc[cost['label'] != 1].index, inplace=True)
        # drop label column as it is no longer needed
        if 'label' in cost.columns:
            cost.drop(columns=['label'], inplace=True)

        # read scheduling requirements
        requirements = pd.read_excel(args.requirements)
        for idx, c in cost.iterrows():
            # schedule() requires a 1D list as input pricing
            results = sd.schedule(requirements,
                                  cost=c.to_numpy().flatten().tolist())
            # results are returns as a 1D array of length 24 * (total number
            # of tasks)
            results_matrix = results.x.reshape(len(results.x) // 24, 24)
            # use the transposed matrix in order to plot the chart based on
            # hours (instead of the tasks)
            sd.plot_consumption(results_matrix.T,
                                os.path.join(out_path, 'figures',
                                             'fig' + str(idx) + '.png'))
            print('#' * 55, '\n')
            print(results.message)
            print('The minimised total cost is:  ', results.fun)
            if args.solutions:
                pd.DataFrame(results_matrix).to_csv(
                    os.path.join(out_path, 'scheduling',
                                 'scheduling' + str(idx) + '.csv'))
            print('\n\n')


if __name__ == '__main__':
    main()
