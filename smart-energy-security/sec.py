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
    os.mkdir(os.path.join(os.path.dirname(Path(__file__)), '../out'))
    resources_path = os.path.normpath(
        os.path.join(os.path.dirname(Path(__file__)), '../resources'))
    out_path = os.path.normpath(
        os.path.join(os.path.dirname(Path(__file__)), '../out'))

    parser = argparse.ArgumentParser()
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

    schedule_args.add_argument('-S', '--schedule', action='store_true',
                               required='--detect' not in sys.argv and '-D'
                                        not in sys.argv,
                               help='select scheduling mode')
    schedule_args.add_argument('-r', '--requirements', type=str,
                               required='--schedule' in sys.argv or '-S' in
                                        sys.argv,
                               help='file containing scheduling requirements '
                                    'that has to be satisfied')
    schedule_args.add_argument('-p', '--pricing', type=str,
                               required=('--schedule' in sys.argv or '-S' in
                                         sys.argv) and not ('--detect' in
                                                            sys.argv or '-D'
                                                            in sys.argv),
                               help='file containing pricing')
    schedule_args.add_argument('-l', '--label', type=str,
                               choices=['none', 'normal', 'abnormal', 'all'],
                               required=False, default='all')
    schedule_args.add_argument('--solutions', action='store_true',
                               required=False)

    args = parser.parse_args()

    if args.detect:
        model = tf.keras.models.load_model(args.model)
        scaler = pickle.load(open(args.scaler, 'rb'))
        price_data = pd.read_table(args.data, sep=',',
                                   names=list(map(str, range(24))))
        scaled_data = scaler.transform(price_data)
        prediction = hm.classify_price(model, scaled_data.reshape(
            scaled_data.shape[0], 24, 1)).flatten()
        predicted_data = pd.concat([price_data, pd.Series(prediction,
                                                          name='label')],
                                   axis=1)
        predicted_data.to_csv(os.path.join(out_path, 'TestingResults.txt'),
                              header=False, index=False)

    if args.schedule:
        os.mkdir(os.path.join(out_path, 'figures'))
        if args.solutions:
            os.mkdir(os.path.join(out_path, 'scheduling'))
        if args.pricing:
            columns = list(map(str, range(24)))
            if args.label != 'none':
                columns.append('label')
            cost = pd.read_table(args.pricing, sep=',', names=columns)
        elif args.detect:
            cost = predicted_data
        if args.label == 'normal':
            #cost = cost.loc[cost['label'] == 0]
            cost.drop(cost.loc[cost['label'] != 0].index, inplace=True)
        elif args.label == 'abnormal':
            #cost = cost.loc[cost['label'] == 1]
            cost.drop(cost.loc[cost['label'] != 1].index, inplace=True)
        if 'label' in cost.columns:
            cost.drop(columns=['label'], inplace=True)
        requirements = pd.read_excel(args.requirements)
        for idx, c in cost.iterrows():
            results = sd.schedule(requirements,
                                  cost=c.to_numpy().flatten().tolist())
            results_matrix = results.x.reshape(len(results.x) // 24, 24)
            sd.plot_consumption(results_matrix.T,
                                os.path.join(out_path, 'figures',
                                             'fig' + str(idx) + '.png'))
            print('#' * 55, '\n')
            print(results.message)
            print('The minimised total cost is:  ', results.fun)
            if args.solutions:
                pd.DataFrame(results_matrix).to_csv(os.path.join(out_path,
                                                                 'scheduling',
                                                                 'scheduling' + str(idx) + '.csv'))
            print('\n\n')


if __name__ == '__main__':
    main()
