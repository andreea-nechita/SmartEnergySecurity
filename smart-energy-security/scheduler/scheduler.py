import numpy as np
import pandas as pd
from scipy.optimize import linprog


def schedule(input_df, cost, method='simplex'):
    """

    :param input_df: input DataFrame
    :type input_df: DataFrame
    :param cost:
    :type cost: list
    :param method:
    :type method: str
    :return: OptimizeResult of the computed scheduling
    :rtype: OptimizeResult
    """
    task_count = len(input_df)
    total_cost = cost * task_count
    energy_consumptions = np.zeros((task_count, 24 * task_count), dtype=int)
    energy_demands = []
    hourly_bounds = []
    for idx, task in input_df.iterrows():
        energy_consumptions[idx, idx * 24:(idx + 1) * 24] = 1
        energy_demands.append(task[4])
        for hour in range(24):
            if task[1] <= hour <= task[2]:
                bound = (0, task[3])
            else:
                bound = (0, 0)
            hourly_bounds.append(bound)
    result = linprog(total_cost, bounds=hourly_bounds,
                     A_eq=energy_consumptions, b_eq=energy_demands,
                     method=method)
    return result


def plot_consumption(consumption_matrix, path=None):
    """

    :param consumption_matrix:
    :type consumption_matrix:
    :param path: path to save the plot
    :type path: str
    """
    df = pd.DataFrame(consumption_matrix)
    ax = df.plot.bar(stacked=True, color='blue', figsize=(10, 5.5),
                     legend=False)
    ax.set_xlabel('Hour of the day', labelpad=7.5)
    ax.set_ylabel('Energy consumption', labelpad=7.5)

    if path is not None:
        fig = ax.get_figure()
        fig.savefig(path)
