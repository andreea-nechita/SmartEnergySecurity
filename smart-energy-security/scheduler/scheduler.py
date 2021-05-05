import numpy as np
import pandas as pd
from scipy.optimize import linprog


def schedule(input_df, cost, method='simplex'):
    """Schedules energy consumption given the task requirements and the
    pricing curve using linear programming

    By default, it uses the 'simplex' algorithm. For the other methods that
    are supported, check the SciPy documentation for scipy.optimize.linprog
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)

    :param input_df: scheduling requirements to be satisfied
    :type input_df: DataFrame
    :param cost: guideline pricing curves
    :type cost: list
    :param method: method used in linear-programming solving
    :type method: str
    :return: OptimizeResult of the computed scheduling
    :rtype: OptimizeResult
    """
    # find the number of tasks to be scheduled
    task_count = len(input_df)
    # the total cost function to be minimised, expressed as a list of
    # coefficients of the price for each hour, which is repeated for each task
    total_cost = cost * task_count
    # generate a matrix of coefficients for all the scheduling requirements
    # equations (i.e. the hours when each task consumes energy)
    energy_consumptions = np.zeros((task_count, 24 * task_count), dtype=int)
    # list of energy demand of each task
    energy_demands = []
    # list of tuples representing the min and max hourly energy consumption
    # of each task
    hourly_bounds = []
    for idx, task in input_df.iterrows():
        # set the hours when this task can consume energy
        energy_consumptions[idx, idx * 24:(idx + 1) * 24] = 1
        energy_demands.append(task[4])
        # set the max scheduled energy per hour for this task as a tuple of
        # (min, max)
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
    """Generates a bar chart of the given energy consumption

    The input data has to be of size (24, tasks), where tasks is the total
    number of scheduled tasks. Each entry represents how much energy is
    scheduled for a task at each hour.

    :param consumption_matrix: ndarray (or any data that can be used to
    construct a pandas DataFrame) of the scheduled tasks
    :type consumption_matrix: numpy.ndarray
    :param path: path to save the plot
    :type path: str
    """
    df = pd.DataFrame(consumption_matrix)
    ax = df.plot.bar(stacked=True, color='blue', figsize=(10, 5.5),
                     legend=False)
    ax.set_xlabel('Hour of the day', labelpad=7.5)
    ax.set_ylabel('Energy consumption', labelpad=7.5)

    # save the figure if a path is provided
    if path is not None:
        fig = ax.get_figure()
        fig.savefig(path)
