import numpy as np
import pandas as pd
from scipy.optimize import linprog


def schedule(input_df, cost, method='simplex'):
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


def plot_consumption(consumption_matrix):
    df = pd.DataFrame(consumption_matrix)
    df.plot.bar(stacked=True, color='blue', figsize=(10, 4),
                xlabel='Hour of the day', ylabel='Energy consumption',
                legend=False)