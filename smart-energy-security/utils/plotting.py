import pandas as pd


def plot_consumption(consumption_matrix):
    df = pd.DataFrame(consumption_matrix)
    df.plot.bar(stacked=True, color='blue', figsize=(10, 4),
                xlabel='Hour of the day', ylabel='Energy consumption',
                legend=False)
