import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from pandas import DataFrame
from pandas.plotting import parallel_coordinates


def transform_data_year_only(data: DataFrame, column_name: str, year: int) -> DataFrame:
    data.columns = ['Technology', 'loved', 'wanted', 'dreaded']
    data.set_index('Technology', inplace=True)
    data = data[[column_name]]
    data.columns = [f'{year}']
    return data


def merge_rows(data: DataFrame, rowname_a: str, rowname_b: str) -> DataFrame:
    # TODO: Do it properly
    data = data.T
    data[rowname_a] = data[rowname_a].fillna(data[rowname_b])
    data = data.T
    data = data.drop(rowname_b)
    return data


def main():
    years = list(range(2015, 2023))
    dataframes = dict.fromkeys(years)
    for year in years:
        dataframes[year] = pd.read_csv(f'csv_data/{year}_loved_dreaded.csv')
        dataframes[year] = transform_data_year_only(dataframes[year], 'loved', year)

    # Join the dataframes
    data = dataframes[2015]
    for year in years[1:]:
        data = pd.merge(data, dataframes[year], how='outer', left_index=True, right_index=True)

    # Plot the data
    data = merge_rows(data, 'Bash/Shell', 'Bash/Shell/PowerShell')
    data = data.dropna(thresh=data.shape[1] * 0.3, how='all', axis=0)
    print(f'Data points: {len(data)}')

    data.reset_index(level=0, inplace=True)
    data = data.sort_values(by=[str(i) for i in years[::-1]], ascending=False)
    ax = parallel_coordinates(data, 'Technology', colormap=plt.get_cmap('gist_ncar'), linewidth=3, alpha=0.7, marker='o', markersize=7)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('Most Loved Programming Languages '
              '\nStack Overflow Developer Survey data 2015-2022', fontsize=22)

    plt.show()


if __name__ == '__main__':
    main()
