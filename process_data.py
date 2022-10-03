import os
import shutil

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Store all columns for each year
columns_to_use = {2015: ['Current Lang & Tech: Android', 'Current Lang & Tech: Arduino',
                         'Current Lang & Tech: AngularJS', 'Current Lang & Tech: C',
                         'Current Lang & Tech: C++', 'Current Lang & Tech: C++11',
                         'Current Lang & Tech: C#', 'Current Lang & Tech: Cassandra',
                         'Current Lang & Tech: CoffeeScript', 'Current Lang & Tech: Cordova',
                         'Current Lang & Tech: Clojure', 'Current Lang & Tech: Cloud',
                         'Current Lang & Tech: Dart', 'Current Lang & Tech: F#',
                         'Current Lang & Tech: Go', 'Current Lang & Tech: Hadoop',
                         'Current Lang & Tech: Haskell', 'Current Lang & Tech: iOS',
                         'Current Lang & Tech: Java', 'Current Lang & Tech: JavaScript',
                         'Current Lang & Tech: LAMP', 'Current Lang & Tech: Matlab',
                         'Current Lang & Tech: MongoDB', 'Current Lang & Tech: Node.js',
                         'Current Lang & Tech: Objective-C', 'Current Lang & Tech: Perl',
                         'Current Lang & Tech: PHP', 'Current Lang & Tech: Python',
                         'Current Lang & Tech: R', 'Current Lang & Tech: Redis',
                         'Current Lang & Tech: Ruby', 'Current Lang & Tech: Rust',
                         'Current Lang & Tech: Salesforce', 'Current Lang & Tech: Scala',
                         'Current Lang & Tech: Sharepoint', 'Current Lang & Tech: Spark',
                         'Current Lang & Tech: SQL', 'Current Lang & Tech: SQL Server',
                         'Current Lang & Tech: Swift', 'Current Lang & Tech: Visual Basic',
                         'Current Lang & Tech: Windows Phone', 'Current Lang & Tech: Wordpress',
                         'Current Lang & Tech: Write-In', 'Future Lang & Tech: Android',
                         'Future Lang & Tech: Arduino', 'Future Lang & Tech: AngularJS',
                         'Future Lang & Tech: C', 'Future Lang & Tech: C++',
                         'Future Lang & Tech: C++11', 'Future Lang & Tech: C#',
                         'Future Lang & Tech: Cassandra', 'Future Lang & Tech: CoffeeScript',
                         'Future Lang & Tech: Cordova', 'Future Lang & Tech: Clojure',
                         'Future Lang & Tech: Cloud', 'Future Lang & Tech: Dart',
                         'Future Lang & Tech: F#', 'Future Lang & Tech: Go',
                         'Future Lang & Tech: Hadoop', 'Future Lang & Tech: Haskell',
                         'Future Lang & Tech: iOS', 'Future Lang & Tech: Java',
                         'Future Lang & Tech: JavaScript', 'Future Lang & Tech: LAMP',
                         'Future Lang & Tech: Matlab', 'Future Lang & Tech: MongoDB',
                         'Future Lang & Tech: Node.js', 'Future Lang & Tech: Objective-C',
                         'Future Lang & Tech: Perl', 'Future Lang & Tech: PHP',
                         'Future Lang & Tech: Python', 'Future Lang & Tech: R',
                         'Future Lang & Tech: Redis', 'Future Lang & Tech: Ruby',
                         'Future Lang & Tech: Rust', 'Future Lang & Tech: Salesforce',
                         'Future Lang & Tech: Scala', 'Future Lang & Tech: Sharepoint',
                         'Future Lang & Tech: Spark', 'Future Lang & Tech: SQL',
                         'Future Lang & Tech: SQL Server', 'Future Lang & Tech: Swift',
                         'Future Lang & Tech: Visual Basic', 'Future Lang & Tech: Windows Phone',
                         'Future Lang & Tech: Wordpress', 'Future Lang & Tech: Write-In'],
                  2016: ['tech_do', 'tech_want'],
                  2017: ['HaveWorkedLanguage', 'WantWorkLanguage'],
                  2018: ['LanguageWorkedWith', 'LanguageDesireNextYear'],
                  2021: ['LanguageHaveWorkedWith', 'LanguageWantToWorkWith']
                  }

# Store all path and year for processing
datapaths = {2016: ['./survey_data/2016/2016 Stack Overflow Survey Results/2016 Stack Overflow Survey Responses.csv', 2016],
             2017: ['./survey_data/2017/survey_results_public.csv', 2017],
             2018: ['./survey_data/2018/survey_results_public.csv', 2018],
             2019: ['./survey_data/2019/survey_results_public.csv', 2018],
             2020: ['./survey_data/2020/survey_results_public.csv', 2018],
             2021: ['./survey_data/2021/survey_results_public.csv', 2021],
             2022: ['./survey_data/2022/survey_results_public.csv', 2021]
             }

def extract_zip() -> None:
    zip_files = [file for file in os.listdir() if file.endswith('.zip')]
    for zip_file in zip_files:
        file_end = zip_file.split('-')[4]
        year = file_end.replace('.zip', '')
        unzip_dir = f'./survey_data/{year}'
        if not os.path.exists(unzip_dir):
            os.mkdir(unzip_dir)
        print(f'Unarchiving {year} survey data')
        shutil.unpack_archive(zip_file, unzip_dir)


def parse_two_fifteen_loved_dreaded(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, usecols=columns_to_use[2015], header=1)
    # TODO: Calculate loved/dreaded language ratios
    df = df.notna() * 1
    df.columns = df.columns.str.replace("Current Lang & Tech: ", "current ")
    df.columns = df.columns.str.replace("Future Lang & Tech: ", "future ")
    num_columns = df.shape[1]
    print(num_columns)

    df_first_half = df.iloc[:, :int(num_columns / 2)]
    df_first_half.columns = df_first_half.columns.str.replace("current ", "")
    df_second_half = df.iloc[:, int(num_columns / 2):]
    # Renaming columns to match the first half so we can do some magic
    df_second_half.columns = df_first_half.columns
    # logical and
    loved = (df_first_half + df_second_half) > 1
    # no
    wanted = df_first_half < df_second_half
    # nonlogical
    dreaded = df_first_half > df_second_half

    # users nums
    users_who_use = df_first_half.sum(axis=0)
    users_who_dont_use = df_second_half.sum(axis=0)

    loved = loved.sum(axis=0) / users_who_use
    wanted = wanted.sum(axis=0) / users_who_dont_use
    dreaded = dreaded.sum(axis=0) / users_who_use

    loved = loved.T
    wanted = wanted.T
    dreaded = dreaded.T

    new_cols = ['loved', 'wanted', 'dreaded']
    new_df = pd.concat([loved, wanted, dreaded], axis=1)
    new_df.columns = new_cols

    return new_df


def parse_loved_dreaded(filepath: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, usecols=columns_to_use[year])
    df.dropna(inplace=True)
    df.rename(columns={columns_to_use[year][0]: 'tech_do', columns_to_use[year][1]: 'tech_want'}, inplace=True)
    # df.reset_index(inplace=True)
    if year < 2018:
        df.tech_do = df.tech_do.str.split('; ')
        df.tech_want = df.tech_want.str.split('; ')
    else:
        df.tech_do = df.tech_do.str.split(';')
        df.tech_want = df.tech_want.str.split(';')

    mlb = MultiLabelBinarizer(sparse_output=True)

    mlb.fit(df['tech_do'])
    # mlb.fit(df['tech_want'])

    do = pd.DataFrame.sparse.from_spmatrix(
        mlb.transform(df['tech_do']),
        index=df.index,
        columns=mlb.classes_) * 1.0

    want = pd.DataFrame.sparse.from_spmatrix(
        mlb.transform(df['tech_want']),
        index=df.index,
        columns=mlb.classes_) * 1.0

    # logical and
    loved = (do + want) > 1
    # no
    wanted = (do < want) * 1
    # nonlogical
    dreaded = (do > want) * 1

    # users nums
    users_who_use = do.sum(axis=0)
    users_who_dont_use = want.sum(axis=0)

    loved = loved.sum(axis=0) / users_who_use
    wanted = wanted.sum(axis=0) / users_who_dont_use
    dreaded = dreaded.sum(axis=0) / users_who_use

    loved = loved.T
    wanted = wanted.T
    dreaded = dreaded.T

    new_cols = ['loved', 'wanted', 'dreaded']
    new_df = pd.concat([loved, wanted, dreaded], axis=1)
    new_df.columns = new_cols

    return new_df


def main():
    # Let's create a directory to store all csv files and survey data
    if not os.path.exists('./csv_data'):
        os.mkdir('./csv_data')
    if not os.path.exists('./survey_data'):
        os.mkdir('./survey_data')
    extract_zip()

    two_fifteen_datapath = './survey_data/2015/2015 Stack Overflow Developer Survey Responses.csv'
    two_fifteen_data = parse_two_fifteen_loved_dreaded(two_fifteen_datapath)
    two_fifteen_data.to_csv('csv_data/2015_loved_dreaded.csv')
    print(two_fifteen_data.head())

    for key, value in datapaths.items():
        # Parse from year 2016 onwards
        data = parse_loved_dreaded(value[0], value[1])
        data.to_csv(f'csv_data/{key}_loved_dreaded.csv')
        print(data.head())


if __name__ == '__main__':
    main()
