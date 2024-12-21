
# *** modules/data_loading.py ***

import pandas as pd
import re


def load_data():
    features = pd.read_csv('Data/raw/features.txt', sep=r'\s+', header=None)[1].values

    # get the data from txt files to pandas dataframe
    X_train = pd.read_csv('Data/raw/train/X_train.txt', sep=r'\s+', header=None)
    X_test = pd.read_csv('Data/raw/test/X_test.txt', sep=r'\s+', header=None)

    X_train.columns = [features]
    X_test.columns = [features]

    # add subject column to the dataframe

    def read_to_series(filepath):
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] == 1:  # If the dataframe has only one column
            return df.iloc[:, 0]  # Return it as a Series
        else:
            return df  # Otherwise, return the DataFrame

    # Add subject column to the dataframe, using the read_to_series function for compatibility
    X_train['subject'] = read_to_series('Data/raw/train/subject_train.txt')
    X_test['subject'] = read_to_series('Data/raw/test/subject_test.txt')

    # activity_labels = pd.read_csv('Data/UCI_HAR_Dataset_v2/activity_labels.txt', header=None, squeeze=True)
    activity_labels = pd.read_csv('Data/raw/activity_labels.txt', header=None)
    activity_labels = [re.findall('[^(0-9) | /s]+', value)[0] for value in activity_labels[0].values]

    # train labels
    # Load the dataset
    y_train = pd.read_csv('Data/raw/train/y_train.txt', header=None, sep=r'\s+')
    # Assuming the target values are in the first column, extract and convert to numeric
    y_train = y_train.iloc[:, 0]
    # Convert to Series (if not already) and subtract 1 from each value
    y_train = pd.Series([value - 1 for value in y_train])
    y_train_labels = y_train.map({i: activity_labels[i] for i in range(len(activity_labels))})
    # main.py labels

    y_test = pd.read_csv('Data/raw/test/y_test.txt', header=None, sep=r'\s+')
    y_test = y_test.iloc[:, 0]
    y_test = pd.Series([value - 1 for value in y_test])
    y_test_labels = y_test.map({i: activity_labels[i] for i in range(len(activity_labels))})

    # put all columns in a single dataframe  (train)
    train = X_train
    train['Activity'] = y_train
    train['ActivityName'] = y_train_labels

    # put all columns in a single dataframe (main.py)
    test = X_test
    test['Activity'] = y_test
    test['ActivityName'] = y_test_labels

    train.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    test.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    df = pd.concat([train, test])
    df = df.sample(frac=1).reset_index(drop=True)

    df_feature = pd.DataFrame(features)
    keys_with_multiple_values = df_feature.value_counts()[lambda x: x > 1].index.tolist()
    keys_with_multiple_values = [x[0] for x in keys_with_multiple_values]

    columns = list()
    keys_with_multiple_values_dict = {x: 1 for x in keys_with_multiple_values}
    for col in df.columns:
        if col[0] in keys_with_multiple_values:
            count = keys_with_multiple_values_dict[col[0]]
            columns.append(f'{col[0]}_{count}')
            keys_with_multiple_values_dict[col[0]] += 1
        else:
            columns.append(col[0])

    df.columns = columns

    return df


def load_cleaned_data():
    """Loads the cleaned data if files exist."""

    df_cleaned = pd.read_csv('Data/processed/data_uci_handled_outliers.csv')
    x_cleaned_zs_iqr = {'IQR': pd.read_csv('Data/processed/x_uci_handled_outliers_iqr.csv'),
                        'ZS': pd.read_csv('Data/processed/x_uci_handled_outliers_zs.csv')}

    return df_cleaned, x_cleaned_zs_iqr
