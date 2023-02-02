import numpy as np
import pandas as pd
import neurokit2 as nk
import os



def eda_preprop(eda_label, data_loc='data/eda_data/', window_size = 5):
    file_ids = eda_label['file Id'].values
    X_data = []
    y_data = []
    for file_id in file_ids:
        eda_csv = pd.read_csv(f"{data_loc}{file_id}.csv")
        eda_r = nk.bio_process(eda = eda_csv.values)
        eda_tonic = eda_r[0]['EDA_Tonic'].values
        eda_phasic = eda_r[0]['EDA_Phasic'].values
        

        for i in range(len(eda_tonic)-window_size):
            X = np.array([eda_tonic[i:i+window_size], eda_phasic[i:i+window_size]])
            y = eda_label[eda_label['file Id'] == file_id].label.values[0]
            if not len(X_data):
                X_data = X
                y_data = y
            else:
                X_data = np.dstack((X_data, X))
                y_data = np.append(y_data,y)

    X_data = np.swapaxes(X_data, 0,2)
    return X_data, y_data

def eeg_preprep(eeg_label, data_loc = 'data/Muse_data/', window_size = 10):

    file_ids = eeg_label['file Id'].values
    X_data = []
    y_data = []
    for file_id in file_ids:
        df = pd.read_csv(data_loc+f'{file_id}.csv')
        df = df.fillna(0)
        unique_vars = set([var[0] for var in df.columns.str.split('_')])
        for var in unique_vars:
            df[var] = df.filter(like = var).mean(axis=1)
        eeg_bands = ['Alpha', 'Delta', 'Beta', 'Gamma', 'Theta']
        df_bands = df[eeg_bands]
        window_size = 10
        for i in range(len(df_bands)-window_size):
            X = np.array(df_bands.iloc[i:i+window_size,:])
            y = eeg_label[eeg_label['file Id'] == file_id].label.values[0]
            if not len(X_data):
                X_data = X
                y_data = y
            else:
                X_data = np.dstack((X_data, X))
                y_data = np.append(y_data,y)

    X_data = np.swapaxes(X_data, 0,2)
    X_data = np.swapaxes(X_data, 1,2)
    return X_data, y_data


if __name__ == '__main__':

    eda_label = pd.read_csv("data/eda_labels.csv")
    x, y = eda_preprop(eda_label)

    eeg_label = pd.read_csv("data/eeg_labels.csv")
    data_loc = 'data/Muse_data/'
    x_, y_ = eeg_preprep(eeg_label)
    