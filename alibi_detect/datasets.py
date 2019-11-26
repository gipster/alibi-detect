import logging
import numpy as np
import pandas as pd
import os, zipfile, cv2
from tempfile import TemporaryDirectory
from PIL import Image
import requests
from sklearn.datasets import fetch_kddcup99
from typing import Tuple, Union
from alibi_detect.utils.data import Bunch

pd.options.mode.chained_assignment = None  # default='warn'

logger = logging.getLogger(__name__)


def fetch_traffic_signs(data_folder: str = "../data/traffic/") -> Bunch:
    """
    German traffic signs benchmark data set.
    Parameters
    ----------
    data_folder
        Folder with raw data

    Returns
        train and test set
    -------

    """
    # tmp_dir = data_folder + 'tmp/'
    logger.info('Unzipping file ...')
    with TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(os.path.join(data_folder, 'gtsrb-german-traffic-sign.zip'), "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        train_folder = tmp_dir + '/train/'

        data = []
        labels = []

        height = 32
        width = 32
        channels = 3
        classes = 43
        n_inputs = height * width * channels
        logger.info('Extracting images')
        for i in range(classes):
            path = train_folder + "{0}/".format(i)
            print(path)
            Class = os.listdir(path)
            for a in Class:
                try:
                    image = cv2.imread(path + a)
                    image_from_array = Image.fromarray(image, 'RGB')
                    size_image = image_from_array.resize((height, width))
                    data.append(np.array(size_image))
                    labels.append(i)
                except AttributeError:
                    print(" ")

        y_test = pd.read_csv(os.path.join(tmp_dir, "Test.csv"))
        labels_test = y_test['Path'].as_matrix()
        y_test = y_test['ClassId'].values

        data_test = []

        for f in labels_test:
            pathi = tmp_dir + '/test/' + f.replace('Test/', '')
            image = cv2.imread(pathi)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data_test.append(np.array(size_image))

        X_test = np.array(data_test)

    Cells = np.array(data)
    labels = np.array(labels)

    # Randomize the order of the input images
    s = np.arange(Cells.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    Cells = Cells[s]
    labels = labels[s]

    (X_train, X_val) = Cells[(int)(0.2 * len(labels)):], Cells[:(int)(0.2 * len(labels))]
    (y_train, y_val) = labels[(int)(0.2 * len(labels)):], labels[:(int)(0.2 * len(labels))]

    train, val, test = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    return Bunch(train=train, val=val, test=test)


def fetch_kdd(target: list = ['dos', 'r2l', 'u2r', 'probe'],
              keep_cols: list = ['srv_count', 'serror_rate', 'srv_serror_rate',
                                 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                                 'dst_host_srv_count', 'dst_host_same_srv_rate',
                                 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                                 'dst_host_srv_rerror_rate'],
              percent10: bool = True,
              return_X_y: bool = False) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    KDD Cup '99 dataset. Detect computer network intrusions.

    Parameters
    ----------
    target
        List with attack types to detect.
    keep_cols
        List with columns to keep. Defaults to continuous features.
    percent10
        Bool, whether to only return 10% of the data.
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Dataset and outlier labels (0 means 'normal' and 1 means 'outlier').
    (data, target)
        Tuple if 'return_X_y' equals True.
    """

    # fetch raw data
    data_raw = fetch_kddcup99(subset=None, data_home=None, percent10=percent10)

    # specify columns
    cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    # create dataframe
    data = pd.DataFrame(data=data_raw['data'], columns=cols)

    # add target to dataframe
    data['attack_type'] = data_raw['target']

    # specify and map attack types
    attack_list = np.unique(data['attack_type'])
    attack_category = ['dos', 'u2r', 'r2l', 'r2l', 'r2l', 'probe', 'dos', 'u2r',
                       'r2l', 'dos', 'probe', 'normal', 'u2r', 'r2l', 'dos', 'probe',
                       'u2r', 'probe', 'dos', 'r2l', 'dos', 'r2l', 'r2l']

    attack_types = {}
    for i, j in zip(attack_list, attack_category):
        attack_types[i] = j

    data['attack_category'] = 'normal'
    for k, v in attack_types.items():
        data['attack_category'][data['attack_type'] == k] = v

    # define target
    data['target'] = 0
    for t in target:
        data['target'][data['attack_category'] == t] = 1
    is_outlier = data['target'].values

    # define columns to be dropped
    drop_cols = []
    for col in data.columns.values:
        if col not in keep_cols:
            drop_cols.append(col)

    if drop_cols != []:
        data.drop(columns=drop_cols, inplace=True)

    if return_X_y:
        return data.values, is_outlier

    return Bunch(data=data.values,
                 target=is_outlier,
                 target_names=['normal', 'outlier'],
                 feature_names=keep_cols)


def fetch_nab(ts: str,
              return_X_y: bool = False
              ) -> Union[Bunch, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Get time series in a DataFrame from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

    Parameters
    ----------
    ts

    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Dataset and outlier labels (0 means 'normal' and 1 means 'outlier') in DataFrames with timestamps.
    (data, target)
        Tuple if 'return_X_y' equals True.
    """
    url_labels = 'https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json'
    r = requests.get(url_labels)
    labels_json = r.json()
    outliers = labels_json[ts + '.csv']
    if not outliers:
        logger.warning('The dataset does not contain any outliers.')
    url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/' + ts + '.csv'
    df = pd.read_csv(url, header=0, index_col=0)
    labels = np.zeros(df.shape[0])
    for outlier in outliers:
        outlier_id = np.where(df.index == outlier)[0][0]
        labels[outlier_id] = 1
    df.index = pd.to_datetime(df.index)
    df_labels = pd.DataFrame(data={'is_outlier': labels}, index=df.index)

    if return_X_y:
        return df, df_labels

    return Bunch(data=df,
                 target=df_labels,
                 target_names=['normal', 'outlier'])


def get_list_nab() -> list:
    """
    Get list of possible time series to retrieve from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

    Returns
    -------
    List with time series names.
    """
    url_labels = 'https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json'
    r = requests.get(url_labels)
    labels_json = r.json()
    files = [k[:-4] for k, v in labels_json.items()]
    return files
