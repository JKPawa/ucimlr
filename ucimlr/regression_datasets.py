import os
import sys
import inspect
from copy import deepcopy

import numpy as np
import pandas as pd

from ucimlr.helpers import (download_file, download_unzip, one_hot_encode_df_, xy_split,
                            normalize_df_, split_normalize_sequence, split_df, get_split, split_df_on_column)
from ucimlr.dataset import Dataset
from ucimlr.constants import TRAIN
from ucimlr.constants import REGRESSION


def all_datasets():
    """
    Returns a list of all RegressionDataset classes.
    """
    return [cls for _, cls in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(cls)
            and issubclass(cls, RegressionDataset)
            and cls != RegressionDataset]


class RegressionDataset(Dataset):
    type_ = REGRESSION  # Is this necessary?

    @property
    def num_targets(self):
        return self.y.shape[1]


class Abalone(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Abalone).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, header=None)
        y_columns = df.columns[-1:]
        one_hot_encode_df_(df)
        df_test, df_train, df_valid = split_df(df, [0.2, 0.8 - 0.8 * validation_size, 0.8 * validation_size])
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)


class AirFoil(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'airfoil_self_noise.dat'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep='\t', names =["Frequency(Hz)", "Angle of attacks(Deg)", "Chord length(m)", "Free-stream velocity(m/s)", "Suction side displacement thickness(m)", " Scaled sound pressure level(Db)"])
        y_columns = ['Scaled sound pressure level(Db)']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class AirQuality(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'AirQualityUCI.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep=';', parse_dates=[0, 1])
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        df.Date = (df.Date - df.Date.min()).astype('timedelta64[D]')  # Days as int
        df.Time = df.Time.apply(lambda x: int(x.split('.')[0]))  # Hours as int
        df['C6H6(GT)'] = df['C6H6(GT)'].apply(lambda x: float(x.replace(',', '.')))  # Target as float

        # Some floats are given with ',' instead of '.'
        df = df.applymap(lambda x: float(x.replace(',', '.')) if type(x) is str else x)  # Target as float

        df = df[df['C6H6(GT)'] != -200]  # Drop all rows with missing target values
        df.loc[df['CO(GT)'] == -200, 'CO(GT)'] = -10  # -200 means missing value, shifting this to be closer to
        # the other values for this column

        y_columns = ['C6H6(GT)']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Appliances_energy_prediction(RegressionDataset):
    """
    Link to the dataset [description](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'energydata_complete.csv'
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)
        df = pd.read_csv(file_path, parse_dates=[0, 1])
        
        df.date = (df.date - df.date.min()).astype('timedelta64[D]')
        y_columns = ['Appliances']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)
        self.problem_type = REGRESSION


class AutoMPG(RegressionDataset):
    """
    Link to the dataset [description](https://archive.ics.uci.edu/ml/datasets/Automobile).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'auto-mpg.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep='\s+', names =["mpg", "cylinders", "displacements", "horsepower", "weight", "acceleration", "model year", "origin", "car name"])

        y_columns = ['mpg']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)



class Automobile(RegressionDataset):
    """
    Link to the dataset [description](https://archive.ics.uci.edu/ml/datasets/Automobile).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'imports-85.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,  names = ["symboling", "normalized-losses", "make", "fuel-type", " aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", " length", "width", " height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", " fuel-system", " bore", "stroke", " compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"])
        y_columns = ['']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class BeijingAirQuality(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if 'PRSA_Data' not in fn:
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path))
      
       

class BeijingPM(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'PRSA_data_2010.1.1-2014.12.31.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)
        y_columns=['pm2.5']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type)
        self.problem_type = REGRESSION

class BiasCorrection(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Bias_correction_ucl.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, index_col = 'Date', parse_dates= True)


class BikeSharing(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path))

        
    


class CarbonNanotubes(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Carbon+Nanotubes).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'carbon_nanotubes.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00448/carbon_nanotubes.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,sep=';')


class ChallengerShuttleORing(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'o-ring-erosion-only.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/o-ring-erosion-only.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,sep='\s+')



class BlogFeedback(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/BlogFeedback).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name = 'blogData_train.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'
        download_unzip(url, dataset_path)

        # Iterate all test csv and concatenate to one DataFrame
        test_dfs = []
        for fn in os.listdir(dataset_path):
            if 'blogData_test' not in fn:
                continue
            file_path = os.path.join(dataset_path, fn)
            test_dfs.append(pd.read_csv(file_path, header=None))
        df_test = pd.concat(test_dfs)

        file_path = os.path.join(dataset_path, file_name)
        df_train_valid = pd.read_csv(file_path, header=None)
        y_columns = [280]
        df_train_valid[y_columns[0]] = np.log(df_train_valid[y_columns[0]] + 0.01)
        df_test[y_columns[0]] = np.log(df_test[y_columns[0]] + 0.01)

        page_columns = list(range(50))
        for i, (_, df_group) in enumerate(df_train_valid.groupby(page_columns)):
            df_train_valid.loc[df_group.index, 'page_id'] = i
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'page_id')
        df_train.drop(columns='page_id', inplace=True)
        df_valid.drop(columns='page_id', inplace=True)

        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)


class CommunitiesCrime(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'communities.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,header=None)

    

class ConcreteSlumpTest(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'slump_test.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,sep='\s+')



class PropulsionPlants (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI CBM Dataset.zip'
        download_unzip(url, dataset_path)
        filename = 'data.txt'
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, index_col='dteday', parse_dates=True)



class ConcreteCompressiveStrength (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Concrete_Data.xls'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path)


    
class ComputerHardware (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Computer+Hardware).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'machine.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, names=["vendor name", "Model Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])


class CommunitiesCrimeUnnormalized (RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'CommViolPredUnnormalizedData.txt'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, keep_default_na=False, header=None)




class CTSlices(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'
        download_unzip(url, dataset_path)
        file_name = 'slice_localization_data.csv'
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path)
        # No patient should be in both train and test set
        df_train_valid = deepcopy(df.loc[df.patientId < 80, :])  # Pandas complains if it is a view
        df_test = deepcopy(df.loc[df.patientId >= 80, :])        # - " -
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'patientId')
        y_columns = ['reference']
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        df_res = df_res.drop(columns='patientId')
        self.x, self.y = xy_split(df_res, y_columns)


class  ForecastingOrders(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Daily_Demand_Forecasting_Orders.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,sep=';')



class ForecastingStoreData(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Demand+Forecasting+for+a+store).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Daily_Demand_Forecasting_Orders.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00409/'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path,sep='\s+')




class FacebookComments(RegressionDataset):
    """
    Predict the number of likes on posts from a collection of Facebook pages.
    Every page has multiple posts, making the number of pages less than the samples
    in the dataset (each sample is one post).

    # Note
    The provided test split has a relatively large discrepancy in terms
    of distributions of the features and targets. Training and validation splits are
    also made to ensure that the same page is not in both splits. This makes the distributions
    of features in training and validation splits vary to a relatively large extent, possible
    because the number of pages are not that many, while the features are many.

    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip'
        download_unzip(url, dataset_path)
        dataset_path = os.path.join(dataset_path, 'Dataset')

        # The 5th variant has the most data
        train_path = os.path.join(dataset_path, 'Training', 'Features_Variant_5.csv')
        test_path = os.path.join(dataset_path, 'Testing', 'Features_TestSet.csv')
        df_train_valid = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        y_columns = df_train_valid.columns[-1:]

        # Page ID is not included, but can be derived. Page IDs can not be
        # in both training and validation sets
        page_columns = list(range(29))
        for i, (_, df_group) in enumerate(df_train_valid.groupby(page_columns)):
            df_train_valid.loc[df_group.index, 'page_id'] = i
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'page_id')
        df_train.drop(columns='page_id', inplace=True)
        df_valid.drop(columns='page_id', inplace=True)

        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)



class Facebookmetrics (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip'
        download_unzip(url, dataset_path)
        filename = 'dataset_Facebook.csv'
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep=';')



class ForestFires(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Forest+Fires).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'forestfires.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class GNFUV(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/GNFUV+Unmanned+Surface+Vehicles+Sensor+Data).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00452/GNFUV USV Dataset.zip'
        download_unzip(url, dataset_path)


        dfs = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            dfs.append(pd.read_csv(file_path, header=None))

       



class GNFUV_2(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/GNFUV+Unmanned+Surface+Vehicles+Sensor+Data+Set+2).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00466/CNFUV_Datasets.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, header=None))

   



class Greenhouse_Gas_Observing_Network (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Greenhouse+Gas+Observing+Network).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00328/ghg_data.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, header=None, sep='\s+'))

       


class Hungarian_Chickenpox_Cases (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Hungarian+Chickenpox+Cases).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00580/hungary_chickenpox.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, index_col='Date', parse_dates=True)) 


class IIWA14_R820_Gazebo_Dataset_10Trajectories(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/IIWA14-R820-Gazebo-Dataset-10Trajectories).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'IIWA14-R820-Gazebo-Dataset-10Trayectorias.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00574/IIWA14-R820-Gazebo-Dataset-10Trayectorias.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, header=None)



class Metro_Interstate_Traffic_Volume(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Metro_Interstate_Traffic_Volume.csv.gz'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class News_Popularity_Facebook_Economy(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Facebook_Economy.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/Facebook_Economy.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class News_Popularity_Facebook_Microsoft(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Facebook_Microsoft.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/Facebook_Microsoft.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class News_Popularity_Facebook_Palestine(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Facebook_Palestine.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/Facebook_Palestine.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class News_Popularity_GooglePlus_Economy(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'GooglePlus_Economy.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/GooglePlus_Economy.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class News_Popularity_GooglePlus_Microsoft(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'GooglePlus_Microsoft.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/GooglePlus_Microsoft.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class News_Popularity_GooglePlus_Palestine(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'GooglePlus_Palestine.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/GooglePlus_Palestine.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class News_Popularity_GooglePlus_Obama(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'GooglePlus_Obama.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/GooglePlus_Obama.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)




class News_Popularity_LinkedIn_Economy(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'LinkedIn_Economy.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/LinkedIn_Economy.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class News_Popularity_LinkedIn_Microsoft(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'LinkedIn_Microsoft.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/LinkedIn_Microsoft.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class News_Popularity_LinkedIn_Obama(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'LinkedIn_Obama.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/LinkedIn_Obama.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class News_Popularity_LinkedIn_Palestine(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'LinkedIn_Palestine.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/LinkedIn_Palestine.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)




class News_Popularity_News_Final(RegressionDataset):
    """
    Link to the dataset [descriptionhttp://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'News_Final.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)




class Online_Video_Characteristics_and_Transcoding_Time(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Online+Video+Characteristics+and+Transcoding+Time+Dataset).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00335/online_video_dataset.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if fn == 'README.txt':
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, sep='\t'))



class OnlineNews(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv')
        df = pd.read_csv(file_path, )
        df.drop(columns=['url', ' timedelta'], inplace=True)
        y_columns = [' shares']
        df[y_columns[0]] = np.log(df[y_columns[0]])
        self.x, self. y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Parkinson(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/parkinsons).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path: str = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
              'parkinsons/telemonitoring/parkinsons_updrs.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path)
        y_columns = ['motor_UPDRS', 'total_UPDRS']

        df_train_valid = df[df['subject#'] <= 30]
        df_test = deepcopy(df[df['subject#'] > 30])

        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'subject#')
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        df_res.drop(columns='subject#', inplace=True)
        self.x, self.y = xy_split(df_res, y_columns)


class Physicochemical_Properties_of_Protein_Tertiary_Structure(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'CASP.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)


class PPG_DaLiA_Data_Set(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/PPG-DaLiA).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, sep='\t'))

class QSAR_aquatic_toxicity(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/QSAR+aquatic+toxicity).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'qsar_aquatic_toxicity.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep=';', names=["TPSA(Tot)", "SAacc", "H-050", "MLOGP", "RDCHI", " GATS1p", "nN", "C-040", "quantitative response, LC50 [-LOG(mol/L)]"])


class QSAR_fish_bioconcentration_factor(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/QSAR+fish+bioconcentration+factor+%28BCF%29).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00511/QSAR_fish_BCF.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if fn =='ECFP_1024_m0-2_b2_c.txt':
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, sep='\t'))


class QSAR(RegressionDataset):
    """
    Link to the dataset [description]http://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'qsar_fish_toxicity.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep=';', names=[" CIC0", "SM1_Dz(Z)", " GATS1i", "NdsCH", " NdssC", "MLOGP", "quantitative response, LC50 [-LOG(mol/L)]"])





class PowerPlant(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'CCPP', 'Folds5x2_pp.xlsx')
        df = pd.read_excel(file_path)
        y_columns = ['PE']  # Not clear if this is the aim of the dataset
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class ResidentialBuilding(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Residential-Building-Data-Set.xlsx'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path)
        y_columns = ['Y house price of unit area']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class RealEstate(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Real estate valuation data set.xlsx'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path, index_col='No')
        


class Real_time_Election_Results (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/QSAR+fish+bioconcentration+factor+%28BCF%29).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00513/ElectionData2019.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if '.csv' not in fn:
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path))



class Seoul_Bike_Sharing_Demand(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'SeoulBikeData.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class Servo(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Servo).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'servo.data'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, names=["motor", "screw", " pgain", "vgain", "class"])
       


class SGEMM_GPU_kernel_performance (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if fn == 'Readme.txt':
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path))



class Simulated_data_for_survival_modelling (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Simulated+data+for+survival+modelling).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00581/MLtoSurvival-Data.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if fn == '.gitkeep': 
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path))




class SkillCraft1(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'SkillCraft1_Dataset.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)



class SML2010 (RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/SML2010).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            if fn == '.gitkeep':
                continue
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, sep='\s+'))


       

class Solar_Flare(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Solar+Flare).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'flare.data1'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data1'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df1 = pd.read_csv(file_path, header=None, skiprows=[0], sep='\s+')

        filename = 'flare.data2'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df2 = pd.read_csv(file_path, header=None, skiprows=[0], sep='\s+')

        df = pd.merge(df1, df2)


class Synchronous_Machine(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Synchronous+Machine+Data+Set).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'synchronous machine.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00607/synchronous machine.csv'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path)


class Stock_portfolio(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'stock portfolio performance data set.xlsx'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00390/stock portfolio performance data set.xlsx'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path)

       



class Superconductivity(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'train.csv')
        df = pd.read_csv(file_path)
        y_columns = ['critical_temp']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)



class WaveEnergyConverters(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00494/WECs_DataSet.zip'
        download_unzip(url, dataset_path)


        df = []
        for fn in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, fn)
            df.append(pd.read_csv(file_path, header=None))






class WhiteWineQuality(RegressionDataset):
    """
    Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Wine+Quality).

    Citation:
    ```
    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
    ```

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, sep=';')
        y_columns = ['quality']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class YachtHydrodynamics(RegressionDataset):
    """
    Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics).

    Citation:
    ```
    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
    ```

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'yacht_hydrodynamics.data'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, header=None, sep='\s+')
        
