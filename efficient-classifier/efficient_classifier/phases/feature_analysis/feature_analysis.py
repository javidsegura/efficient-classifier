

from library.phases.dataset.dataset import Dataset
from library.phases.data_preprocessing.data_preprocessing import Preprocessing
from library.phases.feature_analysis.feature_engineering.feature_engineering import FeatureEngineering
from library.phases.feature_analysis.feature_selection.feature_selection import FeatureSelection
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import mutual_info_regression

from imblearn.over_sampling import SMOTENC
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import  LogisticRegression, Lasso
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from library.phases.feature_analysis.feature_transformation.feature_transformation_factory import FeatureTransformation

from library.phases.EDA.EDA import EDA


class FeatureAnalysis:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.feature_transformation = FeatureTransformation(self.dataset)
        self.feature_engineering = FeatureEngineering(self.dataset)
        self.feature_selection = FeatureSelection(self.dataset)



