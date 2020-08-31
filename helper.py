# import python packages for data manipulation
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import categorical_encoding as ce
import featuretools as ft
from featuretools.tests.testing_utils import make_ecommerce_entityset

def preprocess_data():
    
    