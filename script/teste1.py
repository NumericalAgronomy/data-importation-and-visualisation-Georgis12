# Import necessary libraries
import panda as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate
from chemotools.feature_selection import RangeCut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.cross_decomposition import PLSRegression
# Load the spectra data
spectra = pd.read_csv('work/data/combined_data.csv', 
                      sep=';', 
                      index_col=0)

# Load the classes data
labels = pd.read_csv('work/data/labels_3cl.csv',
                     sep=';',
                     index_col=0)
print(labels)