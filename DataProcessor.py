import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    def __init__(self, train_valid_data, test_data, params, code_col='code', date_col='date', factor_cols=None, label_cols=None):
        self.train_valid_data = train_valid_data.copy().set_index([date_col, code_col])
        self.test_data = test_data.set_index([date_col, code_col])
        self.params = params
        self.code_col = code_col
        self.date_col = date_col
        self.factor_cols = factor_cols or list(self.train_valid_data.columns[:-1])
        self.label_cols = label_cols or [self.train_valid_data.columns[-1]]
        self.Processor()
    
    def Processor(self):
        for step in self.params:
            method_name = step.get('class')
            data_group = step.get('data_group', 'all')
            fields_group = step.get('fields_group', 'all')
            method_params = step.get('params')

            if not hasattr(self, method_name):
                raise ValueError(f'Data processing method {method_name} does not exist.')

            func = getattr(self, method_name)

            if data_group in ['all', 'train_valid_data']:
                self.train_valid_data = self._apply_function(self.train_valid_data, func, fields_group, method_params)
            if data_group in ['all', 'test_data']:
                self.test_data = self._apply_function(self.test_data, func, fields_group, method_params)
    
    def _apply_function(self, data, func, fields_group, params):
        cols = {
            'all': data.columns,
            'factor': self.factor_cols,
            'label': self.label_cols
        }.get(fields_group, None)

        if cols is None:
            raise ValueError(f'未知 fields_group {fields_group}')
        
        data.loc[:, cols] = func(data[cols], params)
        return data
    
    def ToNum(self,df,params=None):
        df = df.copy()
        return df.apply(pd.to_numeric, errors='coerce')
    
    def CSMidClipOutlier(self, df, params=5):
        df = df.copy()
        params = params or 5

        def ps_CSMCO(series):
            #series = pd.to_numeric(series, errors='coerce')
            D_M = np.median(series)
            D_M1 = np.median(np.abs(series - D_M))
            return series.clip(lower=D_M - params * D_M1, upper=D_M + params * D_M1)

        return pd.concat([group.apply(ps_CSMCO,axis=0) for _,group in df.groupby(self.date_col)])
        #return df.groupby(self.date_col).apply(lambda group: group.apply(ps_CSMCO, axis=0))

    def CSMinMaxNorm(self,df, params=None):
        return pd.concat([(group - group.min()) / (group.max() - group.min()) for _,group in df.groupby(self.date_col)])
        #df.groupby('date').apply(lambda group: (group - group.min()) / (group.max() - group.min()))

    def CSZScoreNorm(self,df, params=None):
        return pd.concat([(group - group.mean()) / group.std() for _,group in df.groupby(self.date_col)])
        #df.groupby('date').apply(lambda group: (group - group.mean()) / group.std())

    def CSRankNorm(self,df, params=None):
        return pd.concat([(group.rank(method='average')  - group.rank(method='average').mean()) / group.rank(method='average').std() for _,group in df.groupby(self.date_col)])
        #df.groupby('date').apply(lambda group: (group.rank(method='average')  - group.rank(method='average').mean()) / group.rank(method='average').std())

    def DropNa(self, df, params=None):
        return df.dropna()

    def CSZFillNa(self, df, params=None):
        return pd.concat([group.fillna(group.mean()) for _,group in df.groupby(self.date_col)])
        #df.groupby('date').apply(lambda group: group.fillna(group.mean()))
