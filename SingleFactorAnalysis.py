import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SingleFactorAnalysis:
    def __init__(self,data,factor_col,code_col='code',date_col='date',return_col='y',quantile_num=10):
        self.data = data.copy()
        self.factor_col = factor_col
        self.code_col = code_col
        self.date_col = date_col
        self.return_col = return_col
        self.quantile_num = quantile_num
        self.factordf = self.data.set_index([self.date_col,self.code_col])[[self.factor_col,self.return_col]]
        self.icdf = None
        self.quantfactordf = None
    def calcIC(self):
        factor_df = self.factordf.copy().dropna()
        datelist = factor_df.index.get_level_values(0)
        factor_df.columns = ['factor','returns']
        IC_df = factor_df.groupby(datelist).apply(lambda x: spearmanr(x['factor'], x['returns'])[0])
        IC_df = pd.DataFrame(IC_df,columns=['RankIC'])
        self.icdf = IC_df
        self.icmean = IC_df['RankIC'].mean()
    def plotIC(self):
        if self.icdf is None:
            self.calcIC()
        ic = self.icdf.copy()
        ic['RankIC_cum'] = ic['RankIC'].cumsum()
        fig,ax1 = plt.subplots(figsize=(12,6))
        ax1.bar(ic.index, ic["RankIC"], color="red", alpha=0.6, label="RankIC", width=5)
        ax1.axhline(self.icmean,label=f'RankIC_mean={self.icmean}',color='r',linestyle='dashed')
        ax1.set_xlabel('date')
        ax1.set_ylabel("Rank IC", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax2 = ax1.twinx()
        ax2.plot(ic.index, ic["RankIC_cum"], color="gray", linewidth=2, label="RankIC_cum")
        ax2.set_ylabel("Rank IC 累计值", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax1.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=45)
        plt.title(f"因子{self.factor_col} RankIC表现")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        savepath = f'factor_reports/{self.factor_col}'
        os.makedirs(savepath,exist_ok=True)
        plt.savefig(f"{savepath}/RankIC.png", dpi=300, bbox_inches="tight")
        plt.close()
    def calc_quantdf(self):
        factor_df = self.factordf.copy()
        datelist = factor_df.index.get_level_values(0)
        factor_df['_bins'] = pd.concat([pd.qcut(dfs[self.factor_col],q=10,labels=False,duplicates='drop') for _,dfs in factor_df.groupby(datelist)])
        self.quantfactordf = factor_df
        return
    def plot_bins(self):
        if self.quantfactordf is None:
            self.calc_quantdf()
        binsdf = self.quantfactordf.copy()
        grouped = binsdf.groupby([self.date_col, "_bins"]).mean()[self.return_col].unstack()
        cumreturns = (1+grouped).cumprod()
        colors = cm.get_cmap('tab10',self.quantile_num).colors
        plt.figure(figsize=(10,6))
        for i,col in enumerate(cumreturns.columns):
            plt.plot(cumreturns.index,cumreturns[col],label=f'第{col}组',color=colors[i])
        plt.legend()
        plt.xlabel('date')
        plt.ylabel('净值')
        plt.title(f'因子{self.factor_col} 分层回测净值')
        plt.grid(True)
        savepath = f'factor_reports/{self.factor_col}'
        os.makedirs(savepath,exist_ok=True)
        plt.savefig(f"{savepath}/layering.png", dpi=300, bbox_inches="tight")
        plt.close()
        return

