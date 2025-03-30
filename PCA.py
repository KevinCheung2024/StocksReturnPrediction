# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.stats import bartlett
from factor_analyzer import FactorAnalyzer
from scipy.stats import zscore

plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False

# %%

class PCA_analysis:
    def __init__(self,data0):
        self.data = data0.copy()
    
    def run(self):
        self.KMO()
        self.run_model()
        self.var_exp()
        self.scree_plot()
        self.components()
        self.jiangwei()
        self.PCA_exp()
        return "主成分分析已完成"
    def KMO(self):
        corr=list(self.data.corr().to_numpy())
        print('KMO和巴特勒球形度检验',bartlett(*corr))
        self.bartlett = bartlett(*corr)
        return
    def run_model(self):
        self.model= PCA().fit(self.data)
        (self.m,self.n) = self.data.shape
        #调整主成分系数的符号，使得主成分为正
        xml = self.model.components_
        check = xml.sum(axis =1, keepdims = True)
        self.xs2 = xml*np.sign(check)
        return
    def var_exp(self):
        var_explanation = pd.DataFrame((self.model.explained_variance_.T ,self.model.explained_variance_ratio_.T ,  np.cumsum(self.model.explained_variance_ratio_).T))
        var_explanation = var_explanation.T
        index = []
        for i in range(1,self.n+1):
            index.append("主成分"+str(i))
        self.index = index
        var_explanation.index = index
        var_explanation.columns = ['特征根','方差百分比','累积']
        print('方差解释表')
        print(var_explanation)
        self.var_explanation = var_explanation
        return
    def scree_plot(self):
        #复现碎石图
        fig,ax  = plt.subplots()
        ax.plot(np.arange(1,self.n+1),self.model.explained_variance_, marker = "o")
        ax.set_xlabel("因子个数")
        ax.set_ylabel("特征值")
        ax.set_title('碎石图')
        plt.legend()
        plt.show()
        #选择主成分个数
        print("请输入选取的主成分数目：")
        self.k = eval(input())
        return
    def components(self):
        #复现成分矩阵表
        components = pd.DataFrame(self.xs2)
        components.index = self.index
        components.columns = [('x' + str(i)) for i in range(1,len(components.columns) + 1)]
        print("成分矩阵表")
        print(components)
        self.components = components
        return
    def jiangwei(self):
        #取降维后的数据
        jiangwei = self.data @ self.components.iloc[0:self.k,:].T.values
        jiangwei = pd.DataFrame(jiangwei)
        jiangwei.columns = self.index[0:self.k]
        print('降维后数据')
        print(jiangwei)
        self.data_dr = jiangwei
        return
    def PCA_exp(self):
        #主成分解释
        fa = FactorAnalyzer(n_factors=self.k,method='principal',rotation="varimax")
        fa.fit(self.data)
        communalities= fa.get_communalities()#共性因子方差
        loadings=fa.loadings_#成分矩阵，可以看出特征的归属因子
        plt.figure()
        ax = sns.heatmap(loadings, annot=True, cmap="BuPu")
        plt.title('Factor Analysis')
        return
# def PCA_analysis(data0):
#     #输入变量是一个dataframe
#     data = zscore(data0, ddof = 1) #数据标准化
#     corr=list(data0.corr().to_numpy())
#     print('KMO和巴特勒球形度检验',bartlett(*corr))
#     model= PCA().fit(data)
#     (m,n) = data0.shape
    
#     #调整主成分系数的符号，使得主成分为正
#     xml = model.components_
#     check = xml.sum(axis =1, keepdims = True)
#     xs2 = xml*np.sign(check)
    
#     var_explanation = pd.DataFrame((model.explained_variance_.T ,model.explained_variance_ratio_.T ,  np.cumsum(model.explained_variance_ratio_).T))
#     var_explanation = var_explanation.T
#     index = []
#     for i in range(1,n+1):
#         index.append("主成分"+str(i))
#     var_explanation.index = index
#     var_explanation.columns = ['特征根','方差百分比','累积']
#     print('方差解释表')
#     print(var_explanation)
    
#     #复现碎石图
#     fig,ax  = plt.subplots()
#     ax.plot(np.arange(1,n+1),model.explained_variance_, marker = "o")
#     ax.set_xlabel("因子个数")
#     ax.set_ylabel("特征值")
#     ax.set_title('碎石图')
#     plt.legend()
#     plt.show()
    
#     print("请输入选取的主成分数目：")
#     k = eval(input())
    
#     #复现成分矩阵表
#     components = pd.DataFrame(xs2)
#     components.index = index
#     components.columns = [('x' + str(i)) for i in range(1,len(components.columns) + 1)]
#     print("成分矩阵表")
#     print(components)
    
#     #取降维后的数据
#     jiangwei = data @ components.iloc[0:k,:].T.values
#     jiangwei = pd.DataFrame(jiangwei)
#     jiangwei.columns = index[0:k]
#     print('降维后数据')
#     print(jiangwei)
    
#     #主成分解释
#     fa = FactorAnalyzer(n_factors=k,method='principal',rotation="varimax")
#     fa.fit(data)
#     communalities= fa.get_communalities()#共性因子方差
#     loadings=fa.loadings_#成分矩阵，可以看出特征的归属因子
#     plt.figure()
#     ax = sns.heatmap(loadings, annot=True, cmap="BuPu")
#     plt.title('Factor Analysis')
    
#     return "主成分分析已完成"


# %%


