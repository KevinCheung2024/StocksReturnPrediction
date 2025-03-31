from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import make_scorer, r2_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
class BaseTuneParams:
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def GridSearchTune(self,model,params_dicts,cv=5):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params_dicts,
            scoring=make_scorer(r2_score),
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(self.X, self.y)
        return grid_search
    @staticmethod
    def BysOpt(func,params_dicts,init_points=1,n_iter=20):
        bayes = BayesianOptimization(
            func,
            params_dicts
        )
        bayes.maximize(init_points=init_points,n_iter=n_iter)    
        return bayes