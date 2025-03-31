class Model:
    def fit(self,dataset):
         raise NotImplementedError()
    def batch_fit(self,dataset):
         raise NotImplementedError()
    def predict(self,dataset):
         raise NotImplementedError()