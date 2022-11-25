import numpy as np
import xgboost as xgb

class XGBHelper:

    def __init__(self, model_file, var_list):
        self.bst = xgb.Booster(params={'nthread': 1}, model_file=model_file)
        self.var_list = var_list
        print('Load XGBoost model %s, input variables:\n  %s' % (model_file, str(var_list)))

    def eval(self, inputs):
        dmat = xgb.DMatrix(np.array([inputs[k] for k in self.var_list]).T, feature_names=self.var_list)
        return self.bst.predict(dmat)

class XGBEnsemble:

    def __init__(self, model_files, var_list):
        self.bst_list = [xgb.Booster(params={'nthread': 1}, model_file=f) for f in model_files]
        self.var_list = var_list
        print('Load XGBoost models:\n  %s, \ninput variables:\n  %s' % ('\n  '.join(model_files), str(var_list)))

    def eval(self, inputs):
        dmat = xgb.DMatrix(np.array([inputs[k] for k in self.var_list]).T, feature_names=self.var_list)
        preds = np.array([bst.predict(dmat) for bst in self.bst_list])
        return preds.sum(axis=0) / len(self.bst_list)