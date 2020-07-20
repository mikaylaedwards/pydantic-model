import json
import os
from pathlib import Path
from models import RegressionMeta
from sklearn.linear_model import LinearRegression
from utils import get_default_args


__file__ = '{}/deserialize.py'.format(os.getcwd())

PARENT_DIRECTORY = Path(__file__).parent

# TO-DO: Write class to override constructor of LinearRegression
# TO-DO: Add req args to json 

class RegressionLoader(LinearRegression):
    
    # Override LinearRegression constructor: In sklearn(0.23) these four arguments are required as positional arguments
    ## use kwargs in version 0.25+
    def __init__(self,kwargs, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        LinearRegression.__init__(self, fit_intercept, normalize, copy_X, n_jobs)
        self.coef_= kwargs.get('coef_')
        ## set all params from kwargs (non positional)
  
    @classmethod
    def from_json(cls, model_json):
        """
        @param: model_json: json file with serialized model data
        """
        with open(model_json) as m:
            model_dict = json.load(m)
        meta = RegressionMeta.parse_obj(model_dict)
        required_args = get_default_args(LinearRegression.__init__)
        req_args_ = {k:v for (k,v) in meta.params.items() if k in required_args}
        non_req = {k:v for (k,v) in meta.params.items() if k not in required_args}
        return cls(non_req,**req_args_)


model_json = Path('{}/model_files/model.json'.format(PARENT_DIRECTORY))

#required_args = get_default_args(LinearRegression.__init__)
model = RegressionLoader.from_json(model_json= model_json)
print(dir(model))