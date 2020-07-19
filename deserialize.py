import json
from pathlib import Path
from models import RegressionMeta
from sklearn.linear_model import LinearRegression


__file__ = '/home/mikayla/Documents/pydantic-model/deserialize.py'

PARENT_DIRECTORY = Path(__file__).parent

model_json = Path('{}/model_files/model.json'.format(PARENT_DIRECTORY))
with open(model_json) as m:
    model_dict = json.load(m)
# construct, parse_file, parse_raw

rm = RegressionMeta.parse_obj(model_dict)
print(rm.params)

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
    def from_json(cls, model_json, req_args):
        with open(model_json) as m:
            model_dict = json.load(m)
        meta = RegressionMeta.parse_obj(model_dict)
        req_args_ = {k:v for (k,v) in meta.params.items() if k in req_args}
        non_req = {k:v for (k,v) in meta.params.items() if k not in req_args}
        return cls(non_req,**req_args_)


req_args = ['fit_intercept','normalize','copy_X','n_jobs']
model = RegressionLoader.from_json(model_json= model_json, req_args=req_args)
print(model.coef_)
#print(dir(model))



# md = {'fit_intercept': True, 'normalize': False, 'copy_X': True, 'n_jobs': None}
# RegressionLoader(**md)