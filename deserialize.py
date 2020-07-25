import json
import os
from pathlib import Path
from models import RegressionMeta
from sklearn.linear_model import LinearRegression
from utils import get_default_args
from contextlib import redirect_stdout
import pprint


__file__ = '{}/deserialize.py'.format(os.getcwd())

MODEL_FILES_PATH = Path(__file__).parent / 'model_files'


class RegressionLoader(LinearRegression):

    # Override LinearRegression constructor: args to constructor must be keyword for 0.25+
    def __init__(self, args, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        LinearRegression.__init__(self,
                                  fit_intercept=fit_intercept,
                                  normalize=normalize,
                                  copy_X=copy_X,
                                  n_jobs=n_jobs)
        # set all params not passed to LinReg constructor; created in modeling process (e.g. coef_, intercept_, etc.)
        for arg, value in args.items():
            setattr(self, arg, value)

    @classmethod
    def from_json(cls, model_json):
        """Loads in json file containing the serialized RegressionMeta class
        converted to json in serialize.py. Separates required args in
        LinearRegression construcor and passesthem to current class as keyword
        args.

        @param: model_json: json file with serialized model data
        """
        with open(model_json) as m:
            model_json = json.load(m)
        # re-construct instance of RegressionMeta from json
        meta = RegressionMeta.parse_obj(model_json)
        required_args = get_default_args(LinearRegression.__init__)
        req_args_ = {k: v for (k, v) in meta.params.items()
                     if k in required_args}
        non_req_args_ = {
            k: v for (k, v) in meta.params.items() if k not in required_args}
        # print(non_req_args_)
        return cls(non_req_args_, **req_args_)


model_json = Path(f'{MODEL_FILES_PATH}/model.json')

model = RegressionLoader.from_json(model_json=model_json)

with open(f'{MODEL_FILES_PATH}/deserialize_results.txt', 'w') as f:
    with redirect_stdout(f):
        print('Model parameters:')
        pprint.pprint(model.__dict__)
        print('Model methods:')
        pprint.pprint([x for x in dir(model) if not (x.startswith('__') or x in vars(model))])
