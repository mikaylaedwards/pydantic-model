from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from pydantic import BaseModel
from pydantic.types import FilePath
import datetime
import joblib
import typing
from pathlib import Path



class RegressionModel(BaseModel):
    model_id: str
    model_type = 'Linear Regression'
    timestamp: datetime.datetime
    feature_names: list
    random_seed: typing.Optional[int] = None
    stored_path: Path
    params: dict 





diabetes = datasets.load_diabetes()
xdf = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
ydf =  pd.DataFrame(diabetes.target,columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

model_path = Path('./model_files/model.pkl')
with open(model_path):
    joblib.dump(model,model_path)

mod_obj = RegressionModel(model_id='m1001',
                          timestamp=datetime.datetime.today(),
                          feature_names=X_train.columns.to_list(),
                          random_seed=None,
                          stored_path=model_path,
                          params=model.__dict__)

print(mod_obj.schema_json())

## run python3 model.py > schema.json


