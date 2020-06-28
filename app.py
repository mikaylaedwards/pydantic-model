from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import joblib
from pathlib import Path
import os
import sys

# local import: import ModelClass Object created in models.py
cwd= os.getcwd()
sys.path.append(f'{cwd}/models.py')

from models import RegressionModel


diabetes = datasets.load_diabetes()
xdf = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
ydf =  pd.DataFrame(diabetes.target,columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

model_path = Path('./model_files/model.pkl')

with open(model_path):
    joblib.dump(model,model_path)

# convert numpy arrays to list for json serialization
model.__dict__['coef_'] = model.__dict__['coef_'].flatten().tolist() #ndarray: flatten first
model.__dict__['_residues'] = model.__dict__['_residues'].tolist()
model.__dict__['singular_'] = model.__dict__['singular_'].tolist()
model.__dict__['intercept_'] = model.__dict__['intercept_'].tolist()

mod_obj = RegressionModel(model_id='m1001',
                          timestamp=datetime.datetime.today(),
                          feature_names=X_train.columns.to_list(),
                          random_seed=None,
                          stored_path=model_path,
                          params=model.__dict__)

json_file = mod_obj.json()
with open('./model_files/model.json','w+') as jpath:
    jpath.write(json_file)

print(mod_obj.schema_json())
## run python3 model.py > schema.json


