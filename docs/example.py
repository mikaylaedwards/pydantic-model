from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
from scikit_serialize.serialize import RegressionArtifact
from scikit_serialize.meta_models import RegressionMeta
from pydantic import Json 

diabetes = datasets.load_diabetes()
xdf = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
ydf =  pd.DataFrame(diabetes.target,columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# instantiate model artifact with created model
model_artifact = RegressionArtifact(artifact_object = model)

# set output path to be model files in project home (up one directory) /model_files
model_artifact.output_path = Path().cwd() / 'model_files'


mod_obj = RegressionMeta(model_id='m1001',
                          timestamp=datetime.datetime.today(),
                          feature_names=X_train.columns.to_list(),
                          random_seed=None,
                          stored_path=model_artifact.output_path,
                          params=model.__dict__)

