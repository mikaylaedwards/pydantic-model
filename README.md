#### Overview
This project uses the [pydantic](https://github.com/samuelcolvin/pydantic) library to serialize model objects and associated metadata.

The stucture is simple:

**1.** Model metadata is stored in class objects that inherit from class BaseModel in pydantic. These are created in model.py. For example,
```
class RegressionMeta(BaseModel):
    model_id: str
    model_type = 'Linear Regression'
    timestamp: datetime.datetime
    feature_names: list
    random_seed: typing.Optional[int] = None
    stored_path: Path
    params: dict 
```
**2.**
Models are created in app.py, which imports the the metadata class from models
   
```
from models import RegressionMeta

## data imports & pre-processing ##

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
```
**3.** A storage path is then specified for the model using the pathlib library. (using pickle & .pkl file is optional--could use any storage format) 
```
PARENT_DIRECTORY = Path(__file__).parent

model_path = Path('{}/model_files/model.pkl'.format(PARENT_DIRECTORY))

with open(model_path):
    joblib.dump(model,model_path)
```
This ensures the model is stored in the mode_files directory. When app.py is run, model_path is pre-fixed to begin with the parent directory.

**4.** A model object with associated metadata is initialized. These attributes are just examples of what might be useful for reconstructing/evaluating the model at a later point, but additional desired attributes could also be added. 

```
mod_obj = RegressionMeta(model_id='m1001',
                          timestamp=datetime.datetime.today(),
                          feature_names=X_train.columns.to_list(),
                          random_seed=None,
                          stored_path=model_path,
                          params=model.__dict__)
```
**5.**
Because the model metadata class we created inherits methods from BaseModel, we can call .json directly on our instance of this class to create a json file containing its metadata. This can be stored in a specificed path (here initialized as json_path).
```
json_file = mod_obj.json()
json_path = Path('{}/model_files/model.json'.format(PARENT_DIRECTORY))

with open(json_path,'w+'):
    json_path.write(json_file)
```
The json file created will look this (shortened here but full version can be found in model_files/model.json):
```
{
    "model_id": "m1001",
    "timestamp": "2020-07-11T14:53:06.614924",
    "feature_names": [
        "age",
         ...
         ...
        "bmi",
         ...
    ],
    "random_seed": null,
    "stored_path": "model_files/model.pkl",
    "params": {
        "fit_intercept": true,
        "normalize": false,
        "copy_X": true,
        "n_jobs": null,
        "n_features_in_": 10,
        "coef_": [
            -35.556836741535015,
            ...
            ...
            562.7540463245917,
            ...
        ],
        "_residues": [
            965359.4331583094
        ],
        "rank_": 10,
        "singular_": [
            1.8374128023505603,
            ...
            ...
            0.2470579188369371,
            ...
        ],
        "intercept_": [
            152.53813351954062
        ]
    },
    "model_type": "Linear Regression"
}
```

**6.** Finally, a schema.json file is also created with the schema_json() method from BaseModel to facilitate future data validation when the model is deserialized and we attempt to use its metadata and/or make predictions.
```
schema_file = mod_obj.schema_json()
schema_path = Path('{}/model_files/schema.json'.format(PARENT_DIRECTORY))

with open(schema_path,'w+'):
    json_path.write(json_file)
```
The schema file looks like this:
```
{
    "title": "RegressionModel",
    "type": "object",
    "properties": {
        "model_id": {
            "title": "Model Id",
            "type": "string"
        },
        "timestamp": {
            "title": "Timestamp",
            "type": "string",
            "format": "date-time"
        },
        "feature_names": {
            "title": "Feature Names",
            "type": "array",
            "items": {}
        },
        "random_seed": {
            "title": "Random Seed",
            "type": "integer"
        },
        "stored_path": {
            "title": "Stored Path",
            "type": "string",
            "format": "path"
        },
        "params": {
            "title": "Params",
            "type": "object"
        },
        "model_type": {
            "title": "Model Type",
            "default": "Linear Regression",
            "type": "string"
        }
    },
    "required": [
        "model_id",
        "timestamp",
        "feature_names",
        "stored_path",
        "params"
    ]
}
```

### **Future Improvements**

In the future, I will refactor the creation of model artifacts (e.g. model.json file) and storage path objects to simplify the repeated logic. 

Also, the conversion of model attributes into json-serializable objects is currently done by indexing the model dictionary and converting each element to a list, but this can be refactored and improved to enable customization. 
For example, the below snippet flattens the numpy ndarray object containing the model coefficients and converts it to a list, as ndarrays and arrays can be directly converted to json.
```
# convert numpy arrays to list for json serialization
# ndarray must be flattened to one dimensional array first
model.__dict__['coef_'] = model.__dict__['coef_'].flatten().tolist()
```

