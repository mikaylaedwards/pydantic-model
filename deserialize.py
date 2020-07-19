import json
from pathlib import Path
from models import RegressionMeta


PARENT_DIRECTORY = Path(__file__).parent

model_json = Path('{}/model_files/model.json'.format(PARENT_DIRECTORY))
with open(model_json) as m:
    model_dict = json.load(m)
# construct, parse_file, parse_raw

rm = RegressionMeta.parse_obj(model_dict)
print(rm.__dict__)

# TO-DO: Write class to override constructor of LinearRegression

