from pydantic import BaseModel
from pydantic.types import FilePath
import typing
from pathlib import Path
import datetime

__all__ =['RegressionMeta']

class RegressionMeta(BaseModel):
    model_id: str
    model_type = 'Linear Regression'
    timestamp: datetime.datetime
    feature_names: list
    random_seed: typing.Optional[int] = None
    stored_path: Path
    params: dict 