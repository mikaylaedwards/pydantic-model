import pandas as pd
import datetime
import joblib
from pathlib import Path
import os
import sys
from sklearn.linear_model import LinearRegression
from .meta_models import RegressionMeta


DEFAULT_OUTPUT_PATH = Path(__file__).parent / 'model_files'

class ModelArtifactOutput:
    def __init__(self, artifact_object = None, output_path = DEFAULT_OUTPUT_PATH):
        self._artifact_object = artifact_object
        self._output_path = output_path

    @property
    def output_path(self):
        print('Artifact output location: ')
        return self._output_path
        

    @output_path.setter
    def output_path(self, location):
        self._output_path = location

    @property
    def file_ending(self):
        print('File ending: ')
        return self._file_ending
    
    @file_ending.setter
    def file_ending(self, file_ending):
        self._file_ending = file_ending

    def set_file_ending_default(self):
        """
        Called if no file ending provided before calling serialize() on object. 
        Sets default file ending to be .json if object being serialized is of type str (not a model)
        """
        if isinstance(self._artifact_object, str):
            self._file_ending = '.json'
        else:
            self._file_ending = '.pkl'


    def _dump_object(self):
        pkl_path = Path(self.output_path / 'model'/ self.file_ending)
        with open(pkl_path):
            joblib.dump(self._artifact_object,pkl_path)
    
    def _write_object(self,filename):
        ## need to differentiate schema from model.json

        pass

    def serialize(self, filename=None):
        if not hasattr(self,_file_ending):
            self.set_file_ending_default()

        if self._file_ending == '.pkl':
            self._dump_object()
        else:
            self._write_object(filename=filename)
            

class RegressionArtifact(ModelArtifactOutput):
    def __init__(self, artifact_object = None, output_path = DEFAULT_OUTPUT_PATH):
        super().__init__(artifact_object= artifact_object, output_path=output_path)

    def make_serializable(self, artifact_object):
        artifact_object.__dict__['coef_'] = artifact_object.__dict__['coef_'].flatten().tolist() #ndarray: flatten first
        artifact_object.__dict__['_residues'] = artifact_object.__dict__['_residues'].tolist()
        artifact_object.__dict__['singular_'] = artifact_object.__dict__['singular_'].tolist()
        artifact_object.__dict__['intercept_'] = artifact_object.__dict__['intercept_'].tolist()

    def serialize_model_json(self, meta_obj):
        model_json = meta_obj.json(indent = 4)
        self.serialize(model_json)

    def serialize_binary(self,artifact_object):
        self.serialize(artifact_object)
        
    def serialize_schema(self, meta_obj):
        schema_file = meta_obj.schema_json(indent = 4)

    def serialize_outputs(self, meta_obj):
        """
        input = instance of Regression Meta
        """
        # first serialize to binary without modififcation
        self.serialize_binary(self.artifact_object)

        self.make_json_serializable(self.artifact_object)
        self.serialize_model_json()