"""
Pacote src - Módulos do projeto de predição de futebol
"""

__version__ = '1.0.0'
__author__ = 'Soccer Prediction Team'

# Facilitar imports
from .etl import load_and_preprocess_data, SoccerDataLoader
from .feature_engineering import create_features, FeatureEngineer
from .model_xgboost import train_and_evaluate, SoccerPredictor
from .predict import predict_new_matches, MatchPredictor, predict_from_team_ids
from .utils import logger

__all__ = [
    'load_and_preprocess_data',
    'SoccerDataLoader',
    'create_features',
    'FeatureEngineer',
    'train_and_evaluate',
    'SoccerPredictor',
    'predict_new_matches',
    'MatchPredictor',
    'predict_from_team_ids',
    'logger'
]