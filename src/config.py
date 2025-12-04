"""
Configurações centralizadas do projeto de predição de partidas de futebol
"""
import os

# ========== CAMINHOS DOS DADOS ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATA_PATHS = {
    'base': os.path.join(DATA_DIR, 'base_data'),
    'lineup': os.path.join(DATA_DIR, 'lineup_data'),
    'commentary': os.path.join(DATA_DIR, 'commentary_data'),
    'keyEvents': os.path.join(DATA_DIR, 'keyEvents_data'),
    'plays': os.path.join(DATA_DIR, 'plays_data'),
    'playerStats': os.path.join(DATA_DIR, 'playerStats_data')
}

# Arquivos base
BASE_FILES = {
    'fixtures': 'fixtures.csv',
    'leagues': 'leagues.csv',
    'standings': 'standings.csv',
    'status': 'status.csv',
    'teams': 'teams.csv',
    'teamStats': 'teamStats.csv',
    'teamRoster': 'teamRoster.csv',
    'players': 'players.csv',
    'venues': 'venues.csv',
    'keyEventDescription': 'keyEventDescription.csv'
}

# ========== CAMINHOS DE SAÍDA ==========
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

MODEL_FILES = {
    'model': os.path.join(MODELS_DIR, 'best_model.json'),
    'scaler': os.path.join(MODELS_DIR, 'scaler.pkl'),
    'features': os.path.join(MODELS_DIR, 'feature_columns.json'),
    'draw_threshold': os.path.join(MODELS_DIR, 'draw_threshold.json')
}

LOG_FILE = os.path.join(LOGS_DIR, 'training_log.txt')

# ========== PARÂMETROS DO MODELO ==========
MODEL_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'min_child_weight': 1,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}

# ========== CONFIGURAÇÕES DE TREINAMENTO ==========
TRAIN_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'early_stopping_rounds': 50
}

# ========== FEATURES DO MODELO ==========
# Features que serão criadas durante a engenharia de features
FEATURE_GROUPS = {
    'home_form': [
        'home_recent_wins',
        'home_recent_draws',
        'home_recent_losses',
        'home_form_points'
    ],
    'away_form': [
        'away_recent_wins',
        'away_recent_draws',
        'away_recent_losses',
        'away_form_points'
    ],
    'home_performance': [
        'home_goals_per_game',
        'home_goals_against_per_game',
        'home_goal_difference',
        'home_points',
        'home_wins',
        'home_losses',
        'home_draws'
    ],
    'away_performance': [
        'away_goals_per_game',
        'away_goals_against_per_game',
        'away_goal_difference',
        'away_points',
        'away_wins',
        'away_losses',
        'away_draws'
    ],
    'match_stats': [
        'home_possession_avg',
        'away_possession_avg',
        'home_pass_accuracy',
        'away_pass_accuracy',
        'home_shot_accuracy',
        'away_shot_accuracy'
    ],
    'lineup_quality': [
        'home_avg_age',
        'away_avg_age',
        'home_avg_height',
        'away_avg_height',
        'home_avg_weight',
        'away_avg_weight'
    ]
}

# Lista completa de features
ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group)

# ========== MAPEAMENTOS ==========
TARGET_MAPPING = {
    'draw': 0,
    'home_win': 1,
    'away_win': 2
}

FORM_POINTS = {
    'W': 3,  # Win
    'D': 1,  # Draw
    'L': 0   # Loss
}

# ========== CONVERSÕES DE UNIDADES ==========
LBS_TO_KG = 0.453592
FEET_TO_METERS = 0.3048
INCHES_TO_METERS = 0.0254

# ========== CONFIGURAÇÕES DE PROCESSAMENTO ==========
MISSING_VALUE_STRATEGY = {
    'numeric': 'mean',  # mean, median, zero
    'categorical': 'mode',
    'attendance': 0,
    'stats': 0
}

# Número de jogos para calcular forma recente
RECENT_GAMES_WINDOW = 5

# Status de partidas completas (Full Time)
COMPLETED_STATUS = [28, 30]  # 28 = Full Time, 30 = After Extra Time

# ========== CONFIGURAÇÕES DE LOG ==========
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ========== CRIAR DIRETÓRIOS SE NÃO EXISTIREM ==========
for directory in [MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)