"""
Soccer Match Outcome Predictor - Sistema Otimizado
Prediz resultados de partidas (Home Win / Draw / Away Win)
Com validação temporal, otimização avançada e métricas de negócio
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, matthews_corrcoef)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import warnings
import os
import optuna
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import json

warnings.filterwarnings('ignore')

class SoccerMatchPredictor:
    """
    Preditor de resultados de partidas de futebol com otimizações
    Target: 0 = Away Win, 1 = Draw, 2 = Home Win
    """
    
    def __init__(self, base_path: str, experiment_name: str = None):
        self.base_path = base_path
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_logging()
        self.best_thresholds = None
        self.models = {}
        
    def setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(f'{self.experiment_name}.log', encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # Console handler com encoding utf-8
            import sys
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            
            # Formatter simples sem emojis
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def load_data(self, cache: bool = True):
        """Carrega e pré-processa os dados"""
        self.logger.info("=" * 70)
        self.logger.info("CARREGANDO DADOS DO DATASET")
        self.logger.info("=" * 70)
        
        cache_file = os.path.join(self.base_path, 'processed_data.pkl')
        
        if cache and os.path.exists(cache_file):
            self.logger.info("Carregando dados do cache...")
            data = joblib.load(cache_file)
            self.fixtures, self.standings, self.team_stats, self.teams, self.leagues = data
        else:
            self.logger.info("Carregando arquivos base...")
            
            # Verificar se o diretório existe
            base_data_path = os.path.join(self.base_path, 'base_data')
            if not os.path.exists(base_data_path):
                self.logger.error(f"Diretório não encontrado: {base_data_path}")
                raise FileNotFoundError(f"Diretório não encontrado: {base_data_path}")
            
            # Listar arquivos disponíveis para debug
            files = os.listdir(base_data_path)
            self.logger.info(f"Arquivos encontrados: {files}")
            
            # Carregar os arquivos CSV
            try:
                self.fixtures = pd.read_csv(os.path.join(base_data_path, 'fixtures.csv'))
                self.leagues = pd.read_csv(os.path.join(base_data_path, 'leagues.csv'))
                self.teams = pd.read_csv(os.path.join(base_data_path, 'teams.csv'))
                self.standings = pd.read_csv(os.path.join(base_data_path, 'standings.csv'))
                self.team_stats = pd.read_csv(os.path.join(base_data_path, 'teamStats.csv'))
            except Exception as e:
                self.logger.error(f"Erro ao carregar arquivos: {e}")
                raise
            
            # Pré-processamento
            self.preprocess_data()
            
            if cache:
                joblib.dump([self.fixtures, self.standings, self.team_stats, 
                           self.teams, self.leagues], cache_file)
        
        self.analyze_data_distribution()
    
    def preprocess_data(self):
        """Pré-processa os dados"""
        self.logger.info("Pré-processando dados...")
        
        # Converter datas
        if 'date' in self.fixtures.columns:
            self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
        else:
            self.logger.warning("Coluna 'date' não encontrada em fixtures")
        
        # Criar colunas auxiliares
        if 'date' in self.fixtures.columns and self.fixtures['date'].dtype == 'datetime64[ns]':
            self.fixtures['year'] = self.fixtures['date'].dt.year
            self.fixtures['month'] = self.fixtures['date'].dt.month
            self.fixtures['day_of_week'] = self.fixtures['date'].dt.dayofweek
        
        # Criar dicionários para fácil acesso - ESTA LINHA DEVE EXISTIR
        self._create_dictionaries()
        
        # Criar coluna de resultado
        if 'homeTeamScore' in self.fixtures.columns and 'awayTeamScore' in self.fixtures.columns:
            def get_result(row):
                if pd.isna(row['homeTeamScore']) or pd.isna(row['awayTeamScore']):
                    return None
                if row['homeTeamScore'] > row['awayTeamScore']:
                    return 'H'
                elif row['homeTeamScore'] < row['awayTeamScore']:
                    return 'A'
                else:
                    return 'D'
            
            self.fixtures['result'] = self.fixtures.apply(get_result, axis=1)
        
        # Remover jogos futuros (sem resultado)
        if 'result' in self.fixtures.columns:
            self.fixtures = self.fixtures.dropna(subset=['result'])
    
    def _create_dictionaries(self):
        """Cria dicionários para acesso rápido aos nomes dos times e ligas"""
        self.logger.info("Criando dicionários de times e ligas...")
        
        # Para times
        team_dict_created = False
        if hasattr(self, 'teams') and 'teamId' in self.teams.columns:
            # Encontrar coluna de nome
            name_cols = ['name', 'displayName', 'slug', 'shortName', 'fullName']
            for col in name_cols:
                if col in self.teams.columns:
                    self.team_dict = dict(zip(self.teams['teamId'], self.teams[col]))
                    self.logger.info(f"Dicionário de times criado usando coluna: {col}")
                    self.logger.info(f"  Total de times no dicionário: {len(self.team_dict)}")
                    team_dict_created = True
                    break
            
            if not team_dict_created:
                # Usar primeira coluna de string
                text_cols = [col for col in self.teams.columns 
                        if self.teams[col].dtype == 'object' and col != 'teamId']
                if text_cols:
                    self.team_dict = dict(zip(self.teams['teamId'], self.teams[text_cols[0]]))
                    self.logger.info(f"Dicionário de times criado usando coluna: {text_cols[0]}")
                    self.logger.info(f"  Total de times no dicionário: {len(self.team_dict)}")
                    team_dict_created = True
        
        if not team_dict_created:
            self.team_dict = {}
            self.logger.warning("Não foi possível criar dicionário de times")
        
        # Para ligas
        league_dict_created = False
        if hasattr(self, 'leagues') and 'leagueId' in self.leagues.columns:
            # Encontrar coluna de nome
            name_cols = ['leagueName', 'name', 'midsizeName', 'slug', 'shortName']
            for col in name_cols:
                if col in self.leagues.columns:
                    self.league_dict = dict(zip(self.leagues['leagueId'], self.leagues[col]))
                    self.logger.info(f"Dicionário de ligas criado usando coluna: {col}")
                    self.logger.info(f"  Total de ligas no dicionário: {len(self.league_dict)}")
                    league_dict_created = True
                    break
            
            if not league_dict_created:
                # Usar primeira coluna de string
                text_cols = [col for col in self.leagues.columns 
                        if self.leagues[col].dtype == 'object' and col != 'leagueId']
                if text_cols:
                    self.league_dict = dict(zip(self.leagues['leagueId'], self.leagues[text_cols[0]]))
                    self.logger.info(f"Dicionário de ligas criado usando coluna: {text_cols[0]}")
                    self.logger.info(f"  Total de ligas no dicionário: {len(self.league_dict)}")
                    league_dict_created = True
        
        if not league_dict_created:
            self.league_dict = {}
            self.logger.warning("Não foi possível criar dicionário de ligas")
    
    def analyze_data_distribution(self):
        """Analisa distribuição dos dados"""
        completed = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) & 
            (self.fixtures['awayTeamScore'].notna())
        ]
        
        self.logger.info("\nVISÃO GERAL:")
        if 'date' in self.fixtures.columns and len(self.fixtures) > 0:
            self.logger.info(f"   Período: {self.fixtures['date'].min().date()} até {self.fixtures['date'].max().date()}")
        self.logger.info(f"   Partidas completas: {len(completed):,} ({len(completed)/len(self.fixtures)*100:.1f}%)")
        
        # Distribuição por liga
        if len(completed) > 0 and 'leagueId' in completed.columns and hasattr(self, 'league_dict'):
            league_counts = completed['leagueId'].value_counts().head(10)
            self.logger.info("\nTOP 10 Ligas:")
            for league_id, count in league_counts.items():
                league_name = self.league_dict.get(league_id, f"League {league_id}")
                self.logger.info(f"   {league_name}: {count:,} partidas")
    
    def create_target(self, row: pd.Series, raw: bool = False) -> Optional[Union[int, str]]:
        """Cria a variável target"""
        if pd.isna(row['homeTeamScore']) or pd.isna(row['awayTeamScore']):
            return None
        
        home_score = row['homeTeamScore']
        away_score = row['awayTeamScore']
        
        if raw:
            # Retorna string para análise
            if home_score > away_score:
                return 'H'
            elif home_score < away_score:
                return 'A'
            else:
                return 'D'
        
        # Retorno numérico para modelo
        if home_score > away_score:
            return 2  # Home Win
        elif home_score < away_score:
            return 0  # Away Win
        else:
            return 1  # Draw
    
    def get_recent_form(self, team_id: int, current_date: pd.Timestamp, 
                       n_games: int = 5) -> Dict:
        """Calcula forma recente do time"""
        if not hasattr(self, 'fixtures'):
            return {'ppg': 0, 'gf': 0, 'ga': 0, 'win_rate': 0, 'games': 0}
        
        recent_matches = self.fixtures[
            ((self.fixtures['homeTeamId'] == team_id) | 
             (self.fixtures['awayTeamId'] == team_id)) &
            (self.fixtures['date'] < current_date) &
            (self.fixtures['homeTeamScore'].notna()) &
            (self.fixtures['awayTeamScore'].notna())
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(recent_matches) == 0:
            return {'ppg': 0, 'gf': 0, 'ga': 0, 'win_rate': 0, 'games': 0}
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = 0
        
        for _, match in recent_matches.iterrows():
            is_home = match['homeTeamId'] == team_id
            
            if is_home:
                goals_scored += match['homeTeamScore']
                goals_conceded += match['awayTeamScore']
                if match['homeTeamScore'] > match['awayTeamScore']:
                    points += 3
                    wins += 1
                elif match['homeTeamScore'] == match['awayTeamScore']:
                    points += 1
            else:
                goals_scored += match['awayTeamScore']
                goals_conceded += match['homeTeamScore']
                if match['awayTeamScore'] > match['homeTeamScore']:
                    points += 3
                    wins += 1
                elif match['awayTeamScore'] == match['homeTeamScore']:
                    points += 1
        
        games_played = len(recent_matches)
        return {
            'ppg': points / games_played,
            'gf': goals_scored / games_played,
            'ga': goals_conceded / games_played,
            'win_rate': wins / games_played,
            'games': games_played
        }
    
    def get_h2h_stats(self, home_id: int, away_id: int, 
                     current_date: pd.Timestamp, n_games: int = 10) -> Dict:
        """Estatísticas de confrontos diretos"""
        if not hasattr(self, 'fixtures'):
            return {'home_win_rate': 0.33, 'draw_rate': 0.33, 'games': 0}
        
        h2h = self.fixtures[
            (((self.fixtures['homeTeamId'] == home_id) & (self.fixtures['awayTeamId'] == away_id)) |
             ((self.fixtures['homeTeamId'] == away_id) & (self.fixtures['awayTeamId'] == home_id))) &
            (self.fixtures['date'] < current_date) &
            (self.fixtures['homeTeamScore'].notna()) &
            (self.fixtures['awayTeamScore'].notna())
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(h2h) == 0:
            return {'home_win_rate': 0.33, 'draw_rate': 0.33, 'games': 0}
        
        home_wins = 0
        draws = 0
        
        for _, match in h2h.iterrows():
            if match['homeTeamId'] == home_id:
                if match['homeTeamScore'] > match['awayTeamScore']:
                    home_wins += 1
                elif match['homeTeamScore'] == match['awayTeamScore']:
                    draws += 1
            else:
                if match['awayTeamScore'] > match['homeTeamScore']:
                    home_wins += 1
                elif match['awayTeamScore'] == match['homeTeamScore']:
                    draws += 1
        
        games = len(h2h)
        return {
            'home_win_rate': home_wins / games,
            'draw_rate': draws / games,
            'games': games
        }
    
    def get_team_stats_features(self, team_id: int, season_type: str) -> Dict:
        """Pega estatísticas do time"""
        if not hasattr(self, 'team_stats'):
            return {}
        
        stats = self.team_stats[
            (self.team_stats['teamId'] == team_id) &
            (self.team_stats['seasonType'] == season_type)
        ]
        
        if len(stats) == 0:
            return {}
        
        stat = stats.iloc[0]
        features = {}
        
        # Adicionar features disponíveis
        possible_features = [
            'possessionPct', 'foulsCommitted', 'yellowCards', 'redCards',
            'offsides', 'wonCorners', 'saves', 'totalShots', 'shotsOnTarget',
            'goals', 'passAccuracy', 'aerialsWon', 'tackles', 'interceptions'
        ]
        
        for feat in possible_features:
            if feat in stat:
                features[feat] = stat[feat]
        
        return features
    
    def get_standings_features(self, team_id: int, league_id: int, 
                              season_type: str) -> Dict:
        """Pega features da tabela de classificação"""
        if not hasattr(self, 'standings'):
            return {}
        
        standing = self.standings[
            (self.standings['teamId'] == team_id) &
            (self.standings['leagueId'] == league_id) &
            (self.standings['seasonType'] == season_type)
        ]
        
        if len(standing) == 0:
            return {}
        
        s = standing.iloc[0]
        total_games = max(s.get('gamesPlayed', 1), 1)
        
        goals_for = s.get('gf', 0)
        goals_against = s.get('ga', 0)
        
        return {
            'team_rank': s.get('teamRank', 10),
            'points': s.get('points', 0),
            'games_played': total_games,
            'wins': s.get('wins', 0),
            'losses': s.get('losses', 0),
            'ties': s.get('ties', 0),
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': s.get('gd', 0),
            'points_per_game': s.get('points', 0) / total_games,
            'win_rate': s.get('wins', 0) / total_games,
            'avg_goals_for': goals_for / total_games,
            'avg_goals_against': goals_against / total_games
        }
    
    def engineer_features(self, sample_size: Optional[int] = None, 
                         min_games_history: int = 5) -> None:
        """Feature Engineering"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("ENGENHARIA DE FEATURES")
        self.logger.info("=" * 70)
        
        # Filtrar partidas finalizadas
        completed_matches = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) &
            (self.fixtures['awayTeamScore'].notna())
        ].copy()
        
        self.logger.info(f"Partidas finalizadas: {len(completed_matches):,}")
        
        # Ordenar por data para split temporal
        completed_matches = completed_matches.sort_values('date').reset_index(drop=True)
        
        # Amostragem se necessário
        if sample_size and sample_size < len(completed_matches):
            if len(completed_matches) > sample_size * 3:
                third = len(completed_matches) // 3
                indices = list(range(0, third, third//(sample_size//3))) + \
                         list(range(third, 2*third, third//(sample_size//3))) + \
                         list(range(2*third, len(completed_matches), 
                                   (len(completed_matches)-2*third)//(sample_size//3)))
                indices = indices[:sample_size]
                completed_matches = completed_matches.iloc[indices]
            else:
                completed_matches = completed_matches.head(sample_size)
        
        self.logger.info(f"Processando {len(completed_matches):,} partidas...")
        
        features_list = []
        targets = []
        dates = []
        
        for idx, match in completed_matches.iterrows():
            try:
                features = self._create_match_features(match, min_games_history)
                if features:
                    features_list.append(features)
                    targets.append(self.create_target(match))
                    dates.append(match['date'])
            except Exception as e:
                if idx % 1000 == 0:
                    self.logger.warning(f"Erro processando partida {idx}: {e}")
                continue
        
        # Criar DataFrame
        self.X = pd.DataFrame(features_list)
        self.y = np.array(targets)
        self.match_dates = np.array(dates)
        
        self.logger.info(f"✅ {len(self.X):,} partidas com features completas")
        
        # Verificar distribuição
        self._analyze_final_distribution()
        
        # Preencher NaN
        self.X = self.X.fillna(self.X.median())
        self.feature_columns = self.X.columns.tolist()
        
        self.logger.info(f"Total de features criadas: {len(self.feature_columns)}")
    
    def _create_match_features(self, match: pd.Series, min_games_history: int) -> Optional[Dict]:
        """Cria features para uma partida específica"""
        home_id = match['homeTeamId']
        away_id = match['awayTeamId']
        league_id = match['leagueId']
        season_type = match.get('seasonType', 'Regular Season')
        current_date = match['date']
        
        features = {}
        
        # Forma recente
        home_form = self.get_recent_form(home_id, current_date, n_games=5)
        away_form = self.get_recent_form(away_id, current_date, n_games=5)
        
        if home_form['games'] < min_games_history or away_form['games'] < min_games_history:
            return None
        
        features['home_recent_ppg'] = home_form['ppg']
        features['home_recent_gf'] = home_form['gf']
        features['home_recent_ga'] = home_form['ga']
        features['home_recent_win_rate'] = home_form['win_rate']
        
        features['away_recent_ppg'] = away_form['ppg']
        features['away_recent_gf'] = away_form['gf']
        features['away_recent_ga'] = away_form['ga']
        features['away_recent_win_rate'] = away_form['win_rate']
        
        # H2H
        h2h = self.get_h2h_stats(home_id, away_id, current_date)
        features['h2h_home_win_rate'] = h2h['home_win_rate']
        features['h2h_draw_rate'] = h2h['draw_rate']
        features['h2h_matches'] = h2h['games']
        
        # Standings
        home_standing = self.get_standings_features(home_id, league_id, season_type)
        away_standing = self.get_standings_features(away_id, league_id, season_type)
        
        if not home_standing or not away_standing:
            return None
        
        for key, value in home_standing.items():
            features[f'home_{key}'] = value
        for key, value in away_standing.items():
            features[f'away_{key}'] = value
        
        # Team stats
        home_stats = self.get_team_stats_features(home_id, season_type)
        away_stats = self.get_team_stats_features(away_id, season_type)
        
        for key, value in home_stats.items():
            features[f'home_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_{key}'] = value
        
        # Features derivadas
        if home_standing and away_standing:
            features['rank_diff'] = away_standing.get('team_rank', 10) - home_standing.get('team_rank', 10)
            features['points_diff'] = home_standing.get('points', 0) - away_standing.get('points', 0)
            features['goal_diff_diff'] = home_standing.get('goal_diff', 0) - away_standing.get('goal_diff', 0)
            features['form_diff'] = features['home_recent_ppg'] - features['away_recent_ppg']
        
        return features
    
    def _analyze_final_distribution(self):
        """Analisa distribuição final"""
        self.logger.info("\nDISTRIBUICAO FINAL do target:")
        unique, counts = np.unique(self.y, return_counts=True)
        total = len(self.y)
        
        result_names = ['Away Win', 'Draw', 'Home Win']
        for val, count in zip(unique, counts):
            result_name = result_names[val]
            percentage = count/total*100
            self.logger.info(f"   {result_name} ({val}): {count:,} ({percentage:.1f}%)")

    def evaluate_with_cv(self):
        """Avaliação com validação cruzada temporal"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("AVALIACAO COM VALIDACAO CRUZADA TEMPORAL")
        self.logger.info("=" * 70)
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Normalizar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Treinar
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=30,
                min_samples_leaf=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
            
            self.logger.info(f"  Fold {fold+1}: Acurácia = {score:.3f}")
        
        self.logger.info(f"\n📊 Acurácia média CV: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
        return cv_scores
    
    def train_ensemble(self, n_trials: int = 20, use_optuna: bool = True):
        """Treina ensemble de modelos"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TREINAMENTO ENSEMBLE")
        self.logger.info("=" * 70)
        
        # Treinar modelos simples sem Optuna para testes
        self.logger.info("\nTreinando Random Forest...")
        # Substitua os parâmetros atuais por:
        rf = RandomForestClassifier(
            n_estimators=300,      # Mais árvores
            max_depth=6,           # Menor profundidade para reduzir overfitting
            min_samples_split=30,  # Mais amostras para dividir
            min_samples_leaf=15,   # Mais amostras por folha
            max_features=0.5,      # Usar apenas 50% das features
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Split temporal
        split_idx = int(len(self.X) * 0.8)
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]
        
        self.logger.info(f"Treino: {len(X_train):,} partidas")
        self.logger.info(f"Teste: {len(X_test):,} partidas")
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar
        rf.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred_train = rf.predict(X_train_scaled)
        y_pred_test = rf.predict(X_test_scaled)
        y_proba_test = rf.predict_proba(X_test_scaled)
        
        # Métricas
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        self.logger.info(f"\nRESULTADOS DO MODELO:")
        self.logger.info("=" * 70)
        self.logger.info(f"Acurácia Treino: {train_acc:.1%}")
        self.logger.info(f"Acurácia Teste: {test_acc:.1%}")
        
        # Relatório de classificação
        target_names = ['Away Win', 'Draw', 'Home Win']
        self.logger.info("\nRelatório Detalhado:")
        self.logger.info("-" * 70)
        
        # Verificar classes presentes
        unique_test = np.unique(y_test)
        unique_pred = np.unique(y_pred_test)
        labels_present = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
        names_present = [target_names[i] for i in labels_present]
        
        report = classification_report(y_test, y_pred_test, labels=labels_present, 
                                     target_names=names_present, digits=3, zero_division=0)
        self.logger.info(report)
        
        # Matriz de confusão
        self.logger.info("\nMatriz de Confusão:")
        self.logger.info("-" * 70)
        cm = confusion_matrix(y_test, y_pred_test, labels=labels_present)
        
        # Feature importance
        self.logger.info("\nTop 15 Features Mais Importantes:")
        self.logger.info("-" * 70)
        
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(15).iterrows():
            self.logger.info(f"   {row['feature']:30s} {row['importance']:.4f}")
        
        self.model = rf
        self.results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test,
            'feature_importances': importances
        }
        
        return self.results
    
    def feature_importance_analysis(self, top_n: int = 20):
        """Análise da importância das features"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("ANALISE DE IMPORTANCIA DAS FEATURES")
        self.logger.info("=" * 70)
        
        if not hasattr(self, 'model'):
            self.logger.error("Modelo nao treinado. Treine o modelo primeiro.")
            return None
        
        try:
            # Verificar se o modelo tem feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Para modelos lineares
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                self.logger.error("Modelo nao suporta analise de importancia de features")
                return None
            
            # Criar DataFrame com importâncias
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"\nTOP {top_n} FEATURES MAIS IMPORTANTES:")
            self.logger.info("-" * 70)
            
            for idx, row in importance_df.head(top_n).iterrows():
                # Criar barra visual
                bar_length = int(row['importance'] / importance_df['importance'].max() * 50)
                bar = '█' * bar_length
                self.logger.info(f"   {row['feature']:40s} {bar} {row['importance']:.4f}")
            
            self.logger.info(f"\n📊 Resumo:")
            self.logger.info(f"   Total de features: {len(importance_df)}")
            self.logger.info(f"   Feature mais importante: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f})")
            self.logger.info(f"   Feature menos importante: {importance_df.iloc[-1]['feature']} ({importance_df.iloc[-1]['importance']:.4f})")
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Erro na analise de importancia: {e}")
            return None
    
    def predict_match(self, home_team_id: int, away_team_id: int, 
                     league_id: int, season_type: str = 'Regular Season',
                     match_date: Optional[pd.Timestamp] = None):
        """Prediz o resultado de uma partida específica"""
        if match_date is None:
            match_date = pd.Timestamp.now()
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FAZENDO PREDICAO")
        self.logger.info("=" * 70)
        
        # Buscar nomes dos times
        home_name = self.team_dict.get(home_team_id, f"Team {home_team_id}")
        away_name = self.team_dict.get(away_team_id, f"Team {away_team_id}")
        
        self.logger.info(f"\nPartida: {home_name} vs {away_name}")
        
        # Criar features para predição
        match_features = {
            'homeTeamId': home_team_id,
            'awayTeamId': away_team_id,
            'leagueId': league_id,
            'seasonType': season_type,
            'date': match_date
        }
        
        features = self._create_match_features(pd.Series(match_features), min_games_history=3)
        
        if not features:
            self.logger.error("Nao foi possivel criar features para esta partida")
            return None
        
        # Preparar para predição
        X_pred = pd.DataFrame([features])
        
        # Garantir mesmas colunas do treino
        for col in self.feature_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        X_pred = X_pred[self.feature_columns].fillna(0)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predição
        prediction = self.model.predict(X_pred_scaled)[0]
        probabilities = self.model.predict_proba(X_pred_scaled)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        prediction_name = result_map[prediction]
        
        self.logger.info(f"\nRESULTADO PREVISTO: {prediction_name}")
        self.logger.info(f"\nProbabilidades:")
        self.logger.info(f"   {away_name} Win: {probabilities[0]:.1%}")
        self.logger.info(f"   Draw:           {probabilities[1]:.1%}")
        self.logger.info(f"   {home_name} Win: {probabilities[2]:.1%}")
        
        return {
            'prediction': prediction_name,
            'probabilities': {
                'away_win': float(probabilities[0]),
                'draw': float(probabilities[1]),
                'home_win': float(probabilities[2])
            },
            'home_team': home_name,
            'away_team': away_name
        }
    
    def save_model(self, path: str = None):
        """Salva o modelo treinado"""
        if path is None:
            path = f'models/{self.experiment_name}'
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Tentar criar dicionários se não existirem
        try:
            if not hasattr(self, 'team_dict') or not self.team_dict:
                self._create_dictionaries()
        except Exception as e:
            self.logger.warning(f"Erro ao criar dicionários: {e}")
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'team_dict': getattr(self, 'team_dict', {}),
            'league_dict': getattr(self, 'league_dict', {}),
            'experiment_name': self.experiment_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(save_data, f'{path}.pkl')
        self.logger.info(f"Modelo salvo em: {path}.pkl")

# =============================================================================
# EXECUÇÃO PRINCIPAL SIMPLIFICADA
# =============================================================================

if __name__ == "__main__":
    
    print("\n")
    print("=" * 70)
    print("SOCCER MATCH PREDICTOR - SISTEMA OTIMIZADO")
    print("=" * 70)
    print("\nPredicao avancada com ensemble, validacao temporal e metricas de betting")
    print("=" * 70)
    
    # Caminho base
    BASE_PATH = r'C:\Users\Rafaribas\Desktop\Faculdade\Curso\6º período\SI\Projeto-Final\kaggle_data\data'
    
    # Inicializar predictor
    predictor = SoccerMatchPredictor(BASE_PATH)
    
    try:
        # Carregar dados
        print("\nCarregando dados...")
        predictor.load_data(cache=True)
        
        # Engenharia de features
        print("\nCriando features...")
        predictor.engineer_features(sample_size=5000)  # Pequeno para teste rápido
        
        # Treinar modelo
        print("\nTreinando modelo...")
        results = predictor.train_ensemble(n_trials=10, use_optuna=False)
        
        print("\n" + "=" * 70)
        print("MODELO TREINADO E PRONTO PARA USO!")
        print("=" * 70)
        
        # Exemplo de predição
        print("\nPara fazer predicoes, use:")
        print("predictor.predict_match(")
        print("    home_team_id=SEU_ID_HOME,")
        print("    away_team_id=SEU_ID_AWAY,")
        print("    league_id=SEU_ID_LIGA,")
        print("    season_type=TIPO_TEMPORADA")
        print(")")
        
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()