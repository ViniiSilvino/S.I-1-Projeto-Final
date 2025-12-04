"""
Feature Engineering - Criação do master_df com todas as features preditivas
"""
import pandas as pd
import numpy as np
from src.config import FEATURE_GROUPS, ALL_FEATURES, RECENT_GAMES_WINDOW, FORM_POINTS
from src.utils import logger, parse_form_string, safe_divide, check_data_quality

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self, data_dict):
        """
        Inicializa com os dados pré-processados
        
        Args:
            data_dict: Dicionário com todos os DataFrames processados
        """
        self.fixtures = data_dict['fixtures'].copy()
        self.teams = data_dict['teams'].copy()
        self.players = data_dict['players'].copy()
        self.standings = data_dict['standings'].copy()
        self.team_stats = data_dict['team_stats'].copy()
        self.leagues = data_dict['leagues'].copy()
        self.lineup_files = data_dict.get('lineup_files', [])
        self.player_stats_files = data_dict.get('player_stats_files', [])
        
        self.master_df = None
    
    def create_master_dataframe(self):
        """Cria o DataFrame mestre com todas as features"""
        logger.info("\n" + "="*60)
        logger.info("INICIANDO FEATURE ENGINEERING")
        logger.info("="*60)
        
        try:
            # 1. Inicializar com fixtures
            self._initialize_master_df()
            
            # 2. Adicionar features de forma recente
            self._add_form_features()
            
            # 3. Adicionar features de performance
            self._add_performance_features()
            
            # 4. Adicionar features de estatísticas de jogo
            self._add_match_stats_features()
            
            # 5. Adicionar features de escalação
            self._add_lineup_features()
            
            # 6. Adicionar features derivadas
            self._add_derived_features()
            
            # 7. Limpar e validar
            self._clean_and_validate()
            
            logger.info("\n✓ Master DataFrame criado com sucesso!")
            logger.info(f"Shape final: {self.master_df.shape}")
            logger.info(f"Features criadas: {len([c for c in self.master_df.columns if c in ALL_FEATURES])}")
            
            return self.master_df
            
        except Exception as e:
            logger.error(f"Erro na criação do master DataFrame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_master_df(self):
        """Inicializa o DataFrame mestre com fixtures e target"""
        logger.info("\n--- Inicializando Master DataFrame ---")
        
        # Começar com fixtures que têm resultado
        self.master_df = self.fixtures[self.fixtures['result'].notna()].copy()
        
        # Renomear result para target
        self.master_df['target'] = self.master_df['result'].astype(int)
        
        # Adicionar informações básicas da liga (se disponível)
        if 'leagueId' in self.master_df.columns and len(self.leagues) > 0:
            # Verificar quais colunas existem em leagues
            logger.info(f"Colunas em leagues: {list(self.leagues.columns)}")
            
            # Tentar diferentes nomes de coluna para o nome da liga
            name_col = None
            for col in ['name', 'league_name', 'leagueName', 'displayName']:
                if col in self.leagues.columns:
                    name_col = col
                    break
            
            if name_col:
                self.master_df = self.master_df.merge(
                    self.leagues[['leagueId', name_col]].rename(columns={name_col: 'league_name'}),
                    on='leagueId',
                    how='left'
                )
                logger.info(f"✓ Informações da liga adicionadas usando coluna '{name_col}'")
            else:
                logger.warning("Nome da liga não encontrado. Continuando sem essa informação.")
        
        logger.info(f"✓ Master DF inicializado: {len(self.master_df):,} partidas")
        logger.info(f"  Distribuição do target:")
        logger.info(f"    Empate (0): {(self.master_df['target']==0).sum():,}")
        logger.info(f"    Vitória Casa (1): {(self.master_df['target']==1).sum():,}")
        logger.info(f"    Vitória Visitante (2): {(self.master_df['target']==2).sum():,}")
    
    def _add_form_features(self):
        """Adiciona features de forma recente dos times"""
        logger.info("\n--- Adicionando features de forma recente ---")
        
        if 'form' not in self.standings.columns:
            logger.warning("Coluna 'form' não encontrada em standings. Pulando features de forma.")
            # Criar features vazias
            for feature in FEATURE_GROUPS['home_form'] + FEATURE_GROUPS['away_form']:
                self.master_df[feature] = 0
            return
        
        # Preparar standings com métricas de forma
        standings_with_form = self.standings.copy()
        
        # Parse da string de forma
        form_metrics = standings_with_form['form'].apply(parse_form_string)
        standings_with_form['form_wins'] = form_metrics.apply(lambda x: x['wins'])
        standings_with_form['form_draws'] = form_metrics.apply(lambda x: x['draws'])
        standings_with_form['form_losses'] = form_metrics.apply(lambda x: x['losses'])
        standings_with_form['form_points'] = form_metrics.apply(lambda x: x['points'])
        
        # Merge para home team
        home_standings = standings_with_form.copy()
        home_standings.columns = ['home_' + col if col not in ['teamId', 'leagueId'] else col 
                                  for col in home_standings.columns]
        
        self.master_df = self.master_df.merge(
            home_standings[['teamId', 'leagueId', 'home_form_wins', 'home_form_draws', 
                           'home_form_losses', 'home_form_points']],
            left_on=['homeTeamId', 'leagueId'],
            right_on=['teamId', 'leagueId'],
            how='left',
            suffixes=('', '_home')
        )
        
        # Merge para away team
        away_standings = standings_with_form.copy()
        away_standings.columns = ['away_' + col if col not in ['teamId', 'leagueId'] else col 
                                  for col in away_standings.columns]
        
        self.master_df = self.master_df.merge(
            away_standings[['teamId', 'leagueId', 'away_form_wins', 'away_form_draws', 
                           'away_form_losses', 'away_form_points']],
            left_on=['awayTeamId', 'leagueId'],
            right_on=['teamId', 'leagueId'],
            how='left',
            suffixes=('', '_away')
        )
        
        # Renomear para match com FEATURE_GROUPS
        rename_map = {
            'home_form_wins': 'home_recent_wins',
            'home_form_draws': 'home_recent_draws',
            'home_form_losses': 'home_recent_losses',
            'away_form_wins': 'away_recent_wins',
            'away_form_draws': 'away_recent_draws',
            'away_form_losses': 'away_recent_losses'
        }
        self.master_df.rename(columns=rename_map, inplace=True)
        
        # Preencher NaN com 0
        form_cols = FEATURE_GROUPS['home_form'] + FEATURE_GROUPS['away_form']
        for col in form_cols:
            if col in self.master_df.columns:
                self.master_df[col].fillna(0, inplace=True)
        
        logger.info("✓ Features de forma recente adicionadas")
    
    def _add_performance_features(self):
        """Adiciona features de performance geral dos times"""
        logger.info("\n--- Adicionando features de performance ---")
        
        # Selecionar colunas relevantes do standings
        performance_cols = ['teamId', 'leagueId', 'points', 'wins', 'losses', 'draws',
                           'gamesPlayed', 'goals_per_game', 'goals_against_per_game', 
                           'goal_difference']
        
        available_cols = [col for col in performance_cols if col in self.standings.columns]
        standings_perf = self.standings[available_cols].copy()
        
        # Home team performance
        home_perf = standings_perf.copy()
        home_perf.columns = ['home_' + col if col not in ['teamId', 'leagueId'] else col 
                             for col in home_perf.columns]
        
        self.master_df = self.master_df.merge(
            home_perf,
            left_on=['homeTeamId', 'leagueId'],
            right_on=['teamId', 'leagueId'],
            how='left',
            suffixes=('', '_home_perf')
        )
        
        # Away team performance
        away_perf = standings_perf.copy()
        away_perf.columns = ['away_' + col if col not in ['teamId', 'leagueId'] else col 
                             for col in away_perf.columns]
        
        self.master_df = self.master_df.merge(
            away_perf,
            left_on=['awayTeamId', 'leagueId'],
            right_on=['teamId', 'leagueId'],
            how='left',
            suffixes=('', '_away_perf')
        )
        
        # Preencher NaN
        perf_cols = FEATURE_GROUPS['home_performance'] + FEATURE_GROUPS['away_performance']
        for col in perf_cols:
            if col in self.master_df.columns:
                self.master_df[col].fillna(0, inplace=True)
        
        logger.info("✓ Features de performance adicionadas")
    
    def _add_match_stats_features(self):
        """Adiciona features de estatísticas de jogo (posse, passes, chutes)"""
        logger.info("\n--- Adicionando features de estatísticas de jogo ---")
        
        if len(self.team_stats) == 0:
            logger.warning("Team stats vazio. Pulando features de estatísticas.")
            for feature in FEATURE_GROUPS['match_stats']:
                self.master_df[feature] = 0
            return
        
        # Calcular médias históricas por time
        team_stats_agg = self.team_stats.groupby('teamId').agg({
            'possessionPct': 'mean',
            'pass_accuracy': 'mean',
            'shot_accuracy': 'mean'
        }).reset_index()
        
        team_stats_agg.columns = ['teamId', 'possession_avg', 'pass_accuracy_avg', 'shot_accuracy_avg']
        
        # Home team stats
        home_stats = team_stats_agg.copy()
        home_stats.columns = ['homeTeamId', 'home_possession_avg', 'home_pass_accuracy', 'home_shot_accuracy']
        
        self.master_df = self.master_df.merge(
            home_stats,
            on='homeTeamId',
            how='left'
        )
        
        # Away team stats
        away_stats = team_stats_agg.copy()
        away_stats.columns = ['awayTeamId', 'away_possession_avg', 'away_pass_accuracy', 'away_shot_accuracy']
        
        self.master_df = self.master_df.merge(
            away_stats,
            on='awayTeamId',
            how='left'
        )
        
        # Preencher NaN com valores padrão
        stats_defaults = {
            'home_possession_avg': 50,
            'away_possession_avg': 50,
            'home_pass_accuracy': 75,
            'away_pass_accuracy': 75,
            'home_shot_accuracy': 30,
            'away_shot_accuracy': 30
        }
        
        for col, default in stats_defaults.items():
            if col in self.master_df.columns:
                self.master_df[col].fillna(default, inplace=True)
        
        logger.info("✓ Features de estatísticas de jogo adicionadas")
    
    def _add_lineup_features(self):
        """Adiciona features de qualidade da escalação"""
        logger.info("\n--- Adicionando features de escalação ---")
        
        if len(self.lineup_files) == 0 or len(self.players) == 0:
            logger.warning("Dados de lineup ou players não disponíveis. Pulando features de escalação.")
            for feature in FEATURE_GROUPS['lineup_quality']:
                self.master_df[feature] = 0
            return
        
        # Combinar todos os arquivos de lineup
        lineup_df = pd.concat(self.lineup_files, ignore_index=True)
        
        # Filtrar apenas titulares (starter=True)
        if 'starter' in lineup_df.columns:
            lineup_df = lineup_df[lineup_df['starter'] == True].copy()
        
        # Merge com dados dos jogadores
        lineup_with_players = lineup_df.merge(
            self.players[['athleteId', 'age', 'height_m', 'weight_kg']],
            on='athleteId',
            how='left'
        )
        
        # Calcular médias por time e eventId
        lineup_agg = lineup_with_players.groupby(['eventId', 'teamId']).agg({
            'age': 'mean',
            'height_m': 'mean',
            'weight_kg': 'mean'
        }).reset_index()
        
        lineup_agg.columns = ['eventId', 'teamId', 'avg_age', 'avg_height', 'avg_weight']
        
        # Home team lineup
        home_lineup = lineup_agg.copy()
        home_lineup.columns = ['eventId', 'homeTeamId', 'home_avg_age', 'home_avg_height', 'home_avg_weight']
        
        self.master_df = self.master_df.merge(
            home_lineup,
            left_on=['eventId', 'homeTeamId'],
            right_on=['eventId', 'homeTeamId'],
            how='left'
        )
        
        # Away team lineup
        away_lineup = lineup_agg.copy()
        away_lineup.columns = ['eventId', 'awayTeamId', 'away_avg_age', 'away_avg_height', 'away_avg_weight']
        
        self.master_df = self.master_df.merge(
            away_lineup,
            left_on=['eventId', 'awayTeamId'],
            right_on=['eventId', 'awayTeamId'],
            how='left'
        )
        
        # Preencher NaN com médias globais
        lineup_defaults = {
            'home_avg_age': self.players['age'].mean() if 'age' in self.players.columns else 25,
            'away_avg_age': self.players['age'].mean() if 'age' in self.players.columns else 25,
            'home_avg_height': self.players['height_m'].mean() if 'height_m' in self.players.columns else 1.80,
            'away_avg_height': self.players['height_m'].mean() if 'height_m' in self.players.columns else 1.80,
            'home_avg_weight': self.players['weight_kg'].mean() if 'weight_kg' in self.players.columns else 75,
            'away_avg_weight': self.players['weight_kg'].mean() if 'weight_kg' in self.players.columns else 75
        }
        
        for col, default in lineup_defaults.items():
            if col in self.master_df.columns:
                self.master_df[col].fillna(default, inplace=True)
        
        logger.info("✓ Features de escalação adicionadas")
    
    def _add_derived_features(self):
        """Adiciona features derivadas (diferenças, ratios, etc)"""
        logger.info("\n--- Adicionando features derivadas ---")
        
        # Diferença de pontos
        if 'home_points' in self.master_df.columns and 'away_points' in self.master_df.columns:
            self.master_df['points_difference'] = self.master_df['home_points'] - self.master_df['away_points']
        
        # Diferença de forma
        if 'home_form_points' in self.master_df.columns and 'away_form_points' in self.master_df.columns:
            self.master_df['form_difference'] = self.master_df['home_form_points'] - self.master_df['away_form_points']
        
        # Diferença de gols por jogo
        if 'home_goals_per_game' in self.master_df.columns and 'away_goals_per_game' in self.master_df.columns:
            self.master_df['attack_difference'] = (
                self.master_df['home_goals_per_game'] - self.master_df['away_goals_per_game']
            )
        
        # Diferença defensiva
        if 'home_goals_against_per_game' in self.master_df.columns and 'away_goals_against_per_game' in self.master_df.columns:
            self.master_df['defense_difference'] = (
                self.master_df['away_goals_against_per_game'] - self.master_df['home_goals_against_per_game']
            )
        
        logger.info("✓ Features derivadas adicionadas")
    
    def _clean_and_validate(self):
        """Limpa e valida o DataFrame final"""
        logger.info("\n--- Limpando e validando ---")
        
        # Selecionar apenas colunas necessárias
        essential_cols = ['eventId', 'date', 'homeTeamId', 'awayTeamId', 'target']
        feature_cols = [col for col in ALL_FEATURES if col in self.master_df.columns]
        derived_cols = ['points_difference', 'form_difference', 'attack_difference', 'defense_difference']
        derived_cols = [col for col in derived_cols if col in self.master_df.columns]
        
        final_cols = essential_cols + feature_cols + derived_cols
        
        # Remover duplicatas
        initial_len = len(self.master_df)
        self.master_df.drop_duplicates(subset=['eventId'], inplace=True)
        self.master_df = self.master_df[final_cols].copy()
        if len(self.master_df) < initial_len:
            logger.warning(f"Removidas {initial_len - len(self.master_df)} linhas duplicadas")
        
        # Remover linhas com target nulo
        self.master_df = self.master_df[self.master_df['target'].notna()].copy()
        
        # Preencher qualquer NaN remanescente com 0
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns
        self.master_df[numeric_cols] = self.master_df[numeric_cols].fillna(0)
        
        # Verificar infinitos
        self.master_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Verificar qualidade final
        check_data_quality(self.master_df, 'Master DataFrame')
        
        logger.info("✓ Limpeza e validação concluídas")

def create_features(data_dict):
    """
    Função principal para criar features
    
    Args:
        data_dict: Dicionário com dados processados
    
    Returns:
        DataFrame com todas as features
    """
    engineer = FeatureEngineer(data_dict)
    master_df = engineer.create_master_dataframe()
    return master_df

if __name__ == "__main__":
    # Teste do módulo
    from src.etl import load_and_preprocess_data
    
    data = load_and_preprocess_data()
    master_df = create_features(data)
    
    logger.info(f"\n✓ Feature Engineering concluído!")
    logger.info(f"DataFrame final: {master_df.shape}")
    logger.info(f"\nPrimeiras linhas:")
    print(master_df.head())