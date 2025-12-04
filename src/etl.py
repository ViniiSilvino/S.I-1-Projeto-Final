"""
ETL - Extração, Transformação e Carga de dados
"""
import os
import glob
import pandas as pd
import numpy as np
from src.config import DATA_PATHS, BASE_FILES, COMPLETED_STATUS, MISSING_VALUE_STRATEGY
from src.utils import (
    logger, parse_weight, parse_height, calculate_bmi,
    validate_dataframe, check_data_quality, reduce_mem_usage
)

class SoccerDataLoader:
    """Classe para carregar e processar dados de futebol"""
    
    def __init__(self):
        self.fixtures = None
        self.teams = None
        self.players = None
        self.standings = None
        self.team_stats = None
        self.leagues = None
        self.status = None
        self.venues = None
        self.lineup_files = []
        self.player_stats_files = []
        
    def load_all_data(self):
        """Carrega todos os dados necessários"""
        logger.info("="*60)
        logger.info("INICIANDO CARREGAMENTO DE DADOS")
        logger.info("="*60)
        
        try:
            # Carregar dados base
            self._load_base_data()
            
            # Carregar dados de lineup
            self._load_lineup_data()
            
            # Carregar dados de player stats
            self._load_player_stats_data()
            
            logger.info("\n✓ Todos os dados carregados com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False
    
    def _load_base_data(self):
        """Carrega arquivos da pasta base_data"""
        logger.info("\n--- Carregando dados base ---")
        base_path = DATA_PATHS['base']
        
        # Fixtures
        fixtures_path = os.path.join(base_path, BASE_FILES['fixtures'])
        self.fixtures = pd.read_csv(fixtures_path, parse_dates=['date'])
        logger.info(f"✓ Fixtures carregado: {len(self.fixtures):,} partidas")
        
        # Teams
        teams_path = os.path.join(base_path, BASE_FILES['teams'])
        self.teams = pd.read_csv(teams_path)
        logger.info(f"✓ Teams carregado: {len(self.teams):,} times")
        
        # Players
        players_path = os.path.join(base_path, BASE_FILES['players'])
        self.players = pd.read_csv(players_path, low_memory=False)
        logger.info(f"✓ Players carregado: {len(self.players):,} jogadores")
        
        # Standings
        standings_path = os.path.join(base_path, BASE_FILES['standings'])
        self.standings = pd.read_csv(standings_path, parse_dates=['last_matchDateTime'])
        logger.info(f"✓ Standings carregado: {len(self.standings):,} registros")
        
        # Team Stats
        team_stats_path = os.path.join(base_path, BASE_FILES['teamStats'])
        self.team_stats = pd.read_csv(team_stats_path)
        logger.info(f"✓ Team Stats carregado: {len(self.team_stats):,} registros")
        
        # Leagues
        leagues_path = os.path.join(base_path, BASE_FILES['leagues'])
        self.leagues = pd.read_csv(leagues_path)
        logger.info(f"✓ Leagues carregado: {len(self.leagues):,} ligas")
        
        # Status
        status_path = os.path.join(base_path, BASE_FILES['status'])
        self.status = pd.read_csv(status_path)
        logger.info(f"✓ Status carregado: {len(self.status):,} status")
        
        # Venues
        venues_path = os.path.join(base_path, BASE_FILES['venues'])
        if os.path.exists(venues_path):
            self.venues = pd.read_csv(venues_path)
            logger.info(f"✓ Venues carregado: {len(self.venues):,} estádios")
    
    def _load_lineup_data(self):
        """Carrega arquivos da pasta lineup_data"""
        logger.info("\n--- Carregando dados de lineup ---")
        lineup_path = DATA_PATHS['lineup']
        
        if os.path.exists(lineup_path):
            lineup_files = glob.glob(os.path.join(lineup_path, '*.csv'))
            
            if lineup_files:
                self.lineup_files = [pd.read_csv(f) for f in lineup_files]
                total_rows = sum(len(df) for df in self.lineup_files)
                logger.info(f"✓ {len(lineup_files)} arquivos de lineup carregados ({total_rows:,} registros)")
            else:
                logger.warning("Nenhum arquivo de lineup encontrado")
        else:
            logger.warning(f"Diretório de lineup não existe: {lineup_path}")
    
    def _load_player_stats_data(self):
        """Carrega arquivos da pasta playerStats_data"""
        logger.info("\n--- Carregando dados de player stats ---")
        stats_path = DATA_PATHS['playerStats']
        
        if os.path.exists(stats_path):
            stats_files = glob.glob(os.path.join(stats_path, '*.csv'))
            
            if stats_files:
                self.player_stats_files = [pd.read_csv(f) for f in stats_files]
                total_rows = sum(len(df) for df in self.player_stats_files)
                logger.info(f"✓ {len(stats_files)} arquivos de player stats carregados ({total_rows:,} registros)")
            else:
                logger.warning("Nenhum arquivo de player stats encontrado")
        else:
            logger.warning(f"Diretório de player stats não existe: {stats_path}")
    
    def preprocess_data(self):
        """Aplica pré-processamento em todos os dados"""
        logger.info("\n" + "="*60)
        logger.info("INICIANDO PRÉ-PROCESSAMENTO")
        logger.info("="*60)
        
        try:
            # Processar fixtures
            self._preprocess_fixtures()
            
            # Processar players
            self._preprocess_players()
            
            # Processar standings
            self._preprocess_standings()
            
            # Processar team stats
            self._preprocess_team_stats()
            
            # Filtrar apenas partidas completas
            self._filter_completed_matches()
            
            logger.info("\n✓ Pré-processamento concluído com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _preprocess_fixtures(self):
        """Pré-processa dados de fixtures"""
        logger.info("\n--- Pré-processando Fixtures ---")
        
        # Validar colunas essenciais
        required_cols = ['eventId', 'date', 'homeTeamId', 'awayTeamId', 'statusId']
        if not validate_dataframe(self.fixtures, required_cols, 'Fixtures'):
            raise ValueError("Fixtures não possui colunas necessárias")
        
        # Converter data se necessário
        if not pd.api.types.is_datetime64_any_dtype(self.fixtures['date']):
            self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
        
        # Verificar colunas de placar
        logger.info(f"Colunas em fixtures: {list(self.fixtures.columns)}")
        
        # Tentar encontrar colunas de placar (podem ter nomes diferentes)
        score_cols = [col for col in self.fixtures.columns if 'score' in col.lower()]
        logger.info(f"Colunas de placar encontradas: {score_cols}")
        
        # Criar coluna de resultado
        if 'homeScore' in self.fixtures.columns and 'awayScore' in self.fixtures.columns:
            self.fixtures['result'] = self.fixtures.apply(self._determine_result, axis=1)
            results_created = self.fixtures['result'].notna().sum()
            logger.info(f"✓ Coluna 'result' criada ({results_created:,} resultados)")
        elif len(score_cols) >= 2:
            # Tentar usar as primeiras duas colunas de score encontradas
            logger.info(f"Tentando usar colunas: {score_cols[0]} e {score_cols[1]}")
            self.fixtures['homeScore'] = self.fixtures[score_cols[0]]
            self.fixtures['awayScore'] = self.fixtures[score_cols[1]]
            self.fixtures['result'] = self.fixtures.apply(self._determine_result, axis=1)
            results_created = self.fixtures['result'].notna().sum()
            logger.info(f"✓ Coluna 'result' criada ({results_created:,} resultados)")
        else:
            raise ValueError("Não foi possível encontrar colunas de placar. Verifique o arquivo fixtures.csv")
        
        # Verificar qualidade
        check_data_quality(self.fixtures, 'Fixtures')
    
    def _determine_result(self, row):
        """Determina o resultado da partida (0=Empate, 1=Casa, 2=Visitante)"""
        if pd.isna(row['homeScore']) or pd.isna(row['awayScore']):
            return np.nan
        
        if row['homeScore'] > row['awayScore']:
            return 1  # Vitória casa
        elif row['homeScore'] < row['awayScore']:
            return 2  # Vitória visitante
        else:
            return 0  # Empate
    
    def _preprocess_players(self):
        """Pré-processa dados de jogadores"""
        logger.info("\n--- Pré-processando Players ---")
        
        # Converter peso e altura
        if 'displayWeight' in self.players.columns:
            logger.info("Convertendo peso (lbs → kg)...")
            self.players['weight_kg'] = self.players['displayWeight'].apply(parse_weight)
            converted = self.players['weight_kg'].notna().sum()
            logger.info(f"✓ {converted:,} pesos convertidos")
        
        if 'displayHeight' in self.players.columns:
            logger.info("Convertendo altura (ft'in\" → metros)...")
            self.players['height_m'] = self.players['displayHeight'].apply(parse_height)
            converted = self.players['height_m'].notna().sum()
            logger.info(f"✓ {converted:,} alturas convertidas")
        
        # Calcular BMI
        if 'weight_kg' in self.players.columns and 'height_m' in self.players.columns:
            logger.info("Calculando BMI...")
            self.players['bmi'] = self.players.apply(
                lambda row: calculate_bmi(row['weight_kg'], row['height_m']),
                axis=1
            )
            calculated = self.players['bmi'].notna().sum()
            logger.info(f"✓ {calculated:,} BMIs calculados")
        
        # Converter idade
        if 'age' in self.players.columns:
            self.players['age'] = pd.to_numeric(self.players['age'], errors='coerce')
        
        # Verificar qualidade
        check_data_quality(self.players, 'Players')
    
    def _preprocess_standings(self):
        """Pré-processa dados de classificação"""
        logger.info("\n--- Pré-processando Standings ---")
        
        # Converter data se necessário
        if 'last_matchDateTime' in self.standings.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.standings['last_matchDateTime']):
                self.standings['last_matchDateTime'] = pd.to_datetime(
                    self.standings['last_matchDateTime']
                )
        
        # Calcular métricas derivadas
        if 'gf' in self.standings.columns and 'gamesPlayed' in self.standings.columns:
            self.standings['goals_per_game'] = (
                self.standings['gf'] / self.standings['gamesPlayed'].replace(0, np.nan)
            )
            self.standings['goals_against_per_game'] = (
                self.standings['ga'] / self.standings['gamesPlayed'].replace(0, np.nan)
            )
            logger.info("✓ Métricas de gols por jogo calculadas")
        
        # Calcular saldo de gols
        if 'gf' in self.standings.columns and 'ga' in self.standings.columns:
            self.standings['goal_difference'] = self.standings['gf'] - self.standings['ga']
            logger.info("✓ Saldo de gols calculado")
        
        # Verificar qualidade
        check_data_quality(self.standings, 'Standings')
    
    def _preprocess_team_stats(self):
        """Pré-processa estatísticas de times"""
        logger.info("\n--- Pré-processando Team Stats ---")
        
        # Calcular precisão de passes
        if 'accuratePasses' in self.team_stats.columns and 'totalPasses' in self.team_stats.columns:
            self.team_stats['pass_accuracy'] = (
                self.team_stats['accuratePasses'] / 
                self.team_stats['totalPasses'].replace(0, np.nan)
            ) * 100
            logger.info("✓ Precisão de passes calculada")
        
        # Calcular precisão de chutes
        if 'shotsOnTarget' in self.team_stats.columns and 'totalShots' in self.team_stats.columns:
            self.team_stats['shot_accuracy'] = (
                self.team_stats['shotsOnTarget'] / 
                self.team_stats['totalShots'].replace(0, np.nan)
            ) * 100
            logger.info("✓ Precisão de chutes calculada")
        
        # Converter posse de bola para float se for string
        if 'possessionPct' in self.team_stats.columns:
            self.team_stats['possessionPct'] = pd.to_numeric(
                self.team_stats['possessionPct'], 
                errors='coerce'
            )
        
        # Verificar qualidade
        check_data_quality(self.team_stats, 'Team Stats')
    
    def _filter_completed_matches(self):
        """Filtra apenas partidas completas"""
        logger.info("\n--- Filtrando partidas completas ---")
        
        initial_count = len(self.fixtures)
        
        # Filtrar por status
        self.fixtures = self.fixtures[
            self.fixtures['statusId'].isin(COMPLETED_STATUS)
        ].copy()
        
        # Remover partidas sem resultado
        if 'result' in self.fixtures.columns:
            self.fixtures = self.fixtures[
                self.fixtures['result'].notna()
            ].copy()
        
        final_count = len(self.fixtures)
        removed = initial_count - final_count
        
        logger.info(f"✓ {final_count:,} partidas completas mantidas")
        logger.info(f"  ({removed:,} partidas removidas)")
    
    def handle_missing_values(self):
        """Trata valores faltantes em todos os datasets"""
        logger.info("\n--- Tratando valores faltantes ---")
        
        # Fixtures - preencher attendance com 0
        if 'attendance' in self.fixtures.columns:
            self.fixtures.loc[:, 'attendance'] = self.fixtures['attendance'].fillna(0)
        
        # Team Stats - preencher stats com 0
        numeric_cols = self.team_stats.select_dtypes(include=[np.number]).columns
        self.team_stats[numeric_cols] = self.team_stats[numeric_cols].fillna(0)
        
        # Players - preencher com medianas
        if 'age' in self.players.columns:
            median_age = self.players['age'].median()
            self.players.loc[:, 'age'] = self.players['age'].fillna(median_age)
        
        if 'weight_kg' in self.players.columns:
            median_weight = self.players['weight_kg'].median()
            self.players.loc[:, 'weight_kg'] = self.players['weight_kg'].fillna(median_weight)
        
        if 'height_m' in self.players.columns:
            median_height = self.players['height_m'].median()
            self.players.loc[:, 'height_m'] = self.players['height_m'].fillna(median_height)
        
        logger.info("✓ Valores faltantes tratados")
    
    def get_processed_data(self):
        """Retorna todos os dados processados"""
        return {
            'fixtures': self.fixtures,
            'teams': self.teams,
            'players': self.players,
            'standings': self.standings,
            'team_stats': self.team_stats,
            'leagues': self.leagues,
            'status': self.status,
            'venues': self.venues,
            'lineup_files': self.lineup_files,
            'player_stats_files': self.player_stats_files
        }

def load_and_preprocess_data():
    """Função principal para carregar e pré-processar dados"""
    loader = SoccerDataLoader()
    
    # Carregar dados
    if not loader.load_all_data():
        raise Exception("Falha ao carregar dados")
    
    # Pré-processar
    if not loader.preprocess_data():
        raise Exception("Falha no pré-processamento")
    
    # Tratar valores faltantes
    loader.handle_missing_values()
    
    # Retornar dados processados
    return loader.get_processed_data()

if __name__ == "__main__":
    # Teste do módulo
    data = load_and_preprocess_data()
    logger.info("\n✓ ETL executado com sucesso!")
    logger.info(f"Partidas prontas para feature engineering: {len(data['fixtures']):,}")