"""
Soccer Match Outcome Predictor - Sistema Completo
Prediz resultados de partidas (Home Win / Draw / Away Win)
Usando Random Forest com Feature Engineering Avançado
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

class SoccerMatchPredictor:
    """
    Preditor de resultados de partidas de futebol
    Target: 0 = Away Win, 1 = Draw, 2 = Home Win
    """
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Carrega os dados dos CSVs"""
        print("="*70)
        print("📂 CARREGANDO DADOS DO DATASET")
        print("="*70)
        
        # Carregar arquivos base
        print("\n📄 Carregando arquivos base...")
        self.fixtures = pd.read_csv(os.path.join(self.base_path, 'base_data', 'fixtures.csv'))
        self.standings = pd.read_csv(os.path.join(self.base_path, 'base_data', 'standings.csv'))
        self.team_stats = pd.read_csv(os.path.join(self.base_path, 'base_data', 'teamStats.csv'))
        self.teams = pd.read_csv(os.path.join(self.base_path, 'base_data', 'teams.csv'))
        self.leagues = pd.read_csv(os.path.join(self.base_path, 'base_data', 'leagues.csv'))
        
        # Converter datas
        self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
        
        print(f"   ✅ Fixtures: {len(self.fixtures):,} partidas")
        print(f"   ✅ Standings: {len(self.standings):,} registros")
        print(f"   ✅ Team Stats: {len(self.team_stats):,} registros")
        print(f"   ✅ Teams: {len(self.teams):,} times")
        print(f"   ✅ Leagues: {len(self.leagues):,} ligas")
        
        # Análise inicial
        completed = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) & 
            (self.fixtures['awayTeamScore'].notna())
        ]
        
        print(f"\n📊 Visão Geral:")
        print(f"   Período: {self.fixtures['date'].min().date()} até {self.fixtures['date'].max().date()}")
        print(f"   Partidas completas: {len(completed):,} ({len(completed)/len(self.fixtures)*100:.1f}%)")
        
        # Distribuição de resultados
        home_wins = (completed['homeTeamScore'] > completed['awayTeamScore']).sum()
        draws = (completed['homeTeamScore'] == completed['awayTeamScore']).sum()
        away_wins = (completed['homeTeamScore'] < completed['awayTeamScore']).sum()
        
        print(f"\n🎯 Distribuição histórica:")
        print(f"   Home Win: {home_wins:,} ({home_wins/len(completed)*100:.1f}%)")
        print(f"   Draw: {draws:,} ({draws/len(completed)*100:.1f}%)")
        print(f"   Away Win: {away_wins:,} ({away_wins/len(completed)*100:.1f}%)")
        
    def create_target(self, row):
        """Cria a variável target (0=Away Win, 1=Draw, 2=Home Win)"""
        # Verificar se os placares existem e são válidos
        if pd.isna(row['homeTeamScore']) or pd.isna(row['awayTeamScore']):
            return None
        
        home_score = row['homeTeamScore']
        away_score = row['awayTeamScore']
        
        # Debug: imprimir alguns exemplos
        # print(f"Home: {home_score}, Away: {away_score}")
        
        if home_score > away_score:
            return 2  # Home Win
        elif home_score < away_score:
            return 0  # Away Win
        else:
            return 1  # Draw
    
    def get_recent_form(self, team_id, current_date, n_games=5):
        """Calcula forma recente do time (últimos N jogos)"""
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
    
    def get_h2h_stats(self, home_id, away_id, current_date, n_games=5):
        """Estatísticas de confrontos diretos"""
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
    
    def get_team_stats_features(self, team_id, season_type):
        """Pega estatísticas do time da tabela teamStats"""
        stats = self.team_stats[
            (self.team_stats['teamId'] == team_id) &
            (self.team_stats['seasonType'] == season_type)
        ]
        
        if len(stats) == 0:
            return {}
        
        stat = stats.iloc[0]
        return {
            'possession_pct': stat.get('possessionPct', 50),
            'fouls_committed': stat.get('foulsCommitted', 0),
            'yellow_cards': stat.get('yellowCards', 0),
            'red_cards': stat.get('redCards', 0),
            'offsides': stat.get('offsides', 0),
            'won_corners': stat.get('wonCorners', 0),
            'saves': stat.get('saves', 0),
            'total_shots': stat.get('totalShots', 0),
            'shots_on_target': stat.get('shotsOnTarget', 0)
        }
    
    def get_standings_features(self, team_id, league_id, season_type):
        """Pega features da tabela de classificação"""
        standing = self.standings[
            (self.standings['teamId'] == team_id) &
            (self.standings['leagueId'] == league_id) &
            (self.standings['seasonType'] == season_type)
        ]
        
        if len(standing) == 0:
            return {}
        
        s = standing.iloc[0]
        total_games = s.get('gamesPlayed', 1)
        if total_games == 0:
            total_games = 1
        
        return {
            'team_rank': s.get('teamRank', 10),
            'points': s.get('points', 0),
            'games_played': total_games,
            'wins': s.get('wins', 0),
            'losses': s.get('losses', 0),
            'ties': s.get('ties', 0),
            'goals_for': s.get('gf', 0),
            'goals_against': s.get('ga', 0),
            'goal_diff': s.get('gd', 0),
            'points_per_game': s.get('points', 0) / total_games,
            'win_rate': s.get('wins', 0) / total_games
        }
    
    def engineer_features(self, sample_size=None):
        """Feature Engineering completo"""
        print("\n" + "="*70)
        print("🔧 ENGENHARIA DE FEATURES")
        print("="*70)
        
        features_list = []
        targets = []
        valid_indices = []
        
        # Filtrar apenas jogos finalizados com placar VÁLIDO
        # Remover jogos 0-0 que são provavelmente placeholders de jogos futuros
        completed_matches = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) &
            (self.fixtures['awayTeamScore'].notna()) &
            ~((self.fixtures['homeTeamScore'] == 0) & (self.fixtures['awayTeamScore'] == 0))
        ].copy()
        
        # Verificar distribuição inicial
        completed_matches['temp_result'] = completed_matches.apply(
            lambda row: 2 if row['homeTeamScore'] > row['awayTeamScore'] 
            else (0 if row['homeTeamScore'] < row['awayTeamScore'] else 1), 
            axis=1
        )
        
        print(f"\n📊 Distribuição ANTES da amostragem:")
        dist = completed_matches['temp_result'].value_counts()
        total = len(completed_matches)
        for result, count in dist.items():
            result_name = ['Away Win', 'Draw', 'Home Win'][result]
            print(f"   {result_name}: {count:,} ({count/total*100:.1f}%)")
        
        completed_matches = completed_matches.drop('temp_result', axis=1).sort_values('date').reset_index(drop=True)
        
        # Usar amostra se especificado
        if sample_size and sample_size < len(completed_matches):
            # IMPORTANTE: Não pegar do final (tem jogos futuros)
            # Pegar do MEIO do dataset para ter boa distribuição
            if len(completed_matches) > sample_size * 2:
                start_idx = len(completed_matches) // 4  # Começar em 25% do dataset
                end_idx = start_idx + sample_size
                completed_matches = completed_matches.iloc[start_idx:end_idx]
                print(f"\n⚠️  Usando {sample_size:,} partidas (índices {start_idx:,} a {end_idx:,})")
            else:
                completed_matches = completed_matches.head(sample_size)
                print(f"\n⚠️  Usando {sample_size:,} primeiras partidas")
        
        print(f"\n📊 Processando {len(completed_matches):,} partidas finalizadas...")
        
        total = len(completed_matches)
        for idx, match in completed_matches.iterrows():
            if idx % 1000 == 0:
                progress = len(features_list) / total * 100 if total > 0 else 0
                print(f"   Progresso: {progress:.1f}% ({len(features_list):,}/{total:,})")
            
            features = {}
            
            # IDs e informações básicas
            home_id = match['homeTeamId']
            away_id = match['awayTeamId']
            league_id = match['leagueId']
            season_type = match['seasonType']
            current_date = match['date']
            
            # Verificar se temos histórico suficiente (pelo menos 30 dias antes)
            min_history_date = current_date - pd.Timedelta(days=30)
            historical_matches = self.fixtures[self.fixtures['date'] < min_history_date]
            
            if len(historical_matches) < 100:  # Precisamos de histórico
                continue
            
            # FORMA RECENTE
            home_form = self.get_recent_form(home_id, current_date, n_games=5)
            away_form = self.get_recent_form(away_id, current_date, n_games=5)
            
            # Pular se não tem jogos recentes
            if home_form['games'] < 3 or away_form['games'] < 3:
                continue
            
            features['home_recent_ppg'] = home_form['ppg']
            features['home_recent_gf'] = home_form['gf']
            features['home_recent_ga'] = home_form['ga']
            features['home_recent_win_rate'] = home_form['win_rate']
            
            features['away_recent_ppg'] = away_form['ppg']
            features['away_recent_gf'] = away_form['gf']
            features['away_recent_ga'] = away_form['ga']
            features['away_recent_win_rate'] = away_form['win_rate']
            
            # CONFRONTOS DIRETOS
            h2h = self.get_h2h_stats(home_id, away_id, current_date)
            features['h2h_home_win_rate'] = h2h['home_win_rate']
            features['h2h_draw_rate'] = h2h['draw_rate']
            features['h2h_matches'] = h2h['games']
            
            # STANDINGS
            home_standing = self.get_standings_features(home_id, league_id, season_type)
            away_standing = self.get_standings_features(away_id, league_id, season_type)
            
            for key, value in home_standing.items():
                features[f'home_{key}'] = value
            for key, value in away_standing.items():
                features[f'away_{key}'] = value
            
            # ESTATÍSTICAS DO TIME
            home_stats = self.get_team_stats_features(home_id, season_type)
            away_stats = self.get_team_stats_features(away_id, season_type)
            
            for key, value in home_stats.items():
                features[f'home_{key}'] = value
            for key, value in away_stats.items():
                features[f'away_{key}'] = value
            
            # FEATURES DERIVADAS (diferenças)
            if home_standing and away_standing:
                features['rank_diff'] = away_standing.get('team_rank', 10) - home_standing.get('team_rank', 10)
                features['points_diff'] = home_standing.get('points', 0) - away_standing.get('points', 0)
                features['goal_diff_diff'] = home_standing.get('goal_diff', 0) - away_standing.get('goal_diff', 0)
                features['form_diff'] = features['home_recent_ppg'] - features['away_recent_ppg']
            
            # TARGET
            target = self.create_target(match)
            
            if target is not None and len(features) > 10:
                features_list.append(features)
                targets.append(target)
                valid_indices.append(idx)
        
        print(f"\n✅ {len(features_list):,} partidas com features completas")
        
        # Criar DataFrame
        self.X = pd.DataFrame(features_list)
        self.y = np.array(targets)
        
        # Verificar distribuição final
        print(f"\n🎯 Distribuição FINAL do target:")
        unique, counts = np.unique(self.y, return_counts=True)
        total = len(self.y)
        for val, count in zip(unique, counts):
            result_name = ['Away Win', 'Draw', 'Home Win'][val]
            print(f"   {result_name} ({val}): {count:,} ({count/total*100:.1f}%)")
        
        # VERIFICAR SE DISTRIBUIÇÃO É VÁLIDA
        if len(unique) < 3:
            print(f"\n❌ ERRO: Apenas {len(unique)} classe(s) no dataset!")
            print("   Isso indica problema nos dados. Verifique:")
            print("   1. Se há partidas com placares variados")
            print("   2. Se a coluna de placar está correta")
            print("   3. Considere usar mais dados (remova sample_size)")
        
        if len(completed_matches) > 0:
            self.match_dates = completed_matches.loc[valid_indices, 'date'].values
        else:
            self.match_dates = []
        
        # Preencher NaN com mediana
        self.X = self.X.fillna(self.X.median())
        
        self.feature_columns = self.X.columns.tolist()
        
        print(f"\n📈 Total de features criadas: {len(self.feature_columns)}")
    
    def train_model(self, test_size=0.2):
        """Treina o modelo com validação temporal"""
        print("\n" + "="*70)
        print("🎓 TREINAMENTO DO MODELO")
        print("="*70)
        
        # Split temporal (importante para séries temporais!)
        split_idx = int(len(self.X) * (1 - test_size))
        
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]
        
        print(f"\n📊 Divisão dos dados:")
        print(f"   Treino: {len(X_train):,} partidas")
        print(f"   Teste: {len(X_test):,} partidas")
        
        # Normalizar features
        print(f"\n⚙️  Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar
        print(f"🤖 Treinando Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Probabilidades
        y_proba_test = self.model.predict_proba(X_test_scaled)
        
        # Métricas
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n" + "="*70)
        print(f"📊 RESULTADOS DO MODELO")
        print("="*70)
        print(f"\n🎯 Acurácia:")
        print(f"   Treino: {train_acc:.1%}")
        print(f"   Teste:  {test_acc:.1%}")
        
        print(f"\n📋 Relatório Detalhado (Conjunto de Teste):")
        print("-"*70)
        target_names = ['Away Win', 'Draw', 'Home Win']
        
        # Verificar classes presentes no teste
        unique_test = np.unique(y_test)
        unique_pred = np.unique(y_pred_test)
        
        if len(unique_test) < 3:
            print(f"⚠️  AVISO: Conjunto de teste tem apenas {len(unique_test)} classe(s): {unique_test}")
            print("   Isso pode indicar dados desbalanceados. Considere usar mais dados.")
            print()
        
        # Usar labels presentes apenas
        labels_present = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
        names_present = [target_names[i] for i in labels_present]
        
        print(classification_report(y_test, y_pred_test, labels=labels_present, 
                                   target_names=names_present, digits=3, zero_division=0))
        
        print(f"\n🔀 Matriz de Confusão:")
        print("-"*70)
        cm = confusion_matrix(y_test, y_pred_test, labels=labels_present)
        print(f"\n              {'Pred Away':>12} {'Pred Draw':>12} {'Pred Home':>12}")
        
        for i, label in enumerate(names_present):
            row_str = f"   Real {label:8s}  "
            for j in range(len(labels_present)):
                if j < cm.shape[1] and i < cm.shape[0]:
                    row_str += f"{cm[i][j]:>10,}  "
            print(row_str)
        
        # Feature importance
        print(f"\n⭐ Top 15 Features Mais Importantes:")
        print("-"*70)
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(15).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"   {row['feature']:30s} {bar} {row['importance']:.4f}")
        
        self.results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test,
            'feature_importances': importances
        }
        
        return self.results
    
    def predict_match(self, home_team_id, away_team_id, league_id, season_type, match_date=None):
        """Prediz o resultado de uma partida específica"""
        if match_date is None:
            match_date = pd.Timestamp.now()
        
        print("\n" + "="*70)
        print("🔮 FAZENDO PREDIÇÃO")
        print("="*70)
        
        # Buscar nomes dos times
        home_team = self.teams[self.teams['teamId'] == home_team_id]
        away_team = self.teams[self.teams['teamId'] == away_team_id]
        
        home_name = home_team['name'].values[0] if len(home_team) > 0 else f"Team {home_team_id}"
        away_name = away_team['name'].values[0] if len(away_team) > 0 else f"Team {away_team_id}"
        
        print(f"\n⚽ Partida: {home_name} vs {away_name}")
        
        features = {}
        
        # Forma recente
        home_form = self.get_recent_form(home_team_id, match_date)
        away_form = self.get_recent_form(away_team_id, match_date)
        
        features['home_recent_ppg'] = home_form['ppg']
        features['home_recent_gf'] = home_form['gf']
        features['home_recent_ga'] = home_form['ga']
        features['home_recent_win_rate'] = home_form['win_rate']
        
        features['away_recent_ppg'] = away_form['ppg']
        features['away_recent_gf'] = away_form['gf']
        features['away_recent_ga'] = away_form['ga']
        features['away_recent_win_rate'] = away_form['win_rate']
        
        # H2H
        h2h = self.get_h2h_stats(home_team_id, away_team_id, match_date)
        features['h2h_home_win_rate'] = h2h['home_win_rate']
        features['h2h_draw_rate'] = h2h['draw_rate']
        features['h2h_matches'] = h2h['games']
        
        # Standings
        home_standing = self.get_standings_features(home_team_id, league_id, season_type)
        away_standing = self.get_standings_features(away_team_id, league_id, season_type)
        
        for key, value in home_standing.items():
            features[f'home_{key}'] = value
        for key, value in away_standing.items():
            features[f'away_{key}'] = value
        
        # Stats
        home_stats = self.get_team_stats_features(home_team_id, season_type)
        away_stats = self.get_team_stats_features(away_team_id, season_type)
        
        for key, value in home_stats.items():
            features[f'home_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_{key}'] = value
        
        # Derivadas
        if home_standing and away_standing:
            features['rank_diff'] = away_standing.get('team_rank', 10) - home_standing.get('team_rank', 10)
            features['points_diff'] = home_standing.get('points', 0) - away_standing.get('points', 0)
            features['goal_diff_diff'] = home_standing.get('goal_diff', 0) - away_standing.get('goal_diff', 0)
            features['form_diff'] = features['home_recent_ppg'] - features['away_recent_ppg']
        
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
        
        print(f"\n🎯 RESULTADO PREVISTO: {result_map[prediction]}")
        print(f"\n📊 Probabilidades:")
        print(f"   {away_name} Win: {probabilities[0]:.1%}")
        print(f"   Draw:           {probabilities[1]:.1%}")
        print(f"   {home_name} Win: {probabilities[2]:.1%}")
        
        # Mostrar contexto
        print(f"\n📈 Contexto da partida:")
        if home_standing:
            print(f"   {home_name}: {home_standing.get('team_rank', 'N/A')}º posição, {home_standing.get('points', 0)} pontos")
        if away_standing:
            print(f"   {away_name}: {away_standing.get('team_rank', 'N/A')}º posição, {away_standing.get('points', 0)} pontos")
        print(f"   H2H últimos jogos: {h2h['games']} confrontos")
        
        return {
            'prediction': result_map[prediction],
            'probabilities': {
                'away_win': probabilities[0],
                'draw': probabilities[1],
                'home_win': probabilities[2]
            },
            'home_team': home_name,
            'away_team': away_name
        }


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    
    print("\n")
    print("="*70)
    print("⚽ SOCCER MATCH OUTCOME PREDICTOR - SISTEMA COMPLETO")
    print("="*70)
    print("\nDesenvolvido para predição de resultados de partidas")
    print("Target: Home Win / Draw / Away Win")
    print("="*70)
    
    # Caminho base
    BASE_PATH = r'C:\Users\Rafaribas\Desktop\Faculdade\Curso\6º período\SI\Projeto-Final\kaggle_data\data'
    
    # Inicializar
    predictor = SoccerMatchPredictor(BASE_PATH)
    
    # Carregar dados
    predictor.load_data()
    
    # Engenharia de features
    # IMPORTANTE: Não use sample_size pequeno! Ou use None para TODOS os dados
    # Se quiser testar rápido, use sample_size=15000 (não 10000)
    predictor.engineer_features(sample_size=15000)  # Ajuste conforme necessário
    
    # Treinar modelo
    results = predictor.train_model(test_size=0.2)
    
    print("\n" + "="*70)
    print("✅ MODELO TREINADO E PRONTO PARA USO!")
    print("="*70)
    
    # Exemplo de predição
    print("\n\n💡 Para fazer predições, use:")
    print("predictor.predict_match(")
    print("    home_team_id=SEU_ID_HOME,")
    print("    away_team_id=SEU_ID_AWAY,")
    print("    league_id=SEU_ID_LIGA,")
    print("    season_type=TIPO_TEMPORADA")
    print(")")
    print("\n💡 Para descobrir IDs de times, use o Data Explorer")