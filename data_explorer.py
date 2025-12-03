"""
Data Explorer - Explora o dataset de futebol
Versão OTIMIZADA para compatibilidade com o sistema preditivo
"""

import pandas as pd
import os
import json
from datetime import datetime

class SoccerDataExplorer:
    
    def __init__(self, base_path):
        self.base_path = os.path.join(base_path, 'base_data')
        self.metadata_path = os.path.join(base_path, 'metadata')
        print("="*80)
        print("🔍 SOCCER DATA EXPLORER - VERSÃO OTIMIZADA")
        print("="*80)
        
        # Criar diretório para metadados
        os.makedirs(self.metadata_path, exist_ok=True)
        
    def load_all_data(self, cache=True):
        """Carrega todos os CSVs disponíveis com cache"""
        cache_file = os.path.join(self.metadata_path, 'explorer_cache.pkl')
        
        if cache and os.path.exists(cache_file):
            print("\n📦 Carregando dados do cache...")
            import pickle
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.fixtures = data['fixtures']
            self.leagues = data['leagues']
            self.teams = data['teams']
            self.standings = data['standings']
            self.team_stats = data['team_stats']
            
            # Criar dicionários para rápido acesso
            self.team_dict = dict(zip(self.teams['teamId'], self.teams['name']))
            self.league_dict = dict(zip(self.leagues['leagueId'], self.leagues['leagueName']))
            
        else:
            print("\n📂 Carregando dados dos arquivos...")
            self.fixtures = pd.read_csv(os.path.join(self.base_path, 'fixtures.csv'))
            self.leagues = pd.read_csv(os.path.join(self.base_path, 'leagues.csv'))
            self.teams = pd.read_csv(os.path.join(self.base_path, 'teams.csv'))
            self.standings = pd.read_csv(os.path.join(self.base_path, 'standings.csv'))
            self.team_stats = pd.read_csv(os.path.join(self.base_path, 'teamStats.csv'))
            
            # Pré-processamento
            self._preprocess_data()
            
            # Salvar cache
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'fixtures': self.fixtures,
                    'leagues': self.leagues,
                    'teams': self.teams,
                    'standings': self.standings,
                    'team_stats': self.team_stats
                }, f)
        
        self._print_summary()
    
    def _preprocess_data(self):
        """Pré-processa os dados"""
        # Converter datas
        self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
        
        # Criar colunas auxiliares
        self.fixtures['year'] = self.fixtures['date'].dt.year
        self.fixtures['month'] = self.fixtures['date'].dt.month
        self.fixtures['day_of_week'] = self.fixtures['date'].dt.dayofweek
        
        # Criar dicionários para rápido acesso
        self.team_dict = dict(zip(self.teams['teamId'], self.teams['name']))
        self.league_dict = dict(zip(self.leagues['leagueId'], self.leagues['leagueName']))
        
        # Criar coluna de resultado
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
    
    def _print_summary(self):
        """Imprime resumo dos dados carregados"""
        print(f"   ✅ Fixtures: {len(self.fixtures):,} partidas")
        print(f"   ✅ Leagues: {len(self.leagues):,} ligas")
        print(f"   ✅ Teams: {len(self.teams):,} times")
        print(f"   ✅ Standings: {len(self.standings):,} registros")
        print(f"   ✅ Team Stats: {len(self.team_stats):,} registros")
    
    def show_dataset_overview(self):
        """Mostra visão geral do dataset"""
        print("\n" + "="*80)
        print("📊 VISÃO GERAL DO DATASET")
        print("="*80)
        
        # Período de dados
        print(f"\n📅 Período dos dados:")
        print(f"   Primeira partida: {self.fixtures['date'].min().strftime('%Y-%m-%d')}")
        print(f"   Última partida: {self.fixtures['date'].max().strftime('%Y-%m-%d')}")
        
        # Partidas finalizadas com placar
        completed = self.fixtures[self.fixtures['result'].notna()]
        print(f"\n✅ Partidas com resultado: {len(completed):,} ({len(completed)/len(self.fixtures)*100:.1f}%)")
        
        # Distribuição de resultados
        result_counts = completed['result'].value_counts()
        print(f"\n🎯 Distribuição histórica de resultados:")
        for result, count in result_counts.items():
            result_name = {
                'H': 'Vitórias Mandante',
                'A': 'Vitórias Visitante',
                'D': 'Empates'
            }.get(result, result)
            print(f"   {result_name}: {count:,} ({count/len(completed)*100:.1f}%)")
        
        # Estatísticas por ano
        print(f"\n📈 Estatísticas por ano:")

        # Garantir que o completed tenha coluna year
        completed = completed.copy()
        completed['year'] = completed['date'].dt.year

        # Detectar coluna de ID da partida
        id_col = 'fixtureId' if 'fixtureId' in completed.columns else ('Rn' if 'Rn' in completed.columns else None)

        if id_col is None:
            print("❌ Nenhuma coluna de ID encontrada (esperado: 'fixtureId' ou 'Rn')")
        else:
            yearly_stats = completed.groupby('year').agg({
                id_col: 'count',
                'result': lambda x: (x == 'H').sum() / len(x) if len(x) > 0 else 0
            }).rename(columns={id_col: 'matches', 'result': 'home_win_rate'})

            for year, row in yearly_stats.tail(5).iterrows():
                print(f"   {year}: {int(row['matches']):,} partidas, {row['home_win_rate']*100:.1f}% vitórias em casa")

    
    def show_top_leagues(self, n=20):
        """Mostra as principais ligas"""
        print("\n" + "="*80)
        print(f"🏆 TOP {n} LIGAS COM MAIS PARTIDAS")
        print("="*80)
        
        # Agrupar por liga - CORRIGIDO
        # Contar usando 'Rn' (ID da partida)
        league_counts = self.fixtures.groupby('leagueId').size().reset_index(name='total_matches')
        
        # Calcular taxa de conclusão
        completed_matches = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) & 
            (self.fixtures['awayTeamScore'].notna())
        ]
        completed_counts = completed_matches.groupby('leagueId').size().reset_index(name='completed_matches')
        
        # Merge
        league_counts = league_counts.merge(
            completed_counts, on='leagueId', how='left'
        ).fillna(0)
        
        # Merge com nomes das ligas
        league_counts = league_counts.merge(
            self.leagues[['leagueId', 'leagueName', 'midsizeName']], 
            on='leagueId', 
            how='left'
        ).sort_values('total_matches', ascending=False)
        
        print(f"\n{'Pos':<5} {'ID':<8} {'Liga':<50} {'Partidas':<12} {'Completas':<12}")
        print("-"*100)
        
        for idx, (_, row) in enumerate(league_counts.head(n).iterrows(), 1):
            league_name = str(row['leagueName'])[:50] if pd.notna(row['leagueName']) else 'N/A'
            completed = int(row['completed_matches'])
            total = int(row['total_matches'])
            print(f"{idx:<5} "
                f"{row['leagueId']:<8} "
                f"{league_name:<50} "
                f"{total:<12,} "
                f"{completed:<12,}")
        
        return league_counts.head(n)
    
    def show_top_teams(self, n=30):
        """Mostra os times com mais partidas"""
        print("\n" + "="*80)
        print(f"⚽ TOP {n} TIMES COM MAIS PARTIDAS")
        print("="*80)
        
        # Contar partidas como mandante e visitante
        home_matches = self.fixtures.groupby('homeTeamId').size().reset_index(name='home_matches')
        away_matches = self.fixtures.groupby('awayTeamId').size().reset_index(name='away_matches')
        
        # Merge e calcular total
        team_matches = home_matches.merge(
            away_matches, 
            left_on='homeTeamId', 
            right_on='awayTeamId', 
            how='outer'
        )
        
        # Limpar e somar
        team_matches['teamId'] = team_matches['homeTeamId'].combine_first(team_matches['awayTeamId'])
        team_matches['home_matches'] = team_matches['home_matches'].fillna(0)
        team_matches['away_matches'] = team_matches['away_matches'].fillna(0)
        team_matches['total_matches'] = team_matches['home_matches'] + team_matches['away_matches']
        
        # Merge com informações dos times
        team_matches = team_matches.merge(
            self.teams[['teamId', 'name', 'location']], 
            on='teamId', 
            how='left'
        ).sort_values('total_matches', ascending=False)
        
        print(f"\n{'Pos':<5} {'ID':<8} {'Time':<40} {'País':<20} {'Partidas':<10} {'(Casa/Fora)':<15}")
        print("-"*103)
        
        for idx, (_, row) in enumerate(team_matches.head(n).iterrows(), 1):
            team_name = str(row['name'])[:40] if pd.notna(row['name']) else 'N/A'
            location = str(row['location'])[:20] if pd.notna(row['location']) else 'N/A'
            print(f"{idx:<5} "
                  f"{row['teamId']:<8} "
                  f"{team_name:<40} "
                  f"{location:<20} "
                  f"{int(row['total_matches']):<10,} "
                  f"({int(row['home_matches'])}/{int(row['away_matches'])})")
        
        return team_matches.head(n)
    
    def search_team(self, search_term):
        """Busca times por nome"""
        print(f"\n🔍 Buscando times com '{search_term}'...")
        print("="*80)
        
        # Buscar por name e displayName
        results = self.teams[
            self.teams['name'].str.contains(search_term, case=False, na=False) |
            self.teams['displayName'].str.contains(search_term, case=False, na=False) |
            self.teams['slug'].str.contains(search_term, case=False, na=False) |
            self.teams['location'].str.contains(search_term, case=False, na=False)
        ]
        
        if len(results) == 0:
            print("❌ Nenhum time encontrado")
            return None
        
        print(f"\n✅ {len(results)} time(s) encontrado(s):\n")
        print(f"{'ID':<8} {'Nome':<40} {'Display Name':<30} {'País':<15}")
        print("-"*98)
        
        for _, team in results.iterrows():
            team_name = str(team['name'])[:40] if pd.notna(team['name']) else 'N/A'
            display_name = str(team['displayName'])[:30] if pd.notna(team['displayName']) else 'N/A'
            location = str(team['location'])[:15] if pd.notna(team['location']) else 'N/A'
            
            print(f"{team['teamId']:<8} {team_name:<40} {display_name:<30} {location:<15}")
            
            # Estatísticas rápidas
            team_stats = self._get_team_quick_stats(team['teamId'])
            if team_stats['total_matches'] > 0:
                print(f"       Estatísticas: {team_stats['total_matches']}J "
                      f"{team_stats['wins']}V {team_stats['draws']}E {team_stats['losses']}D "
                      f"({team_stats['win_rate']:.1%} vitórias)")
        
        return results
    
    def analyze_target_distribution(self, matches: pd.DataFrame):
        """Analisa distribuição do target"""
        results = matches.apply(lambda x: self.create_target(x, raw=True), axis=1)
        dist = results.value_counts()
        total = len(dist)
        
        print(f"\n🎯 Distribuição de Resultados:")
        for result, count in dist.items():
            percentage = count/total*100
            print(f"   {result}: {count:,} ({percentage:.1f}%)")
        
        # Estatísticas por ano - CORRIGIDO
        print(f"\n📈 Estatísticas por ano:")
        if 'date' in matches.columns:
            matches['year'] = pd.to_datetime(matches['date']).dt.year
            # Usar 'Rn' para contar partidas
            yearly_stats = matches.groupby('year').size().reset_index(name='matches')
            yearly_stats['home_win_rate'] = matches.groupby('year')['result'].apply(
                lambda x: (x == 'H').sum() / len(x) if len(x) > 0 else 0
            ).values
            
            for _, row in yearly_stats.tail(5).iterrows():
                print(f"   {int(row['year'])}: {int(row['matches']):,} partidas, {row['home_win_rate']*100:.1f}% vitórias em casa")
    
    def _get_team_quick_stats(self, team_id):
        """Estatísticas rápidas de um time"""
        team_matches = self.fixtures[
            ((self.fixtures['homeTeamId'] == team_id) | 
             (self.fixtures['awayTeamId'] == team_id)) &
            (self.fixtures['result'].notna())
        ]
        
        if len(team_matches) == 0:
            return {'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'win_rate': 0}
        
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['homeTeamId'] == team_id
            result = match['result']
            
            if is_home:
                if result == 'H':
                    wins += 1
                elif result == 'D':
                    draws += 1
                else:
                    losses += 1
            else:
                if result == 'A':
                    wins += 1
                elif result == 'D':
                    draws += 1
                else:
                    losses += 1
        
        total = len(team_matches)
        return {
            'total_matches': total,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / total if total > 0 else 0
        }
    
    def search_league(self, search_term):
        """Busca ligas por nome"""
        print(f"\n🔍 Buscando ligas com '{search_term}'...")
        print("="*80)
        
        # Buscar por leagueName, midsizeName e leagueShortName
        results = self.leagues[
            self.leagues['leagueName'].str.contains(search_term, case=False, na=False) |
            self.leagues['midsizeName'].str.contains(search_term, case=False, na=False) |
            self.leagues['leagueShortName'].str.contains(search_term, case=False, na=False)
        ]
        
        if len(results) == 0:
            print("❌ Nenhuma liga encontrada")
            return None
        
        print(f"\n✅ {len(results)} liga(s) encontrada(s):\n")
        print(f"{'ID':<8} {'Nome Liga':<40} {'Nome Médio':<30} {'Nome Curto':<20}")
        print("-"*103)
        
        for _, league in results.iterrows():
            league_name = str(league['leagueName'])[:40] if pd.notna(league['leagueName']) else 'N/A'
            midsize = str(league['midsizeName'])[:30] if pd.notna(league['midsizeName']) else 'N/A'
            short_name = str(league['leagueShortName'])[:20] if pd.notna(league['leagueShortName']) else 'N/A'
            
            print(f"{league['leagueId']:<8} {league_name:<40} {midsize:<30} {short_name:<20}")
            
            # Contar partidas na liga
            league_matches = len(self.fixtures[self.fixtures['leagueId'] == league['leagueId']])
            completed = len(self.fixtures[
                (self.fixtures['leagueId'] == league['leagueId']) & 
                (self.fixtures['result'].notna())
            ])
            
            print(f"       Partidas: {league_matches:,} total, {completed:,} completas "
                f"({completed/league_matches*100:.1f}%)")
        
        return results
    
    def get_team_info(self, team_id):
        """Informações detalhadas de um time"""
        team = self.teams[self.teams['teamId'] == team_id]
        
        if len(team) == 0:
            print(f"❌ Time ID {team_id} não encontrado")
            return None
        
        team = team.iloc[0]
        
        print(f"\n" + "="*80)
        print(f"⚽ INFORMAÇÕES DETALHADAS DO TIME: {team['name']}")
        print("="*80)
        
        print(f"\n📋 INFORMAÇÕES BÁSICAS:")
        print(f"   🆔 ID: {team['teamId']}")
        print(f"   📍 Local: {team['location']}")
        print(f"   🏷️  Display Name: {team['displayName']}")
        print(f"   🔗 Slug: {team['slug']}")
        if 'color' in team:
            print(f"   🎨 Cor: #{team['color']}")
        
        # Partidas do time
        team_matches = self.fixtures[
            (self.fixtures['homeTeamId'] == team_id) | 
            (self.fixtures['awayTeamId'] == team_id)
        ]
        
        completed = team_matches[team_matches['result'].notna()]
        
        print(f"\n📊 ESTATÍSTICAS DE PARTIDAS:")
        print(f"   Total de partidas: {len(team_matches):,}")
        print(f"   Partidas completas: {len(completed):,} ({len(completed)/len(team_matches)*100:.1f}%)")
        
        if len(completed) > 0:
            # Calcular estatísticas
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0
            
            home_wins = 0
            home_draws = 0
            home_losses = 0
            away_wins = 0
            away_draws = 0
            away_losses = 0
            
            for _, match in completed.iterrows():
                is_home = match['homeTeamId'] == team_id
                
                if is_home:
                    gf = match['homeTeamScore']
                    ga = match['awayTeamScore']
                    
                    if gf > ga:
                        wins += 1
                        home_wins += 1
                    elif gf == ga:
                        draws += 1
                        home_draws += 1
                    else:
                        losses += 1
                        home_losses += 1
                else:
                    gf = match['awayTeamScore']
                    ga = match['homeTeamScore']
                    
                    if gf > ga:
                        wins += 1
                        away_wins += 1
                    elif gf == ga:
                        draws += 1
                        away_draws += 1
                    else:
                        losses += 1
                        away_losses += 1
                
                goals_for += gf
                goals_against += ga
            
            total = len(completed)
            
            print(f"\n🏆 DESEMPENHO:")
            print(f"   Vitórias: {wins} ({wins/total*100:.1f}%)")
            print(f"   Empates:  {draws} ({draws/total*100:.1f}%)")
            print(f"   Derrotas: {losses} ({losses/total*100:.1f}%)")
            
            print(f"\n🏠 CASA:")
            print(f"   {home_wins}V {home_draws}E {home_losses}D "
                  f"({home_wins/(home_wins+home_draws+home_losses)*100:.1f}% vitórias)")
            
            print(f"\n✈️  FORA:")
            print(f"   {away_wins}V {away_draws}E {away_losses}D "
                  f"({away_wins/(away_wins+away_draws+away_losses)*100:.1f}% vitórias)")
            
            print(f"\n⚽ GOLS:")
            print(f"   Marcados: {goals_for} ({goals_for/total:.2f} por jogo)")
            print(f"   Sofridos: {goals_against} ({goals_against/total:.2f} por jogo)")
            print(f"   Saldo:    {goals_for - goals_against:+d} ({goals_for/total - goals_against/total:+.2f} por jogo)")
            
            # Forma recente (últimos 5 jogos)
            recent_matches = completed.sort_values('date', ascending=False).head(5)
            print(f"\n📈 FORMA RECENTE (últimos 5 jogos):")
            recent_form = []
            for _, match in recent_matches.iterrows():
                is_home = match['homeTeamId'] == team_id
                opponent_id = match['awayTeamId'] if is_home else match['homeTeamId']
                opponent_name = self.team_dict.get(opponent_id, f"Time {opponent_id}")
                
                if is_home:
                    result = 'W' if match['result'] == 'H' else ('D' if match['result'] == 'D' else 'L')
                    score = f"{match['homeTeamScore']}-{match['awayTeamScore']}"
                else:
                    result = 'W' if match['result'] == 'A' else ('D' if match['result'] == 'D' else 'L')
                    score = f"{match['awayTeamScore']}-{match['homeTeamScore']}"
                
                recent_form.append(result)
                print(f"   {result} vs {opponent_name[:20]:<20} {score}")
            
            # Pontos por jogo recente
            ppg_recent = (recent_form.count('W') * 3 + recent_form.count('D')) / len(recent_form)
            print(f"   Pontos por jogo (últimos 5): {ppg_recent:.2f}")
        
        # Ligas que participa
        leagues_participated = team_matches['leagueId'].unique()
        print(f"\n🏆 LIGAS PARTICIPADAS ({len(leagues_participated)}):")
        for league_id in leagues_participated[:10]:  # Mostrar no máximo 10
            league_name = self.league_dict.get(league_id, f"Liga {league_id}")
            matches_in_league = len(team_matches[team_matches['leagueId'] == league_id])
            print(f"   {league_name[:40]:<40} - {matches_in_league} partidas")
        
        if len(leagues_participated) > 10:
            print(f"   ... e mais {len(leagues_participated) - 10} ligas")
        
        # Estatísticas avançadas do teamStats
        team_stats = self.team_stats[self.team_stats['teamId'] == team_id]
        if len(team_stats) > 0:
            print(f"\n📈 ESTATÍSTICAS AVANÇADAS:")
            stats = team_stats.iloc[0]
            
            if 'possessionPct' in stats:
                print(f"   Posse de bola: {stats['possessionPct']:.1f}%")
            if 'totalShots' in stats and 'shotsOnTarget' in stats:
                accuracy = stats['shotsOnTarget'] / stats['totalShots'] * 100 if stats['totalShots'] > 0 else 0
                print(f"   Chutes: {stats['totalShots']} total, {stats['shotsOnTarget']} no alvo ({accuracy:.1f}%)")
            if 'passAccuracy' in stats:
                print(f"   Precisão de passes: {stats['passAccuracy']:.1f}%")
        
        # Save team info for predictor
        self._save_team_info(team_id)
        
        return team
    
    def _save_team_info(self, team_id):
        """Salva informações do time para uso no predictor"""
        team_info = {
            'team_id': team_id,
            'name': self.team_dict.get(team_id, 'Unknown'),
            'last_accessed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Salvar em arquivo JSON
        info_file = os.path.join(self.metadata_path, f'team_{team_id}_info.json')
        with open(info_file, 'w') as f:
            json.dump(team_info, f, indent=2)
    
    def get_league_standings(self, league_id, season_type='Regular Season'):
        """Mostra a classificação de uma liga"""
        # Buscar info da liga
        league = self.leagues[self.leagues['leagueId'] == league_id]
        
        if len(league) == 0:
            print(f"❌ Liga ID {league_id} não encontrada")
            return None
        
        league_name = league.iloc[0]['leagueName']
        
        print(f"\n" + "="*80)
        print(f"🏆 CLASSIFICAÇÃO: {league_name}")
        print("="*80)
        
        # Filtrar standings para a season_type mais recente
        league_standings = self.standings[self.standings['leagueId'] == league_id]
        
        if len(league_standings) == 0:
            print("❌ Classificação não disponível para esta liga")
            return None
        
        # Se season_type for especificado, filtrar
        if season_type:
            standings = league_standings[league_standings['seasonType'] == season_type]
        else:
            # Pegar a season_type mais recente
            season_types = league_standings['seasonType'].unique()
            standings = league_standings[league_standings['seasonType'] == season_types[-1]]
        
        # Pegar o ano mais recente
        max_year = standings['year'].max()
        standings = standings[standings['year'] == max_year].sort_values('teamRank')
        
        if len(standings) == 0:
            print("❌ Classificação não disponível")
            return None
        
        # Merge com nomes dos times
        standings_with_teams = standings.merge(
            self.teams[['teamId', 'name']], 
            on='teamId', 
            how='left'
        )
        
        # Calcular pontos por jogo
        standings_with_teams['ppg'] = standings_with_teams['points'] / standings_with_teams['gamesPlayed']
        standings_with_teams['goal_ratio'] = standings_with_teams['gf'] / (standings_with_teams['ga'] + 0.1)
        
        print(f"\nTemporada: {max_year}, Tipo: {standings.iloc[0]['seasonType']}")
        print(f"{'Pos':<5} {'Time':<30} {'PJ':<4} {'V':<4} {'E':<4} {'D':<4} {'GF':<5} {'GC':<5} {'SG':<6} {'Pts':<5} {'PPG':<6} {'GR':<6}")
        print("-"*95)
        
        for _, row in standings_with_teams.iterrows():
            team_name = str(row['name'])[:30] if pd.notna(row['name']) else 'N/A'
            print(f"{int(row['teamRank']):<5} "
                  f"{team_name:<30} "
                  f"{int(row['gamesPlayed']):<4} "
                  f"{int(row['wins']):<4} "
                  f"{int(row['ties']):<4} "
                  f"{int(row['losses']):<4} "
                  f"{int(row['gf']):<5} "
                  f"{int(row['ga']):<5} "
                  f"{int(row['gd']):<6} "
                  f"{int(row['points']):<5} "
                  f"{row['ppg']:<6.2f} "
                  f"{row['goal_ratio']:<6.2f}")
        
        # Estatísticas da liga
        print(f"\n📊 ESTATÍSTICAS DA LIGA:")
        print(f"   Média de gols por jogo: {standings_with_teams['gf'].sum() / standings_with_teams['gamesPlayed'].sum():.2f}")
        print(f"   Taxa de vitória em casa estimada: {(standings_with_teams['wins'].sum() / standings_with_teams['gamesPlayed'].sum() * 100):.1f}%")
        
        return standings_with_teams
    
    def find_upcoming_matches(self, team_id=None, league_id=None, days_ahead=30, limit=20):
        """Encontra próximas partidas (sem resultado ainda)"""
        print(f"\n" + "="*80)
        print(f"📅 PRÓXIMAS PARTIDAS (próximos {days_ahead} dias)")
        print("="*80)
        
        today = pd.Timestamp.now()
        future_date = today + pd.Timedelta(days=days_ahead)
        
        # Filtrar partidas futuras
        upcoming = self.fixtures[
            (self.fixtures['date'] >= today) &
            (self.fixtures['date'] <= future_date)
        ].copy()
        
        upcoming = upcoming.sort_values('date')
        
        if team_id:
            upcoming = upcoming[
                (upcoming['homeTeamId'] == team_id) |
                (upcoming['awayTeamId'] == team_id)
            ]
        
        if league_id:
            upcoming = upcoming[upcoming['leagueId'] == league_id]
        
        if len(upcoming) == 0:
            print(f"❌ Nenhuma partida encontrada para os próximos {days_ahead} dias")
            return None
        
        upcoming = upcoming.head(limit)
        
        # Merge com nomes
        upcoming_with_names = upcoming.merge(
            self.teams[['teamId', 'name']].rename(columns={'teamId': 'homeTeamId', 'name': 'home_name'}),
            on='homeTeamId',
            how='left'
        ).merge(
            self.teams[['teamId', 'name']].rename(columns={'teamId': 'awayTeamId', 'name': 'away_name'}),
            on='awayTeamId',
            how='left'
        ).merge(
            self.leagues[['leagueId', 'leagueName']],
            on='leagueId',
            how='left'
        )
        
        print(f"\n{'Data':<12} {'Dia':<4} {'Liga':<25} {'Mandante':<25} vs {'Visitante':<25}")
        print("-"*105)
        
        for _, match in upcoming_with_names.iterrows():
            date_str = pd.to_datetime(match['date']).strftime('%m-%d')
            day_name = pd.to_datetime(match['date']).strftime('%a')
            league = str(match['leagueName'])[:25] if pd.notna(match['leagueName']) else 'N/A'
            home = str(match['home_name'])[:25] if pd.notna(match['home_name']) else 'N/A'
            away = str(match['away_name'])[:25] if pd.notna(match['away_name']) else 'N/A'
            
            print(f"{date_str:<12} {day_name:<4} {league:<25} {home:<25} vs {away:<25}")
            
            # IDs para predição
            print(f"             IDs: Home={match['homeTeamId']}, Away={match['awayTeamId']}, "
                  f"League={match['leagueId']}, SeasonType={match.get('seasonType', 2)}")
            print()
        
        # Salvar próximas partidas para predição
        self._save_upcoming_matches(upcoming_with_names)
        
        return upcoming_with_names
    
    def _save_upcoming_matches(self, matches_df):
        """Salva próximas partidas para predição em lote"""
        matches_file = os.path.join(self.metadata_path, 'upcoming_matches.csv')
        
        # Preparar dados para predição - CORRIGIDO
        # Verificar quais colunas existem
        available_cols = matches_df.columns.tolist()
        
        # Colunas básicas obrigatórias
        base_cols = ['Rn', 'date', 'homeTeamId', 'awayTeamId', 'leagueId', 'seasonType']
        cols_to_include = [col for col in base_cols if col in available_cols]
        
        # Adicionar colunas de nome se disponíveis
        optional_cols = ['home_name', 'away_name', 'leagueName']
        for col in optional_cols:
            if col in available_cols:
                cols_to_include.append(col)
        
        prediction_data = matches_df[cols_to_include].copy()
        
        prediction_data.to_csv(matches_file, index=False)
        print(f"💾 Próximas partidas salvas em: {matches_file}")

    def debug_all_columns(self):
        """Mostra todas as colunas de todos os dataframes"""
        print("\n" + "="*80)
        print("🔍 COLUNAS DISPONÍVEIS EM TODOS OS DATAFRAMES")
        print("="*80)
        
        dataframes = {
            'fixtures': self.fixtures,
            'leagues': self.leagues,
            'teams': self.teams,
            'standings': self.standings,
            'team_stats': self.team_stats
        }
        
        for name, df in dataframes.items():
            print(f"\n📄 {name.upper()} ({len(df.columns)} colunas):")
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                total = len(df)
                print(f"   - {col}: {dtype} ({non_null}/{total} não nulos)")

    def debug_columns(self):
        """Debug das colunas disponíveis"""
        print("\n" + "="*70)
        print("DEBUG - COLUNAS DISPONÍVEIS")
        print("="*70)
        
        print(f"\nFixtures columns: {self.fixtures.columns.tolist()}")
        print(f"\nLeagues columns: {self.leagues.columns.tolist()}")
        print(f"\nTeams columns: {self.teams.columns.tolist()}")
        print(f"\nStandings columns: {self.standings.columns.tolist()}")
        print(f"\nTeam Stats columns: {self.team_stats.columns.tolist()}")
    
    def get_team_fixtures(self, team_id, limit=10):
        """Obtém as próximas partidas de um time específico"""
        print(f"\n" + "="*80)
        print(f"📅 PARTIDAS DO TIME: {self.team_dict.get(team_id, f'Time {team_id}')}")
        print("="*80)
        
        team_matches = self.fixtures[
            ((self.fixtures['homeTeamId'] == team_id) | 
             (self.fixtures['awayTeamId'] == team_id))
        ].sort_values('date')
        
        # Separar passado e futuro
        today = pd.Timestamp.now()
        past_matches = team_matches[team_matches['date'] < today]
        future_matches = team_matches[team_matches['date'] >= today]
        
        print(f"\n⏪ ÚLTIMAS {min(limit, len(past_matches))} PARTIDAS:")
        if len(past_matches) > 0:
            self._display_team_matches(past_matches.tail(limit), team_id)
        else:
            print("   Nenhuma partida passada encontrada")
        
        print(f"\n⏩ PRÓXIMAS {min(limit, len(future_matches))} PARTIDAS:")
        if len(future_matches) > 0:
            self._display_team_matches(future_matches.head(limit), team_id)
        else:
            print("   Nenhuma partida futura encontrada")
        
        return {
            'past': past_matches.tail(limit),
            'future': future_matches.head(limit)
        }
    
    def _display_team_matches(self, matches_df, team_id):
        """Display helper para partidas de um time"""
        for _, match in matches_df.iterrows():
            is_home = match['homeTeamId'] == team_id
            opponent_id = match['awayTeamId'] if is_home else match['homeTeamId']
            opponent_name = self.team_dict.get(opponent_id, f"Time {opponent_id}")
            
            date_str = pd.to_datetime(match['date']).strftime('%Y-%m-%d')
            venue = "(Casa)" if is_home else "(Fora)"
            
            if pd.notna(match['result']):
                # Partida já aconteceu
                if is_home:
                    score = f"{match['homeTeamScore']}-{match['awayTeamScore']}"
                    result = 'W' if match['result'] == 'H' else ('D' if match['result'] == 'D' else 'L')
                else:
                    score = f"{match['awayTeamScore']}-{match['homeTeamScore']}"
                    result = 'W' if match['result'] == 'A' else ('D' if match['result'] == 'D' else 'L')
                
                print(f"   {date_str} {venue:<8} {result} vs {opponent_name[:25]:<25} {score}")
            else:
                # Partida futura
                print(f"   {date_str} {venue:<8}   vs {opponent_name[:25]:<25}")
    
    def generate_prediction_input(self, team_id=None, league_id=None):
        """Gera arquivo de entrada para predições em lote"""
        print("\n" + "="*80)
        print("🤖 GERANDO ARQUIVO PARA PREDIÇÕES EM LOTE")
        print("="*80)
        
        # Encontrar próximas partidas
        upcoming = self.find_upcoming_matches(team_id, league_id, days_ahead=14, limit=50)
        
        if upcoming is None or len(upcoming) == 0:
            print("❌ Nenhuma partida encontrada para gerar predições")
            return None
        
        # Criar arquivo de configuração para predições - CORRIGIDO
        prediction_config = []
        
        for _, match in upcoming.iterrows():
            config = {
                'Rn': int(match['Rn']),  # Usar 'Rn' em vez de 'fixtureId'
                'homeTeamId': int(match['homeTeamId']),
                'awayTeamId': int(match['awayTeamId']),
                'leagueId': int(match['leagueId']),
                'seasonType': int(match.get('seasonType', 2)),
                'date': match['date'].strftime('%Y-%m-%d'),
                'homeTeam': str(match['home_name']),
                'awayTeam': str(match['away_name']),
                'league': str(match['leagueName'])
            }
            prediction_config.append(config)
        
        # Salvar como JSON
        config_file = os.path.join(self.metadata_path, 'prediction_batch.json')
        with open(config_file, 'w') as f:
            json.dump(prediction_config, f, indent=2)
        
        print(f"\n✅ Arquivo de predições gerado: {config_file}")
        print(f"   Total de partidas: {len(prediction_config)}")
        
        return config_file


# =============================================================================
# INTERFACE DE LINHA DE COMANDO
# =============================================================================

def run_interactive_explorer(base_path):
    """Interface interativa para o Data Explorer"""
    explorer = SoccerDataExplorer(base_path)
    
    print("\n" + "="*80)
    print("🔍 SOCCER DATA EXPLORER - MODO INTERATIVO")
    print("="*80)
    print("\nCarregando dados...")
    
    explorer.load_all_data(cache=True)
    
    while True:
        print("\n" + "="*80)
        print("📋 MENU PRINCIPAL")
        print("="*80)
        print("\n  1 - 📊 Visão geral do dataset")
        print("  2 - 🏆 Top ligas")
        print("  3 - ⚽ Top times")
        print("  4 - 🔍 Buscar time")
        print("  5 - 🔍 Buscar liga")
        print("  6 - 📈 Info detalhada de time")
        print("  7 - 🏅 Classificação de liga")
        print("  8 - 📅 Próximas partidas")
        print("  9 - ⏳ Histórico de time")
        print(" 10 - 🤖 Gerar predições em lote")
        print(" 11 - 🔧 Debug todas as colunas")
        print("  0 - ❌ Sair")
        
        try:
            choice = input("\nEscolha uma opção: ").strip()
            
            if choice == '1':
                explorer.show_dataset_overview()
            
            elif choice == '2':
                n = input("Quantas ligas mostrar? (default 20): ").strip()
                n = int(n) if n else 20
                explorer.show_top_leagues(n)
            
            elif choice == '3':
                n = input("Quantos times mostrar? (default 30): ").strip()
                n = int(n) if n else 30
                explorer.show_top_teams(n)
            
            elif choice == '4':
                term = input("Termo de busca para time: ").strip()
                if term:
                    explorer.search_team(term)
            
            elif choice == '5':
                term = input("Termo de busca para liga: ").strip()
                if term:
                    explorer.search_league(term)
            
            elif choice == '6':
                team_id = input("ID do time: ").strip()
                if team_id:
                    try:
                        explorer.get_team_info(int(team_id))
                    except ValueError:
                        print("❌ ID inválido")
            
            elif choice == '7':
                league_id = input("ID da liga: ").strip()
                if league_id:
                    try:
                        season_type = input("Tipo de temporada (deixe em branco para mais recente): ").strip()
                        season_type = season_type if season_type else None
                        explorer.get_league_standings(int(league_id), season_type)
                    except ValueError:
                        print("❌ ID inválido")
            
            elif choice == '8':
                print("\nFiltros (deixe em branco para pular):")
                team_id = input("ID do time: ").strip()
                league_id = input("ID da liga: ").strip()
                days = input("Dias à frente (default 30): ").strip()
                limit = input("Quantidade (default 20): ").strip()
                
                team_id = int(team_id) if team_id else None
                league_id = int(league_id) if league_id else None
                days = int(days) if days else 30
                limit = int(limit) if limit else 20
                
                explorer.find_upcoming_matches(team_id, league_id, days, limit)
            
            elif choice == '9':
                team_id = input("ID do time: ").strip()
                if team_id:
                    try:
                        limit = input("Quantas partidas mostrar? (default 10): ").strip()
                        limit = int(limit) if limit else 10
                        explorer.get_team_fixtures(int(team_id), limit)
                    except ValueError:
                        print("❌ ID inválido")
            
            elif choice == '10':
                print("\nFiltros para predição em lote:")
                team_id = input("ID do time (opcional): ").strip()
                league_id = input("ID da liga (opcional): ").strip()
                
                team_id = int(team_id) if team_id else None
                league_id = int(league_id) if league_id else None
                
                explorer.generate_prediction_input(team_id, league_id)

            elif choice == '11':
                explorer.debug_all_columns()

            elif choice == '0':
                print("\n👋 Até logo!")
                break
            
            else:
                print("❌ Opção inválida")
        
        except KeyboardInterrupt:
            print("\n\n👋 Encerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explorador de Dados de Futebol')
    parser.add_argument('--path', type=str, required=True, help='Caminho base dos dados')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_explorer(args.path)
    else:
        # Execução padrão
        explorer = SoccerDataExplorer(args.path)
        explorer.load_all_data(cache=True)
        explorer.show_dataset_overview()
        explorer.show_top_leagues(15)
        explorer.show_top_teams(20)