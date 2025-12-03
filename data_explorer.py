"""
Data Explorer - Explora o dataset de futebol
Versão CORRIGIDA com as colunas reais do dataset
"""

from pathlib import Path
import pandas as pd
import os

class SoccerDataExplorer:
    
    def __init__(self, base_path):
        self.base_path = os.path.join(base_path, 'base_data')
        print("="*80)
        print("🔍 SOCCER DATA EXPLORER")
        print("="*80)
        
    def load_all_data(self):
        """Carrega todos os CSVs disponíveis"""
        print("\n📂 Carregando dados...")
        
        self.fixtures = pd.read_csv(os.path.join(self.base_path, 'fixtures.csv'))
        self.leagues = pd.read_csv(os.path.join(self.base_path, 'leagues.csv'))
        self.teams = pd.read_csv(os.path.join(self.base_path, 'teams.csv'))
        self.standings = pd.read_csv(os.path.join(self.base_path, 'standings.csv'))
        self.team_stats = pd.read_csv(os.path.join(self.base_path, 'teamStats.csv'))
        
        try:
            self.players = pd.read_csv(os.path.join(self.base_path, 'players.csv'), low_memory=False)
        except:
            self.players = None
            
        try:
            self.team_roster = pd.read_csv(os.path.join(self.base_path, 'teamRoster.csv'))
        except:
            self.team_roster = None
        
        print(f"   ✅ Fixtures: {len(self.fixtures):,} partidas")
        print(f"   ✅ Leagues: {len(self.leagues):,} ligas")
        print(f"   ✅ Teams: {len(self.teams):,} times")
        print(f"   ✅ Standings: {len(self.standings):,} registros")
        print(f"   ✅ Team Stats: {len(self.team_stats):,} registros")
        if self.players is not None:
            print(f"   ✅ Players: {len(self.players):,} jogadores")
        if self.team_roster is not None:
            print(f"   ✅ Team Roster: {len(self.team_roster):,} registros")
    
    def show_dataset_overview(self):
        """Mostra visão geral do dataset"""
        print("\n" + "="*80)
        print("📊 VISÃO GERAL DO DATASET")
        print("="*80)
        
        # Período de dados
        self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
        print(f"\n📅 Período dos dados:")
        print(f"   Primeira partida: {self.fixtures['date'].min()}")
        print(f"   Última partida: {self.fixtures['date'].max()}")
        
        # Partidas finalizadas com placar
        completed = self.fixtures[
            (self.fixtures['homeTeamScore'].notna()) & 
            (self.fixtures['awayTeamScore'].notna())
        ]
        print(f"\n✅ Partidas com placar: {len(completed):,} ({len(completed)/len(self.fixtures)*100:.1f}%)")
        
        # Distribuição de resultados
        home_wins = (completed['homeTeamScore'] > completed['awayTeamScore']).sum()
        draws = (completed['homeTeamScore'] == completed['awayTeamScore']).sum()
        away_wins = (completed['homeTeamScore'] < completed['awayTeamScore']).sum()
        
        print(f"\n🎯 Distribuição histórica de resultados:")
        print(f"   Vitórias Mandante: {home_wins:,} ({home_wins/len(completed)*100:.1f}%)")
        print(f"   Empates:           {draws:,} ({draws/len(completed)*100:.1f}%)")
        print(f"   Vitórias Visitante: {away_wins:,} ({away_wins/len(completed)*100:.1f}%)")
    
    def show_top_leagues(self, n=20):
        """Mostra as principais ligas"""
        print("\n" + "="*80)
        print(f"🏆 TOP {n} LIGAS COM MAIS PARTIDAS")
        print("="*80)
        
        # Usar leagueName que existe no CSV
        fixtures_with_league = self.fixtures.merge(
            self.leagues[['leagueId', 'leagueName']], 
            on='leagueId', 
            how='left'
        )
        
        league_counts = fixtures_with_league.groupby(['leagueId', 'leagueName']).size().reset_index(name='matches')
        league_counts = league_counts.sort_values('matches', ascending=False)
        
        print(f"\n{'ID':<8} {'Liga':<60} {'Partidas':<10}")
        print("-"*83)
        for _, row in league_counts.head(n).iterrows():
            league_name = str(row['leagueName']) if pd.notna(row['leagueName']) else 'N/A'
            print(f"{row['leagueId']:<8} {league_name[:60]:<60} {row['matches']:<10,}")
        
        return league_counts.head(n)
    
    def show_top_teams(self, n=30):
        """Mostra os times com mais partidas"""
        print("\n" + "="*80)
        print(f"⚽ TOP {n} TIMES COM MAIS PARTIDAS")
        print("="*80)
        
        # Contar partidas como mandante e visitante
        home_matches = self.fixtures.groupby('homeTeamId').size()
        away_matches = self.fixtures.groupby('awayTeamId').size()
        total_matches = home_matches.add(away_matches, fill_value=0).reset_index()
        total_matches.columns = ['teamId', 'total_matches']
        
        # Merge com nomes dos times (usar 'name' que existe)
        teams_with_matches = total_matches.merge(
            self.teams[['teamId', 'name', 'location']], 
            on='teamId', 
            how='left'
        )
        teams_with_matches = teams_with_matches.sort_values('total_matches', ascending=False)
        
        print(f"\n{'ID':<8} {'Time':<40} {'País':<20} {'Partidas':<10}")
        print("-"*83)
        for _, row in teams_with_matches.head(n).iterrows():
            team_name = str(row['name']) if pd.notna(row['name']) else 'N/A'
            location = str(row['location']) if pd.notna(row['location']) else 'N/A'
            print(f"{row['teamId']:<8} {team_name[:40]:<40} {location[:20]:<20} {int(row['total_matches']):<10,}")
        
        return teams_with_matches.head(n)
    
    def search_team(self, search_term):
        """Busca times por nome"""
        print(f"\n🔍 Buscando times com '{search_term}'...")
        print("="*80)
        
        # Buscar por name e displayName
        results = self.teams[
            self.teams['name'].str.contains(search_term, case=False, na=False) |
            self.teams['displayName'].str.contains(search_term, case=False, na=False) |
            self.teams['slug'].str.contains(search_term, case=False, na=False)
        ]
        
        if len(results) == 0:
            print("❌ Nenhum time encontrado")
            return None
        
        print(f"\n✅ {len(results)} time(s) encontrado(s):\n")
        print(f"{'ID':<8} {'Nome':<45} {'País':<20}")
        print("-"*78)
        for _, team in results.iterrows():
            team_name = str(team['name']) if pd.notna(team['name']) else 'N/A'
            location = str(team['location']) if pd.notna(team['location']) else 'N/A'
            print(f"{team['teamId']:<8} {team_name[:45]:<45} {location[:20]:<20}")
        
        return results
    
    def search_league(self, search_term):
        """Busca ligas por nome"""
        print(f"\n🔍 Buscando ligas com '{search_term}'...")
        print("="*80)
        
        # Buscar por leagueName e midsizeName
        results = self.leagues[
            self.leagues['leagueName'].str.contains(search_term, case=False, na=False) |
            self.leagues['midsizeName'].str.contains(search_term, case=False, na=False)
        ]
        
        if len(results) == 0:
            print("❌ Nenhuma liga encontrada")
            return None
        
        print(f"\n✅ {len(results)} liga(s) encontrada(s):\n")
        print(f"{'ID':<8} {'Nome':<50} {'Código':<20}")
        print("-"*83)
        for _, league in results.iterrows():
            league_name = str(league['leagueName']) if pd.notna(league['leagueName']) else 'N/A'
            midsize = str(league['midsizeName']) if pd.notna(league['midsizeName']) else 'N/A'
            print(f"{league['leagueId']:<8} {league_name[:50]:<50} {midsize[:20]:<20}")
        
        return results
    
    def get_team_info(self, team_id):
        """Informações detalhadas de um time"""
        team = self.teams[self.teams['teamId'] == team_id]
        
        if len(team) == 0:
            print(f"❌ Time ID {team_id} não encontrado")
            return None
        
        team = team.iloc[0]
        
        print(f"\n" + "="*80)
        print(f"⚽ INFORMAÇÕES DO TIME: {team['name']}")
        print("="*80)
        print(f"\n🆔 ID: {team['teamId']}")
        print(f"📍 Local: {team['location']}")
        print(f"🏷️  Display Name: {team['displayName']}")
        print(f"🔗 Slug: {team['slug']}")
        print(f"🎨 Cor: #{team['color']}")
        
        # Partidas do time
        team_matches = self.fixtures[
            (self.fixtures['homeTeamId'] == team_id) | 
            (self.fixtures['awayTeamId'] == team_id)
        ]
        
        print(f"\n📊 Estatísticas:")
        print(f"   Total de partidas: {len(team_matches):,}")
        
        completed = team_matches[
            (team_matches['homeTeamScore'].notna()) & 
            (team_matches['awayTeamScore'].notna())
        ]
        print(f"   Partidas completas: {len(completed):,}")
        
        # Vitórias/Empates/Derrotas
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in completed.iterrows():
            is_home = match['homeTeamId'] == team_id
            
            if is_home:
                gf = match['homeTeamScore']
                ga = match['awayTeamScore']
            else:
                gf = match['awayTeamScore']
                ga = match['homeTeamScore']
            
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                wins += 1
            elif gf == ga:
                draws += 1
            else:
                losses += 1
        
        if len(completed) > 0:
            print(f"\n🏆 Desempenho:")
            print(f"   Vitórias: {wins} ({wins/len(completed)*100:.1f}%)")
            print(f"   Empates:  {draws} ({draws/len(completed)*100:.1f}%)")
            print(f"   Derrotas: {losses} ({losses/len(completed)*100:.1f}%)")
            print(f"\n⚽ Gols:")
            print(f"   Marcados: {goals_for} ({goals_for/len(completed):.2f} por jogo)")
            print(f"   Sofridos: {goals_against} ({goals_against/len(completed):.2f} por jogo)")
            print(f"   Saldo:    {goals_for - goals_against:+d}")
        
        # Ligas que participa
        leagues_participated = team_matches['leagueId'].unique()
        print(f"\n🏆 Participa de {len(leagues_participated)} liga(s)")
        
        return team
    
    def get_league_standings(self, league_id, season_type=None):
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
        
        # Filtrar standings
        if season_type:
            standings = self.standings[
                (self.standings['leagueId'] == league_id) &
                (self.standings['seasonType'] == season_type)
            ].sort_values('teamRank')
        else:
            # Pegar ano mais recente
            max_year = self.standings[self.standings['leagueId'] == league_id]['year'].max()
            standings = self.standings[
                (self.standings['leagueId'] == league_id) &
                (self.standings['year'] == max_year)
            ].sort_values('teamRank')
        
        if len(standings) == 0:
            print("❌ Classificação não disponível")
            return None
        
        # Merge com nomes dos times
        standings_with_teams = standings.merge(
            self.teams[['teamId', 'name']], 
            on='teamId', 
            how='left'
        )
        
        print(f"\n{'Pos':<5} {'Time':<35} {'PJ':<5} {'V':<5} {'E':<5} {'D':<5} {'GF':<5} {'GC':<5} {'SG':<6} {'Pts':<5}")
        print("-"*90)
        
        for _, row in standings_with_teams.iterrows():
            team_name = str(row['name']) if pd.notna(row['name']) else 'N/A'
            print(f"{int(row['teamRank']):<5} "
                  f"{team_name[:35]:<35} "
                  f"{int(row['gamesPlayed']):<5} "
                  f"{int(row['wins']):<5} "
                  f"{int(row['ties']):<5} "
                  f"{int(row['losses']):<5} "
                  f"{int(row['gf']):<5} "
                  f"{int(row['ga']):<5} "
                  f"{int(row['gd']):<6} "
                  f"{int(row['points']):<5}")
        
        return standings_with_teams
    
    def find_upcoming_matches(self, team_id=None, league_id=None, limit=10):
        """Encontra próximas partidas (sem resultado ainda)"""
        print(f"\n" + "="*80)
        print(f"📅 PRÓXIMAS PARTIDAS")
        print("="*80)
        
        # Filtrar partidas futuras ou sem placar
        upcoming = self.fixtures[
            (self.fixtures['homeTeamScore'].isna()) |
            (self.fixtures['awayTeamScore'].isna())
        ].copy()
        
        if 'date' not in upcoming.columns or upcoming['date'].dtype != 'datetime64[ns]':
            upcoming['date'] = pd.to_datetime(upcoming['date'])
        
        upcoming = upcoming.sort_values('date')
        
        if team_id:
            upcoming = upcoming[
                (upcoming['homeTeamId'] == team_id) |
                (upcoming['awayTeamId'] == team_id)
            ]
        
        if league_id:
            upcoming = upcoming[upcoming['leagueId'] == league_id]
        
        upcoming = upcoming.head(limit)
        
        if len(upcoming) == 0:
            print("❌ Nenhuma partida futura encontrada")
            return None
        
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
            self.leagues[['leagueId', 'leagueName']].rename(columns={'leagueName': 'league_name'}),
            on='leagueId',
            how='left'
        )
        
        print(f"\n{'Data':<12} {'Liga':<25} {'Mandante':<25} vs {'Visitante':<25}")
        print("-"*95)
        
        for _, match in upcoming_with_names.iterrows():
            date_str = pd.to_datetime(match['date']).strftime('%Y-%m-%d')
            league = str(match['league_name'])[:25] if pd.notna(match['league_name']) else 'N/A'
            home = str(match['home_name'])[:25] if pd.notna(match['home_name']) else 'N/A'
            away = str(match['away_name'])[:25] if pd.notna(match['away_name']) else 'N/A'
            
            print(f"{date_str:<12} {league:<25} {home:<25} vs {away:<25}")
            print(f"             IDs: Home={match['homeTeamId']}, Away={match['awayTeamId']}, League={match['leagueId']}")
            print()
        
        return upcoming_with_names


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    
    # Caminho base
    BASE_PATH = Path(__file__).parent / "data"
    
    # Inicializar explorador
    explorer = SoccerDataExplorer(BASE_PATH)
    
    # Carregar dados
    explorer.load_all_data()
    
    # Visão geral
    explorer.show_dataset_overview()
    
    # Top ligas
    explorer.show_top_leagues(20)
    
    # Top times
    explorer.show_top_teams(30)
    
    # Exemplos de buscas
    print("\n\n" + "="*80)
    print("💡 EXEMPLOS DE BUSCA")
    print("="*80)
    print("\n# Descomente para testar:")
    print("# explorer.search_team('Real Madrid')")
    print("# explorer.search_team('Flamengo')")
    print("# explorer.search_league('Premier')")
    print("# explorer.search_league('Champions')")
    print("# explorer.get_team_info(86)")
    print("# explorer.get_league_standings(140)")