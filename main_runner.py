"""
SCRIPT PRINCIPAL - Sistema de Predição de Futebol OTIMIZADO
Execute este arquivo para treinar o modelo e fazer predições

Uso:
1. python main_runner.py --explore
   (para conhecer times e ligas disponíveis)

2. python main_runner.py --train
   (para treinar o modelo)

3. python main_runner.py --predict
   (para fazer predições)

4. python main_runner.py --menu
   (para menu interativo)
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_PATH = Path(__file__).parent / "data"

# =============================================================================
# MODO 1: EXPLORAÇÃO DE DADOS OTIMIZADA
# =============================================================================

def run_exploration():
    """Explora o dataset para conhecer times e ligas"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "MODO: EXPLORAÇÃO DE DADOS OTIMIZADA" + " " * 27 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # CORREÇÃO: Use data_explorer em vez de data_explorer_optimized
    from data_explorer import SoccerDataExplorer
    
    explorer = SoccerDataExplorer(BASE_PATH)
    explorer.load_all_data(cache=True)
    
    # Menu de exploração
    while True:
        print("\n" + "="*80)
        print("🔍 MENU DE EXPLORAÇÃO")
        print("="*80)
        print("\nOpções:")
        print("  1 - 📊 Visão geral do dataset")
        print("  2 - 🏆 Top ligas")
        print("  3 - ⚽ Top times")
        print("  4 - 🔍 Buscar time por nome")
        print("  5 - 🔍 Buscar liga por nome")
        print("  6 - 📈 Info detalhada de time (por ID)")
        print("  7 - 🏅 Classificação de liga (por ID)")
        print("  8 - 📅 Próximas partidas")
        print("  9 - ⏳ Histórico de partidas de um time")
        print(" 10 - 🤖 Gerar arquivo para predições em lote")
        print("  0 - ↩️  Voltar ao menu principal")
        
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
            term = input("Nome do time: ").strip()
            if term:
                explorer.search_team(term)
        
        elif choice == '5':
            term = input("Nome da liga: ").strip()
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
        
        elif choice == '0':
            break
        
        else:
            print("❌ Opção inválida")
    
    return explorer
# =============================================================================
# MODO 2: TREINAMENTO DO MODELO OTIMIZADO
# =============================================================================

def run_training():
    """Treina o modelo preditivo otimizado"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "MODO: TREINAMENTO DO MODELO OTIMIZADO" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        from soccer_predictor import SoccerMatchPredictor
    except ImportError:
        print("❌ Erro: soccer_predictor.py não encontrado")
        print("   Execute primeiro o código otimizado fornecido")
        return None
    
    # Configuração do treinamento
    print("\n⚙️  CONFIGURAÇÃO DO TREINAMENTO AVANÇADO")
    print("="*80)
    
    print("\nEscolha o nível de treinamento:")
    print("  1 - 🔍 Pequeno (5K partidas, rápido, para testes)")
    print("  2 - ⚖️  Médio (10K partidas, balanceado)")
    print("  3 - 📈 Grande (20K partidas, mais preciso)")
    print("  4 - 🚀 Completo (todos os dados, melhor performance)")
    print("  5 - ⚡ Rápido com ensemble (3K partidas, múltiplos modelos)")
    
    choice = input("\nEscolha (1-5): ").strip()
    
    configs = {
        '1': {'sample_size': 5000, 'use_optuna': False, 'n_trials': 10},
        '2': {'sample_size': 10000, 'use_optuna': True, 'n_trials': 20},
        '3': {'sample_size': 20000, 'use_optuna': True, 'n_trials': 30},
        '4': {'sample_size': None, 'use_optuna': True, 'n_trials': 50},
        '5': {'sample_size': 3000, 'use_optuna': False, 'n_trials': 5}
    }
    
    config = configs.get(choice, configs['2'])
    
    print(f"\n📊 Configuração selecionada:")
    print(f"   Amostra: {'Todas' if config['sample_size'] is None else f"{config['sample_size']:,}"} partidas")
    print(f"   Otimização: {'Sim' if config['use_optuna'] else 'Não'}")
    print(f"   Tentativas: {config['n_trials']}")
    
    # Nome do experimento
    experiment_name = input("\n📝 Nome do experimento (deixe em branco para auto): ").strip()
    if not experiment_name:
        import datetime
        experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Inicializar predictor
    predictor = SoccerMatchPredictor(
        base_path=BASE_PATH,
        experiment_name=experiment_name
    )
    
    try:
        # Carregar dados
        print("\n📂 Carregando dados...")
        predictor.load_data(cache=True)
        
        # Engenharia de features
        print("🔧 Criando features...")
        predictor.engineer_features(
            sample_size=config['sample_size'],
            min_games_history=5
        )
        
        # Treinar ensemble
        print("🤖 Treinando modelos...")
        predictor.train_ensemble(
            n_trials=config['n_trials'],
            use_optuna=config['use_optuna']
        )
        
        # Análise de features (COM VERIFICAÇÃO)
        print("📈 Analisando importância das features...")
        if hasattr(predictor, 'feature_importance_analysis'):
            predictor.feature_importance_analysis()
        else:
            # Análise simplificada
            print("⚠️  Método feature_importance_analysis não disponível")
            if hasattr(predictor, 'model') and hasattr(predictor.model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': predictor.feature_columns,
                    'importance': predictor.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n📊 Top 10 Features Mais Importantes (simplificado):")
                for idx, row in importances.head(10).iterrows():
                    print(f"   {row['feature']:30s} {row['importance']:.4f}")
        
        # Salvar modelo
        print("💾 Salvando modelo...")
        predictor.save_model()
        
        print("\n✅ Treinamento concluído com sucesso!")
        print(f"   Experimento: {experiment_name}")
        print(f"   Modelo salvo em: models/{experiment_name}.pkl")
        
        return predictor
    
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# MODO 3: PREDIÇÕES OTIMIZADAS
# =============================================================================

def run_prediction(predictor=None):
    """Faz predições de partidas com modelo otimizado"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "MODO: PREDIÇÃO OTIMIZADA" + " " * 32 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        from soccer_predictor import SoccerMatchPredictor
    except ImportError:
        print("❌ Erro: soccer_predictor.py não encontrado")
        return None
    
    # Carregar modelo existente ou treinar novo
    if predictor is None:
        import os
        import joblib
        
        # Verificar se há modelos salvos
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            if model_files:
                print(f"\n📦 {len(model_files)} modelo(s) encontrado(s):")
                for i, f in enumerate(model_files[:5], 1):
                    print(f"  {i}. {f}")
                
                if len(model_files) > 5:
                    print(f"  ... e mais {len(model_files) - 5} modelos")
                
                choice = input("\nCarregar qual modelo? (número ou 'n' para novo): ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(model_files):
                    model_file = os.path.join(models_dir, model_files[int(choice) - 1])
                    
                    print(f"📂 Carregando {model_file}...")
                    
                    predictor = SoccerMatchPredictor(BASE_PATH)
                    predictor.load_model(model_file)
                    
                    print(f"✅ Modelo carregado: {predictor.experiment_name}")
                else:
                    print("\n🆕 Treinando novo modelo...")
                    predictor = run_training()
            else:
                print("\n📭 Nenhum modelo salvo encontrado. Treinando novo...")
                predictor = run_training()
        else:
            print("\n📭 Diretório de modelos não existe. Treinando novo...")
            predictor = run_training()
    
    if predictor is None:
        print("❌ Não foi possível inicializar o predictor")
        return None
    
    # Menu de predição
    while True:
        print("\n" + "="*80)
        print("🔮 MENU DE PREDIÇÃO")
        print("="*80)
        print("\nOpções:")
        print("  1 - 🔍 Predição individual")
        print("  2 - 📊 Predição em lote (arquivo gerado)")
        print("  3 - 📅 Próximas partidas de uma liga")
        print("  4 - ⚽ Próximas partidas de um time")
        print("  5 - 💰 Simulação de betting")
        print("  0 - ↩️  Voltar ao menu principal")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == '1':
            run_individual_prediction(predictor)
        
        elif choice == '2':
            run_batch_prediction(predictor)
        
        elif choice == '3':
            run_league_predictions(predictor)
        
        elif choice == '4':
            run_team_predictions(predictor)
        
        elif choice == '5':
            run_betting_simulation(predictor)
        
        elif choice == '0':
            break
        
        else:
            print("❌ Opção inválida")
    
    return predictor

def run_individual_prediction(predictor):
    """Predição individual de partida"""
    print("\n" + "="*80)
    print("🔍 PREDIÇÃO INDIVIDUAL")
    print("="*80)
    
    try:
        print("\nInforme os dados da partida:")
        home_id = int(input("ID do time mandante: ").strip())
        away_id = int(input("ID do time visitante: ").strip())
        league_id = int(input("ID da liga: ").strip())
        season_type = input("Tipo de temporada (default 'Regular Season'): ").strip()
        season_type = season_type if season_type else 'Regular Season'
        
        # Data da partida (opcional)
        match_date = input("Data da partida (YYYY-MM-DD, deixe em branco para hoje): ").strip()
        if match_date:
            match_date = pd.Timestamp(match_date)
        else:
            match_date = None
        
        print(f"\n🎯 Fazendo predição...")
        result = predictor.predict_match(
            home_team_id=home_id,
            away_team_id=away_id,
            league_id=league_id,
            season_type=season_type,
            match_date=match_date
        )
        
        if result:
            print(f"\n✅ Predição concluída!")
            
            # Salvar resultado
            save = input("\n💾 Salvar resultado? (s/n): ").strip().lower()
            if save == 's':
                import json
                from datetime import datetime
                
                result_file = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✅ Resultado salvo em: {result_file}")
        
    except ValueError:
        print("❌ Erro: IDs devem ser números")
    except Exception as e:
        print(f"❌ Erro durante predição: {e}")

def run_batch_prediction(predictor):
    """Predição em lote a partir de arquivo gerado"""
    print("\n" + "="*80)
    print("📊 PREDIÇÃO EM LOTE")
    print("="*80)
    
    import json
    import pandas as pd
    
    # Verificar se existe arquivo de predições
    metadata_path = os.path.join(BASE_PATH, 'metadata')
    config_file = os.path.join(metadata_path, 'prediction_batch.json')
    
    if not os.path.exists(config_file):
        print("❌ Arquivo de predições em lote não encontrado")
        print("   Use a opção 10 no menu de exploração para gerar o arquivo")
        return
    
    print(f"📂 Carregando {config_file}...")
    
    with open(config_file, 'r') as f:
        predictions_config = json.load(f)
    
    print(f"✅ {len(predictions_config)} partidas para predição")
    
    results = []
    
    for i, config in enumerate(predictions_config):
        print(f"\n[{i+1}/{len(predictions_config)}] {config['homeTeam']} vs {config['awayTeam']}")
        
        try:
            result = predictor.predict_match(
                home_team_id=config['homeTeamId'],
                away_team_id=config['awayTeamId'],
                league_id=config['leagueId'],
                season_type=config['seasonType'],
                match_date=pd.Timestamp(config['date'])
            )
            
            if result:
                results.append({
                    'fixtureId': config['fixtureId'],
                    'date': config['date'],
                    'homeTeam': config['homeTeam'],
                    'awayTeam': config['awayTeam'],
                    'league': config['league'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'prob_home': result['probabilities']['home_win'],
                    'prob_draw': result['probabilities']['draw'],
                    'prob_away': result['probabilities']['away_win']
                })
                print(f"   🎯 {result['prediction']} ({result['confidence']:.1%})")
        
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            continue
    
    if results:
        # Salvar resultados
        import datetime
        
        df_results = pd.DataFrame(results)
        results_file = f"batch_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_file, index=False)
        
        print(f"\n✅ Predições salvas em: {results_file}")
        
        # Estatísticas
        print(f"\n📈 ESTATÍSTICAS DAS PREDIÇÕES:")
        print(f"   Total de predições: {len(results)}")
        print(f"   Média de confiança: {df_results['confidence'].mean():.1%}")
        
        # Distribuição
        pred_counts = df_results['prediction'].value_counts()
        for pred, count in pred_counts.items():
            print(f"   {pred}: {count} ({count/len(results)*100:.1f}%)")
    else:
        print("\n❌ Nenhuma predição bem-sucedida")

def run_league_predictions(predictor):
    """Predição para próximas partidas de uma liga"""
    print("\n" + "="*80)
    print("🏆 PREDIÇÕES PARA UMA LIGA")
    print("="*80)
    
    try:
        from data_explorer import SoccerDataExplorer
        
        # Carregar dados
        explorer = SoccerDataExplorer(BASE_PATH)
        explorer.load_all_data(cache=True)
        
        # Obter ID da liga
        league_id = int(input("ID da liga: ").strip())
        
        # Verificar se a liga existe
        league = explorer.leagues[explorer.leagues['leagueId'] == league_id]
        if len(league) == 0:
            print(f"❌ Liga ID {league_id} não encontrada")
            return
        
        league_name = league.iloc[0]['leagueName']
        print(f"\n🏆 Liga: {league_name}")
        
        # Obter próximas partidas
        days = input("Dias à frente (default 7): ").strip()
        days = int(days) if days else 7
        
        upcoming = explorer.find_upcoming_matches(
            league_id=league_id,
            days_ahead=days,
            limit=20
        )
        
        if upcoming is None or len(upcoming) == 0:
            print(f"❌ Nenhuma partida encontrada para os próximos {days} dias")
            return
        
        # Fazer predições
        print(f"\n🎯 Fazendo predições para {len(upcoming)} partidas...")
        results = []
        
        for _, match in upcoming.iterrows():
            try:
                result = predictor.predict_match(
                    home_team_id=int(match['homeTeamId']),
                    away_team_id=int(match['awayTeamId']),
                    league_id=int(match['leagueId']),
                    season_type=int(match.get('seasonType', 2)),
                    match_date=match['date']
                )
                
                if result:
                    results.append({
                        'date': match['date'].strftime('%m-%d'),
                        'home': match['home_name'],
                        'away': match['away_name'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'prob_home': result['probabilities']['home_win'],
                        'prob_draw': result['probabilities']['draw'],
                        'prob_away': result['probabilities']['away_win']
                    })
            
            except Exception as e:
                print(f"   ❌ Erro na partida {match['home_name']} vs {match['away_name']}: {e}")
                continue
        
        if results:
            # Mostrar resultados
            print(f"\n{'Data':<8} {'Mandante':<25} {'Visitante':<25} {'Predição':<15} {'Confiança':<10}")
            print("-"*90)
            
            for r in results:
                print(f"{r['date']:<8} "
                      f"{r['home'][:25]:<25} "
                      f"{r['away'][:25]:<25} "
                      f"{r['prediction']:<15} "
                      f"{r['confidence']:<10.1%}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def run_team_predictions(predictor):
    """Predição para próximas partidas de um time"""
    print("\n" + "="*80)
    print("⚽ PREDIÇÕES PARA UM TIME")
    print("="*80)
    
    try:
        from data_explorer import SoccerDataExplorer
        
        # Carregar dados
        explorer = SoccerDataExplorer(BASE_PATH)
        explorer.load_all_data(cache=True)
        
        # Obter ID do time
        team_id = int(input("ID do time: ").strip())
        
        # Verificar se o time existe
        team = explorer.teams[explorer.teams['teamId'] == team_id]
        if len(team) == 0:
            print(f"❌ Time ID {team_id} não encontrado")
            return
        
        team_name = team.iloc[0]['name']
        print(f"\n⚽ Time: {team_name}")
        
        # Obter próximas partidas do time
        team_fixtures = explorer.get_team_fixtures(team_id, limit=10)
        
        if team_fixtures is None or len(team_fixtures['future']) == 0:
            print("❌ Nenhuma partida futura encontrada")
            return
        
        future_matches = team_fixtures['future']
        print(f"\n🎯 Fazendo predições para {len(future_matches)} partidas futuras...")
        
        for _, match in future_matches.iterrows():
            is_home = match['homeTeamId'] == team_id
            opponent_id = match['awayTeamId'] if is_home else match['homeTeamId']
            opponent_name = explorer.team_dict.get(opponent_id, f"Time {opponent_id}")
            
            venue = "(Casa)" if is_home else "(Fora)"
            date_str = match['date'].strftime('%Y-%m-%d')
            
            print(f"\n📅 {date_str} {venue} vs {opponent_name}")
            
            try:
                result = predictor.predict_match(
                    home_team_id=int(match['homeTeamId']),
                    away_team_id=int(match['awayTeamId']),
                    league_id=int(match['leagueId']),
                    season_type=int(match.get('seasonType', 2)),
                    match_date=match['date']
                )
                
                if result:
                    print(f"   🎯 Predição: {result['prediction']}")
                    print(f"   🔒 Confiança: {result['confidence']:.1%}")
                    print(f"   📊 Probabilidades: "
                          f"Casa {result['probabilities']['home_win']:.1%} | "
                          f"Empate {result['probabilities']['draw']:.1%} | "
                          f"Fora {result['probabilities']['away_win']:.1%}")
            
            except Exception as e:
                print(f"   ❌ Erro na predição: {e}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def run_betting_simulation(predictor):
    """Simulação de estratégias de betting"""
    print("\n" + "="*80)
    print("💰 SIMULAÇÃO DE BETTING")
    print("="*80)
    
    print("\n⚠️  AVISO: Esta é uma simulação educacional.")
    print("   Não use para betting real sem verificação adicional.")
    
    try:
        # Carregar dados históricos para backtesting
        from data_explorer import SoccerDataExplorer
        explorer = SoccerDataExplorer(BASE_PATH)
        explorer.load_all_data(cache=True)
        
        # Filtrar partidas com resultado
        completed = explorer.fixtures[explorer.fixtures['result'].notna()]
        
        print(f"\n📊 Base de dados para simulação:")
        print(f"   Total de partidas: {len(completed):,}")
        
        # Escolher amostra para simulação
        sample_size = input("\nTamanho da amostra para simulação (default 1000): ").strip()
        sample_size = int(sample_size) if sample_size else 1000
        
        if sample_size > len(completed):
            sample_size = len(completed)
            print(f"⚠️  Limitado a {sample_size:,} partidas disponíveis")
        
        # Amostrar partidas
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(completed)), min(sample_size, len(completed)))
        sample_matches = completed.iloc[sample_indices]
        
        print(f"\n🎯 Simulando {len(sample_matches)} partidas...")
        
        # Configurar estratégia
        print("\n⚙️  Configurar estratégia de betting:")
        print("  1 - Aposta em todas as partidas")
        print("  2 - Aposta apenas com confiança > 50%")
        print("  3 - Aposta apenas com confiança > 60%")
        print("  4 - Value betting (probabilidade > odds implícitas)")
        
        strategy_choice = input("Escolha (1-4): ").strip()
        
        bankroll = 1000  # Bankroll inicial
        bet_amount = 10  # Valor fixo por aposta
        results = []
        
        for i, match in sample_matches.iterrows():
            if i % 100 == 0:
                print(f"   Progresso: {i}/{len(sample_matches)} partidas")
            
            try:
                # Fazer predição
                prediction = predictor.predict_match(
                    home_team_id=int(match['homeTeamId']),
                    away_team_id=int(match['awayTeamId']),
                    league_id=int(match['leagueId']),
                    season_type=int(match.get('seasonType', 2)),
                    match_date=match['date']
                )
                
                if prediction:
                    # Resultado real
                    actual_result = match['result']  # 'H', 'D', or 'A'
                    actual_result_name = {
                        'H': 'Home Win',
                        'D': 'Draw',
                        'A': 'Away Win'
                    }.get(actual_result, 'Unknown')
                    
                    # Decidir se aposta com base na estratégia
                    should_bet = False
                    
                    if strategy_choice == '1':
                        should_bet = True
                    elif strategy_choice == '2':
                        should_bet = prediction['confidence'] > 0.50
                    elif strategy_choice == '3':
                        should_bet = prediction['confidence'] > 0.60
                    elif strategy_choice == '4':
                        # Value betting: apostar se nossa probabilidade for maior que implied probability
                        # Assumindo odds fixas para simplificação
                        odds = {'Home Win': 2.1, 'Draw': 3.2, 'Away Win': 3.5}
                        implied_prob = 1 / odds[prediction['prediction']]
                        should_bet = prediction['confidence'] > implied_prob * 1.1  # 10% edge
                    else:
                        should_bet = True
                    
                    if should_bet and bankroll >= bet_amount:
                        # Fazer aposta
                        bankroll -= bet_amount
                        
                        # Verificar se ganhou
                        predicted_codes = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
                        predicted_code = predicted_codes.get(prediction['prediction'], '')
                        
                        if predicted_code == actual_result:
                            # Ganhou!
                            win_amount = bet_amount * 2  # Odds simplificadas de 2.0
                            bankroll += win_amount
                            results.append({'result': 'win', 'amount': bet_amount})
                        else:
                            # Perdeu
                            results.append({'result': 'loss', 'amount': bet_amount})
            
            except Exception as e:
                continue
        
        # Calcular resultados
        if results:
            wins = sum(1 for r in results if r['result'] == 'win')
            losses = len(results) - wins
            total_bet = sum(r['amount'] for r in results)
            total_won = wins * bet_amount * 2
            profit = total_won - total_bet
            roi = (profit / total_bet * 100) if total_bet > 0 else 0
            
            print(f"\n📈 RESULTADOS DA SIMULAÇÃO:")
            print(f"   Total de apostas: {len(results)}")
            print(f"   Vitórias: {wins} ({wins/len(results)*100:.1f}%)")
            print(f"   Derrotas: {losses} ({losses/len(results)*100:.1f}%)")
            print(f"   Valor total apostado: ${total_bet:.2f}")
            print(f"   Lucro/Prejuízo: ${profit:+.2f}")
            print(f"   ROI: {roi:+.1f}%")
            print(f"   Bankroll final: ${bankroll:.2f}")
            
            if profit > 0:
                print(f"\n✅ Estratégia lucrativa!")
            else:
                print(f"\n❌ Estratégia não lucrativa nesta simulação")
        
        else:
            print("\n❌ Nenhuma aposta realizada")
    
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")

# =============================================================================
# MODO 4: MENU COMPLETO OTIMIZADO
# =============================================================================

def run_menu():
    """Menu principal interativo otimizado"""
    
    predictor = None
    
    while True:
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 10 + "⚽ SISTEMA DE PREDIÇÃO DE FUTEBOL - VERSÃO OTIMIZADA ⚽" + " " * 10 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        print("\n📋 MENU PRINCIPAL")
        print("="*80)
        print("\n  1 - 🔍 Explorar dados (times, ligas, classificações)")
        print("  2 - 🎓 Treinar modelo preditivo")
        print("  3 - 🔮 Fazer predições")
        print("  4 - 📊 Pipeline completo (treinar + prever + analisar)")
        print("  5 - 💰 Simulação de betting")
        print("  6 - 📈 Analisar performance do modelo")
        print("  0 - ❌ Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == '1':
            run_exploration()
        
        elif choice == '2':
            predictor = run_training()
        
        elif choice == '3':
            predictor = run_prediction(predictor)
        
        elif choice == '4':
            print("\n🚀 EXECUTANDO PIPELINE COMPLETO")
            print("="*80)
            
            # 1. Explorar dados
            print("\n1️⃣  FASE 1: EXPLORAÇÃO DE DADOS")
            explorer = run_exploration()
            
            # 2. Treinar modelo
            print("\n\n2️⃣  FASE 2: TREINAMENTO DO MODELO")
            predictor = run_training()
            
            if predictor:
                # 3. Fazer predições
                print("\n\n3️⃣  FASE 3: PREDIÇÕES")
                input("Pressione ENTER para fazer predições...")
                predictor = run_prediction(predictor)
        
        elif choice == '5':
            if predictor is None:
                print("\n⚠️  Modelo não carregado. Treinando um primeiro...")
                predictor = run_training()
            
            if predictor:
                run_betting_simulation(predictor)
        
        elif choice == '6':
            if predictor is None:
                print("\n⚠️  Modelo não carregado. Carregue ou treine um primeiro.")
            else:
                print("\n📈 ANÁLISE DE PERFORMANCE")
                print("="*80)
                
                # Aqui você pode adicionar análise de performance
                print("Funcionalidade em desenvolvimento...")
                print("Considere implementar:")
                print("  - Backtesting detalhado")
                print("  - Curvas de aprendizado")
                print("  - Matriz de confusão por temporada")
                print("  - Análise de features por importância")
        
        elif choice == '0':
            print("\n👋 Até logo!")
            break
        
        else:
            print("❌ Opção inválida")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Sistema de Predição de Futebol Otimizado')
    parser.add_argument('--explore', action='store_true', help='Modo exploração de dados')
    parser.add_argument('--train', action='store_true', help='Modo treinamento')
    parser.add_argument('--predict', action='store_true', help='Modo predição')
    parser.add_argument('--menu', action='store_true', help='Menu interativo completo')
    parser.add_argument('--batch', type=str, help='Arquivo para predição em lote')
    
    args = parser.parse_args()
    
    # Adicionar o diretório atual ao path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Se nenhum argumento foi passado, mostrar menu
    if not any([args.explore, args.train, args.predict, args.menu, args.batch]):
        run_menu()
    else:
        if args.explore:
            run_exploration()
        if args.train:
            run_training()
        if args.predict:
            run_prediction()
        if args.menu:
            run_menu()
        if args.batch:
            print("⚠️  Funcionalidade de batch em desenvolvimento")