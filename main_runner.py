"""
SCRIPT PRINCIPAL - Sistema de Predição de Futebol
Execute este arquivo para treinar o modelo e fazer predições

Uso:
1. Primeiro execute: python main_runner.py --explore
   (para conhecer times e ligas disponíveis)

2. Depois execute: python main_runner.py --train
   (para treinar o modelo)

3. Por fim: python main_runner.py --predict
   (para fazer predições)
"""

from pathlib import Path
import sys
import os

# Adicione o caminho do seu projeto se necessário
# sys.path.append('caminho/para/seus/scripts')

# Importar as classes (assumindo que estão no mesmo diretório)
# Se estiver em arquivos separados, ajuste os imports

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_PATH = Path(__file__).parent / "data"

# =============================================================================
# MODO 1: EXPLORAÇÃO DE DADOS
# =============================================================================

def run_exploration():
    """Explora o dataset para conhecer times e ligas"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "MODO: EXPLORAÇÃO DE DADOS" + " " * 33 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    from data_explorer import SoccerDataExplorer
    
    explorer = SoccerDataExplorer(BASE_PATH)
    explorer.load_all_data()
    explorer.show_dataset_overview()
    explorer.show_top_leagues(15)
    explorer.show_top_teams(20)
    
    # Busca interativa
    while True:
        print("\n" + "="*80)
        print("🔍 BUSCA INTERATIVA")
        print("="*80)
        print("\nOpções:")
        print("  1 - Buscar time")
        print("  2 - Buscar liga")
        print("  3 - Info detalhada de time (por ID)")
        print("  4 - Classificação de liga (por ID)")
        print("  5 - Próximas partidas")
        print("  0 - Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == '1':
            term = input("Nome do time: ").strip()
            explorer.search_team(term)
        
        elif choice == '2':
            term = input("Nome da liga: ").strip()
            explorer.search_league(term)
        
        elif choice == '3':
            team_id = input("ID do time: ").strip()
            try:
                explorer.get_team_info(int(team_id))
            except ValueError:
                print("❌ ID inválido")
        
        elif choice == '4':
            league_id = input("ID da liga: ").strip()
            try:
                explorer.get_league_standings(int(league_id))
            except ValueError:
                print("❌ ID inválido")
        
        elif choice == '5':
            print("\nFiltros (deixe em branco para pular):")
            team_id = input("ID do time (opcional): ").strip()
            league_id = input("ID da liga (opcional): ").strip()
            limit = input("Quantidade (default 10): ").strip()
            
            team_id = int(team_id) if team_id else None
            league_id = int(league_id) if league_id else None
            limit = int(limit) if limit else 10
            
            explorer.find_upcoming_matches(team_id, league_id, limit)
        
        elif choice == '0':
            break
        
        else:
            print("❌ Opção inválida")

# =============================================================================
# MODO 2: TREINAMENTO DO MODELO
# =============================================================================

def run_training():
    """Treina o modelo preditivo"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "MODO: TREINAMENTO DO MODELO" + " " * 31 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    from soccer_predictor import SoccerMatchPredictor
    
    # Perguntar sobre tamanho da amostra
    print("\n⚙️  CONFIGURAÇÃO DO TREINAMENTO")
    print("="*80)
    print("\nVocê pode treinar com:")
    print("  1 - Amostra pequena (~5,000 partidas) - RÁPIDO para testes")
    print("  2 - Amostra média (~10,000 partidas) - BALANCEADO")
    print("  3 - Amostra grande (~20,000 partidas) - MAIS COMPLETO")
    print("  4 - TODOS os dados - MÁXIMA PERFORMANCE (pode demorar)")
    
    choice = input("\nEscolha (1-4): ").strip()
    
    sample_sizes = {
        '1': 5000,
        '2': 10000,
        '3': 20000,
        '4': None
    }
    
    sample_size = sample_sizes.get(choice, 10000)
    
    # Inicializar e treinar
    predictor = SoccerMatchPredictor(BASE_PATH)
    predictor.load_data()
    predictor.engineer_features(sample_size=sample_size)
    results = predictor.train_model(test_size=0.2)
    
    # Salvar modelo (opcional)
    print("\n" + "="*80)
    print("💾 SALVAR MODELO?")
    print("="*80)
    save = input("Salvar modelo treinado? (s/n): ").strip().lower()
    
    if save == 's':
        import pickle
        
        model_data = {
            'model': predictor.model,
            'scaler': predictor.scaler,
            'feature_columns': predictor.feature_columns,
            'results': results
        }
        
        filename = 'soccer_predictor_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modelo salvo em: {filename}")
    
    return predictor

# =============================================================================
# MODO 3: FAZER PREDIÇÕES
# =============================================================================

def run_prediction():
    """Faz predições de partidas"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "MODO: PREDIÇÃO" + " " * 39 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    from soccer_predictor import SoccerMatchPredictor
    
    # Verificar se existe modelo salvo
    import pickle
    import os
    
    if os.path.exists('soccer_predictor_model.pkl'):
        print("\n📦 Modelo salvo encontrado!")
        load = input("Carregar modelo existente? (s/n): ").strip().lower()
        
        if load == 's':
            with open('soccer_predictor_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            predictor = SoccerMatchPredictor(BASE_PATH)
            predictor.load_data()
            predictor.model = model_data['model']
            predictor.scaler = model_data['scaler']
            predictor.feature_columns = model_data['feature_columns']
            
            print("✅ Modelo carregado com sucesso!")
        else:
            print("\n⚠️  Treinando novo modelo...")
            predictor = run_training()
    else:
        print("\n⚠️  Nenhum modelo salvo encontrado. Treinando...")
        predictor = run_training()
    
    # Loop de predições
    while True:
        print("\n" + "="*80)
        print("🔮 FAZER PREDIÇÃO")
        print("="*80)
        print("\nPara fazer uma predição, você precisa dos IDs:")
        print("  - ID do time mandante (home)")
        print("  - ID do time visitante (away)")
        print("  - ID da liga")
        print("  - Tipo de temporada (geralmente 2)")
        print("\n💡 Use o modo exploração para encontrar estes IDs")
        
        try:
            home_id = int(input("\nID do time mandante: ").strip())
            away_id = int(input("ID do time visitante: ").strip())
            league_id = int(input("ID da liga: ").strip())
            season_type = input("Tipo de temporada (default 2): ").strip()
            season_type = int(season_type) if season_type else 2
            
            # Fazer predição
            result = predictor.predict_match(
                home_team_id=home_id,
                away_team_id=away_id,
                league_id=league_id,
                season_type=season_type
            )
            
            # Perguntar se quer fazer outra
            another = input("\nFazer outra predição? (s/n): ").strip().lower()
            if another != 's':
                break
        
        except ValueError:
            print("❌ Erro: IDs devem ser números")
        except KeyboardInterrupt:
            print("\n\n👋 Encerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

# =============================================================================
# MODO 4: MENU COMPLETO
# =============================================================================

def run_menu():
    """Menu principal interativo"""
    
    while True:
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 15 + "SISTEMA DE PREDIÇÃO DE RESULTADOS DE FUTEBOL" + " " * 19 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        print("\n📋 MENU PRINCIPAL")
        print("="*80)
        print("\n  1 - 🔍 Explorar dados (times, ligas, classificações)")
        print("  2 - 🎓 Treinar modelo")
        print("  3 - 🔮 Fazer predições")
        print("  4 - 📊 Executar pipeline completo (treinar + prever)")
        print("  0 - ❌ Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == '1':
            run_exploration()
        
        elif choice == '2':
            run_training()
        
        elif choice == '3':
            run_prediction()
        
        elif choice == '4':
            print("\n🚀 EXECUTANDO PIPELINE COMPLETO")
            predictor = run_training()
            input("\nPressione ENTER para fazer predições...")
            
            # Usar o predictor já treinado
            from soccer_predictor import SoccerMatchPredictor
            
            while True:
                try:
                    print("\n" + "="*80)
                    print("🔮 FAZER PREDIÇÃO")
                    print("="*80)
                    
                    home_id = int(input("\nID do time mandante: ").strip())
                    away_id = int(input("ID do time visitante: ").strip())
                    league_id = int(input("ID da liga: ").strip())
                    season_type = input("Tipo de temporada (default 2): ").strip()
                    season_type = int(season_type) if season_type else 2
                    
                    predictor.predict_match(home_id, away_id, league_id, season_type)
                    
                    another = input("\nFazer outra predição? (s/n): ").strip().lower()
                    if another != 's':
                        break
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Erro: {e}")
        
        elif choice == '0':
            print("\n👋 Até logo!")
            break
        
        else:
            print("❌ Opção inválida")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Predição de Futebol')
    parser.add_argument('--explore', action='store_true', help='Modo exploração')
    parser.add_argument('--train', action='store_true', help='Modo treinamento')
    parser.add_argument('--predict', action='store_true', help='Modo predição')
    parser.add_argument('--menu', action='store_true', help='Menu interativo')
    
    args = parser.parse_args()
    
    # Se nenhum argumento foi passado, mostrar menu
    if not any(vars(args).values()):
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