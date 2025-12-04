"""
Script principal para treinar e fazer predições de resultados de futebol
"""
import sys
import os
import argparse
import time

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.etl import load_and_preprocess_data
from src.feature_engineering import create_features
from src.model_xgboost import train_and_evaluate, SoccerPredictor
from src.predict import MatchPredictor, predict_from_team_ids
from src.utils import logger, format_time, log_separator

def train_pipeline():
    """Pipeline completo de treinamento"""
    log_separator("PIPELINE DE TREINAMENTO", char='=', width=80)
    start_time = time.time()
    
    try:
        # 1. ETL
        logger.info("\n🔄 ETAPA 1: Carregando e processando dados...")
        data = load_and_preprocess_data()
        logger.info(f"✓ Dados carregados: {len(data['fixtures']):,} partidas")
        
        # 2. Feature Engineering
        logger.info("\n🔄 ETAPA 2: Criando features...")
        master_df = create_features(data)
        logger.info(f"✓ Features criadas: {master_df.shape[1]} colunas, {master_df.shape[0]:,} linhas")
        
        # 3. Treinamento
        logger.info("\n🔄 ETAPA 3: Treinando modelo...")
        predictor, metrics = train_and_evaluate(master_df)
        logger.info(f"✓ Modelo treinado com acurácia: {metrics['accuracy']:.4f}")
        
        # Tempo total
        total_time = time.time() - start_time
        
        log_separator("TREINAMENTO CONCLUÍDO", char='=', width=80)
        logger.info(f"\n✓ Pipeline concluído com sucesso!")
        logger.info(f"⏱️  Tempo total: {format_time(total_time)}")
        logger.info(f"📊 Acurácia final: {metrics['accuracy']:.2%}")
        logger.info(f"📁 Modelo salvo em: models/")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Erro no pipeline de treinamento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def predict_pipeline(home_team_id=None, away_team_id=None):
    """Pipeline de predição"""
    log_separator("PIPELINE DE PREDIÇÃO", char='=', width=80)
    
    try:
        # Criar predictor
        predictor = MatchPredictor()
        
        # Carregar modelo
        logger.info("\n📂 Carregando modelo treinado...")
        if not predictor.load_model():
            logger.error("❌ Falha ao carregar modelo. Execute o treinamento primeiro.")
            return False
        
        # Se IDs fornecidos, fazer predição
        if home_team_id and away_team_id:
            logger.info(f"\n🔮 Predizendo: Time {home_team_id} (casa) vs Time {away_team_id} (visitante)")
            
            # Carregar dados para criar features
            logger.info("📊 Carregando dados para feature engineering...")
            data = load_and_preprocess_data()
            
            # Fazer predição
            result = predict_from_team_ids(home_team_id, away_team_id, data)
            
            if result:
                log_separator("RESULTADO DA PREDIÇÃO", char='=', width=80)
                logger.info(f"\n🎯 Predição: {result['prediction_label']}")
                logger.info(f"📊 Confiança: {result['confidence']:.2%}")
                logger.info(f"\n📈 Probabilidades:")
                logger.info(f"   Empate: {result['probabilities']['empate']:.2%}")
                logger.info(f"   Vitória Casa: {result['probabilities']['vitoria_casa']:.2%}")
                logger.info(f"   Vitória Visitante: {result['probabilities']['vitoria_visitante']:.2%}")
                log_separator(char='=', width=80)
            else:
                logger.error("❌ Erro ao fazer predição")
                return False
        else:
            logger.info("\n✓ Modelo carregado e pronto para predições!")
            logger.info("\n💡 Para fazer uma predição, use:")
            logger.info("   python main.py --mode predict --home_team <ID> --away_team <ID>")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Erro no pipeline de predição: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def evaluate_pipeline():
    """Pipeline de avaliação do modelo"""
    log_separator("PIPELINE DE AVALIAÇÃO", char='=', width=80)
    
    try:
        # Carregar modelo
        logger.info("\n📂 Carregando modelo...")
        predictor = SoccerPredictor()
        
        if not predictor.load_model():
            logger.error("❌ Modelo não encontrado. Execute o treinamento primeiro.")
            return False
        
        logger.info("✓ Modelo carregado com sucesso!")
        
        # Carregar dados de teste
        logger.info("\n📊 Carregando dados de teste...")
        data = load_and_preprocess_data()
        master_df = create_features(data)
        
        # Preparar dados
        predictor.prepare_data(master_df)
        
        # Avaliar
        logger.info("\n🔍 Avaliando modelo...")
        metrics = predictor.evaluate()
        
        log_separator("AVALIAÇÃO CONCLUÍDA", char='=', width=80)
        logger.info(f"\n✓ Avaliação concluída!")
        logger.info(f"📊 Acurácia: {metrics['accuracy']:.2%}")
        logger.info(f"📊 F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Erro na avaliação: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def interactive_mode():
    """Modo interativo para predições"""
    log_separator("MODO INTERATIVO", char='=', width=80)
    
    # Carregar modelo
    predictor = MatchPredictor()
    logger.info("\n📂 Carregando modelo...")
    
    if not predictor.load_model():
        logger.error("❌ Modelo não encontrado. Execute o treinamento primeiro.")
        return False
    
    logger.info("✓ Modelo carregado com sucesso!")
    
    # Carregar dados
    logger.info("\n📊 Carregando dados...")
    data = load_and_preprocess_data()
    
    # Mostrar times disponíveis
    teams = data['teams'][['teamId', 'name']].sort_values('name')
    logger.info(f"\n📋 {len(teams)} times disponíveis")
    logger.info("\nExemplos de times:")
    logger.info(teams.head(10).to_string(index=False))
    
    # Loop interativo
    log_separator("COMEÇAR PREDIÇÕES", char='-', width=80)
    logger.info("\n💡 Digite 'sair' para encerrar\n")
    
    while True:
        try:
            # Solicitar IDs
            home_input = input("\n🏠 ID do time da casa (ou 'sair'): ").strip()
            if home_input.lower() == 'sair':
                break
            
            away_input = input("✈️  ID do time visitante: ").strip()
            
            # Converter para int
            home_team_id = int(home_input)
            away_team_id = int(away_input)
            
            # Verificar se times existem
            home_name = teams[teams['teamId'] == home_team_id]['name'].values
            away_name = teams[teams['teamId'] == away_team_id]['name'].values
            
            if len(home_name) == 0:
                logger.error(f"❌ Time {home_team_id} não encontrado")
                continue
            
            if len(away_name) == 0:
                logger.error(f"❌ Time {away_team_id} não encontrado")
                continue
            
            home_name = home_name[0]
            away_name = away_name[0]
            
            # Fazer predição
            logger.info(f"\n🔮 Predizendo: {home_name} vs {away_name}")
            result = predict_from_team_ids(home_team_id, away_team_id, data)
            
            if result:
                log_separator("RESULTADO", char='-', width=80)
                logger.info(f"\n🎯 Predição: {result['prediction_label']}")
                logger.info(f"📊 Confiança: {result['confidence']:.2%}")
                logger.info(f"\n📈 Probabilidades:")
                logger.info(f"   Empate: {result['probabilities']['empate']:.2%}")
                logger.info(f"   Vitória {home_name}: {result['probabilities']['vitoria_casa']:.2%}")
                logger.info(f"   Vitória {away_name}: {result['probabilities']['vitoria_visitante']:.2%}")
            
        except ValueError:
            logger.error("❌ IDs devem ser números inteiros")
        except KeyboardInterrupt:
            logger.info("\n\n👋 Encerrando...")
            break
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
    
    log_separator("SESSÃO ENCERRADA", char='=', width=80)
    return True

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Predição de Resultados de Futebol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Treinar modelo
  python main.py --mode train
  
  # Fazer predição
  python main.py --mode predict --home_team 123 --away_team 456
  
  # Avaliar modelo
  python main.py --mode evaluate
  
  # Modo interativo
  python main.py --mode interactive
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'evaluate', 'interactive'],
        required=True,
        help='Modo de operação'
    )
    
    parser.add_argument(
        '--home_team',
        type=int,
        help='ID do time da casa (para modo predict)'
    )
    
    parser.add_argument(
        '--away_team',
        type=int,
        help='ID do time visitante (para modo predict)'
    )
    
    args = parser.parse_args()
    
    # Executar modo selecionado
    if args.mode == 'train':
        success = train_pipeline()
    elif args.mode == 'predict':
        success = predict_pipeline(args.home_team, args.away_team)
    elif args.mode == 'evaluate':
        success = evaluate_pipeline()
    elif args.mode == 'interactive':
        success = interactive_mode()
    else:
        logger.error(f"Modo inválido: {args.mode}")
        success = False
    
    # Retornar código de saída
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()