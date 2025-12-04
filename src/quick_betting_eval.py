"""
SOLUÇÃO IMEDIATA: Avaliação de estratégias com dados sintéticos
Funciona mesmo com arquivos corrompidos
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from src.utils import logger, log_separator
from src.config import MODEL_FILES

def quick_evaluate_betting_strategies():
    """
    Avalia estratégias de apostas usando o modelo treinado
    Carrega modelo diretamente, ignorando arquivos corrompidos
    """
    log_separator("AVALIAÇÃO RÁPIDA DE ESTRATÉGIAS", char='=', width=80)
    
    try:
        # 1. Carregar modelo XGBoost diretamente
        logger.info("\n📂 Carregando modelo...")
        
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILES['model'])
        logger.info("✓ Modelo XGBoost carregado!")
        
        # 2. Carregar feature columns
        from src.utils import load_json
        feature_columns = load_json(MODEL_FILES['features'])
        logger.info(f"✓ Features carregadas: {len(feature_columns)} colunas")
        
        # 3. Tentar carregar scaler (se falhar, criar um novo)
        try:
            with open(MODEL_FILES['scaler'], 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✓ Scaler carregado!")
        except Exception as e:
            logger.warning(f"⚠️  Erro ao carregar scaler: {e}")
            logger.info("📊 Criando scaler sintético...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Fit com dados sintéticos
            dummy_data = np.random.randn(100, len(feature_columns))
            scaler.fit(dummy_data)
            logger.info("✓ Scaler sintético criado!")
        
        # 4. Gerar dados de teste
        logger.info("\n📊 Gerando dados de teste...")
        n_samples = 5000
        n_features = len(feature_columns)
        
        np.random.seed(42)
        X_test = pd.DataFrame(
            np.random.randn(n_samples, n_features) * 0.5,
            columns=feature_columns
        )
        
        y_test = np.random.choice(
            [0, 1, 2], 
            size=n_samples,
            p=[0.24, 0.46, 0.30]
        )
        
        logger.info(f"✓ Dataset: {n_samples:,} amostras")
        logger.info(f"  Empates: {(y_test==0).sum()} | Casa: {(y_test==1).sum()} | Visitante: {(y_test==2).sum()}")
        
        # 5. Normalizar
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
        
        # 6. Avaliar estratégias
        logger.info("\n🎯 Avaliando estratégias de apostas...")
        
        # Criar predictor mock
        class MockPredictor:
            def __init__(self, model):
                self.model = model
        
        predictor = MockPredictor(model)
        
        from src.betting_strategies import BettingStrategiesPredictor
        
        betting_pred = BettingStrategiesPredictor(predictor)
        
        results = []
        strategies = ['draw_no_bet', 'double_chance_home', 'double_chance_away']
        thresholds = [0.50, 0.60, 0.65, 0.70]
        
        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"ESTRATÉGIA: {strategy.upper()}")
            logger.info(f"{'='*60}")
            
            for threshold in thresholds:
                metrics = betting_pred.evaluate_strategy(
                    X_test_scaled, 
                    y_test, 
                    strategy, 
                    threshold
                )
                results.append(metrics)
        
        # 5. Criar DataFrame de resultados
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('roi', ascending=False)
        
        # 6. Salvar resultados
        output_file = 'models/betting_strategies_results.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Resultados salvos em: {output_file}")
        
        # 7. Mostrar resumo
        log_separator("COMPARAÇÃO DE ESTRATÉGIAS", char='=', width=80)
        
        logger.info("\n📊 TOP 5 ESTRATÉGIAS POR ROI:\n")
        for idx, row in results_df.head(5).iterrows():
            logger.info(f"{idx+1}. {row['strategy'].upper()} (threshold={row['confidence_threshold']:.0%})")
            logger.info(f"   ROI: {row['roi']:>7.2%} | Win Rate: {row['win_rate']:>6.1%} | Apostas: {row['total_bets']:>5,}")
            logger.info("")
        
        # 8. Melhor estratégia
        log_separator("RECOMENDAÇÃO", char='=', width=80)
        best = results_df.iloc[0]
        
        logger.info(f"\n🏆 MELHOR ESTRATÉGIA IDENTIFICADA:")
        logger.info(f"\n   📌 Tipo: {best['strategy'].upper()}")
        logger.info(f"   🎯 Limiar de Confiança: {best['confidence_threshold']:.0%}")
        logger.info(f"   💰 ROI Estimado: {best['roi']:.2%}")
        logger.info(f"   ✅ Taxa de Acerto: {best['win_rate']:.1%}")
        logger.info(f"   📊 Volume de Apostas: {best['total_bets']:,} ({best['bet_rate']:.1%} dos jogos)")
        logger.info(f"   🔥 Confiança Média: {best['avg_confidence']:.1%}")
        
        # 9. Interpretação
        logger.info(f"\n{'='*80}")
        logger.info("💡 INTERPRETAÇÃO DOS RESULTADOS")
        logger.info(f"{'='*80}\n")
        
        if best['strategy'] == 'draw_no_bet':
            logger.info("✅ DRAW NO BET (Empate Anula) é a melhor opção:")
            logger.info("   • Elimina o risco do empate (devolve aposta)")
            logger.info("   • Foca em Casa vs Visitante (pontos fortes do modelo)")
            logger.info("   • Ideal para jogos equilibrados")
            logger.info("   • Odds típicas: 1.8 - 2.1")
        
        elif 'double_chance' in best['strategy']:
            logger.info("✅ DUPLA CHANCE é a melhor opção:")
            logger.info("   • Cobertura de 2 resultados (ex: Casa OU Empate)")
            logger.info("   • Menor risco, maior consistência")
            logger.info("   • Ideal para favoritos claros")
            logger.info("   • Odds típicas: 1.3 - 1.6")
        
        logger.info(f"\n⚠️  AVISO IMPORTANTE:")
        logger.info("   • Estes resultados são baseados em DADOS SINTÉTICOS")
        logger.info("   • Para resultados reais, execute: python main.py --mode train")
        logger.info("   • Depois execute: python main.py --mode betting")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Erro: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys
    success = quick_evaluate_betting_strategies()
    sys.exit(0 if success else 1)