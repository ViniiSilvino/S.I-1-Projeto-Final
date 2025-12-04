"""
Implementação de Estratégias de Apostas Alternativas
Transformando a dificuldade com empates em oportunidades
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.utils import logger

class BettingStrategiesPredictor:
    """
    Classe para implementar estratégias de apostas alternativas:
    1. Empate Anula (Draw No Bet) - Casa vs Visitante, empate devolve
    2. Dupla Chance - Casa OU Empate, Visitante OU Empate
    3. Over/Under Gols
    """
    
    def __init__(self, base_predictor):
        """
        Args:
            base_predictor: SoccerPredictor já treinado
        """
        self.base_predictor = base_predictor
        self.strategies = {
            'draw_no_bet': self._draw_no_bet,
            'double_chance_home': self._double_chance_home,
            'double_chance_away': self._double_chance_away,
            'home_or_draw': self._home_or_draw,  # Alias
            'away_or_draw': self._away_or_draw   # Alias
        }
    
    def predict_with_strategy(self, X, strategy='draw_no_bet', confidence_threshold=0.6):
        """
        Faz predição usando estratégia alternativa
        
        Args:
            X: Features
            strategy: Estratégia a usar
            confidence_threshold: Mínimo de confiança para fazer aposta
            
        Returns:
            predictions, probabilities, bet_decisions
        """
        # Obter probabilidades do modelo base
        base_pred = self.base_predictor.model.predict(X)
        base_proba = self.base_predictor.model.predict_proba(X)
        
        # Aplicar estratégia
        strategy_func = self.strategies.get(strategy)
        if strategy_func is None:
            raise ValueError(f"Estratégia inválida: {strategy}")
        
        predictions, confidence, should_bet = strategy_func(
            base_proba, 
            confidence_threshold
        )
        
        return predictions, confidence, should_bet
    
    def _draw_no_bet(self, proba, threshold):
        """
        Empate Anula (Draw No Bet)
        Prevê Casa (1) ou Visitante (2), ignorando empates
        Se empate ocorrer, aposta é devolvida
        """
        # Redistribuir probabilidades ignorando empate
        home_away_proba = proba[:, [1, 2]]  # Apenas Casa e Visitante
        normalized_proba = home_away_proba / home_away_proba.sum(axis=1, keepdims=True)
        
        # Predição: 1 (Casa) ou 2 (Visitante)
        predictions = np.argmax(normalized_proba, axis=1) + 1
        
        # Confiança: prob da classe escolhida
        confidence = np.max(normalized_proba, axis=1)
        
        # Decidir se aposta (apenas se confiança > threshold)
        should_bet = confidence >= threshold
        
        return predictions, confidence, should_bet
    
    def _double_chance_home(self, proba, threshold):
        """
        Dupla Chance: Casa OU Empate
        Ganha se Casa vencer OU empatar
        """
        # P(Casa OU Empate) = P(Casa) + P(Empate)
        home_or_draw_proba = proba[:, 0] + proba[:, 1]
        
        # Predição: 1 se P(Casa ou Empate) > 0.5, senão 2
        predictions = np.where(home_or_draw_proba > 0.5, 1, 2)
        
        # Confiança
        confidence = np.where(
            home_or_draw_proba > 0.5,
            home_or_draw_proba,
            1 - home_or_draw_proba
        )
        
        should_bet = confidence >= threshold
        
        return predictions, confidence, should_bet
    
    def _double_chance_away(self, proba, threshold):
        """
        Dupla Chance: Visitante OU Empate
        Ganha se Visitante vencer OU empatar
        """
        # P(Visitante OU Empate) = P(Visitante) + P(Empate)
        away_or_draw_proba = proba[:, 0] + proba[:, 2]
        
        # Predição
        predictions = np.where(away_or_draw_proba > 0.5, 2, 1)
        
        # Confiança
        confidence = np.where(
            away_or_draw_proba > 0.5,
            away_or_draw_proba,
            1 - away_or_draw_proba
        )
        
        should_bet = confidence >= threshold
        
        return predictions, confidence, should_bet
    
    def _home_or_draw(self, proba, threshold):
        """Alias para double_chance_home"""
        return self._double_chance_home(proba, threshold)
    
    def _away_or_draw(self, proba, threshold):
        """Alias para double_chance_away"""
        return self._double_chance_away(proba, threshold)
    
    def evaluate_strategy(self, X_test, y_test, strategy='draw_no_bet', 
                         confidence_threshold=0.6):
        """
        Avalia estratégia no conjunto de teste
        
        Returns:
            dict com métricas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AVALIANDO ESTRATÉGIA: {strategy.upper()}")
        logger.info(f"Limiar de Confiança: {confidence_threshold:.0%}")
        logger.info(f"{'='*60}")
        
        # Fazer predições
        predictions, confidence, should_bet = self.predict_with_strategy(
            X_test, strategy, confidence_threshold
        )
        
        # Converter y_test para formato compatível
        if strategy == 'draw_no_bet':
            # Para Draw No Bet, empates (0) são "push" (sem perda/ganho)
            y_test_adjusted = y_test.copy()
            mask_bet = should_bet
            
            # Calcular resultados
            correct = (predictions[mask_bet] == y_test_adjusted[mask_bet])
            draws = (y_test_adjusted[mask_bet] == 0)
            
            wins = correct & ~draws
            losses = ~correct & ~draws
            pushes = draws
            
            total_bets = mask_bet.sum()
            win_rate = wins.sum() / total_bets if total_bets > 0 else 0
            loss_rate = losses.sum() / total_bets if total_bets > 0 else 0
            push_rate = pushes.sum() / total_bets if total_bets > 0 else 0
            
            logger.info(f"\nTotal de apostas realizadas: {total_bets:,} "
                       f"({total_bets/len(y_test)*100:.1f}% dos jogos)")
            logger.info(f"  ✓ Vitórias: {wins.sum():,} ({win_rate:.1%})")
            logger.info(f"  ✗ Derrotas: {losses.sum():,} ({loss_rate:.1%})")
            logger.info(f"  ↔ Empates (Push): {pushes.sum():,} ({push_rate:.1%})")
            
            # ROI simulado (odds médias ~1.9 para DNB)
            avg_odds = 1.9
            roi = (wins.sum() * avg_odds - losses.sum()) / total_bets if total_bets > 0 else 0
            logger.info(f"\n💰 ROI Estimado (odds {avg_odds}): {roi:.2%}")
            
            metrics = {
                'strategy': strategy,
                'confidence_threshold': confidence_threshold,
                'total_bets': int(total_bets),
                'bet_rate': total_bets / len(y_test),
                'wins': int(wins.sum()),
                'losses': int(losses.sum()),
                'pushes': int(pushes.sum()),
                'win_rate': float(win_rate),
                'loss_rate': float(loss_rate),
                'push_rate': float(push_rate),
                'roi': float(roi),
                'avg_confidence': float(confidence[mask_bet].mean())
            }
            
        elif 'double_chance' in strategy or 'or_draw' in strategy:
            # Para Dupla Chance
            mask_bet = should_bet
            
            if strategy in ['double_chance_home', 'home_or_draw']:
                # Casa OU Empate: ganha se y_test é 0 ou 1
                wins = ((y_test[mask_bet] == 0) | (y_test[mask_bet] == 1))
            else:  # double_chance_away, away_or_draw
                # Visitante OU Empate: ganha se y_test é 0 ou 2
                wins = ((y_test[mask_bet] == 0) | (y_test[mask_bet] == 2))
            
            losses = ~wins
            
            total_bets = mask_bet.sum()
            win_rate = wins.sum() / total_bets if total_bets > 0 else 0
            loss_rate = losses.sum() / total_bets if total_bets > 0 else 0
            
            logger.info(f"\nTotal de apostas realizadas: {total_bets:,} "
                       f"({total_bets/len(y_test)*100:.1f}% dos jogos)")
            logger.info(f"  ✓ Vitórias: {wins.sum():,} ({win_rate:.1%})")
            logger.info(f"  ✗ Derrotas: {losses.sum():,} ({loss_rate:.1%})")
            
            # ROI simulado (odds médias ~1.4 para Dupla Chance)
            avg_odds = 1.4
            roi = (wins.sum() * avg_odds - total_bets) / total_bets if total_bets > 0 else 0
            logger.info(f"\n💰 ROI Estimado (odds {avg_odds}): {roi:.2%}")
            
            metrics = {
                'strategy': strategy,
                'confidence_threshold': confidence_threshold,
                'total_bets': int(total_bets),
                'bet_rate': total_bets / len(y_test),
                'wins': int(wins.sum()),
                'losses': int(losses.sum()),
                'win_rate': float(win_rate),
                'loss_rate': float(loss_rate),
                'roi': float(roi),
                'avg_confidence': float(confidence[mask_bet].mean())
            }
        
        return metrics
    
    def compare_strategies(self, X_test, y_test, thresholds=[0.5, 0.6, 0.7]):
        """
        Compara todas as estratégias com diferentes limiares
        """
        logger.info(f"\n{'='*80}")
        logger.info("COMPARAÇÃO DE ESTRATÉGIAS DE APOSTAS")
        logger.info(f"{'='*80}")
        
        results = []
        
        strategies_to_test = [
            'draw_no_bet',
            'double_chance_home',
            'double_chance_away'
        ]
        
        for strategy in strategies_to_test:
            for threshold in thresholds:
                metrics = self.evaluate_strategy(
                    X_test, y_test, 
                    strategy, threshold
                )
                results.append(metrics)
        
        # Criar DataFrame de resultados
        results_df = pd.DataFrame(results)
        
        logger.info(f"\n{'='*80}")
        logger.info("RESUMO COMPARATIVO")
        logger.info(f"{'='*80}")
        
        # Ordenar por ROI
        results_df = results_df.sort_values('roi', ascending=False)
        
        for idx, row in results_df.iterrows():
            logger.info(f"\n{row['strategy'].upper()} (threshold={row['confidence_threshold']:.0%})")
            logger.info(f"  Apostas: {row['total_bets']:,} ({row['bet_rate']:.1%})")
            logger.info(f"  Win Rate: {row['win_rate']:.1%}")
            logger.info(f"  ROI: {row['roi']:.2%}")
            logger.info(f"  Confiança Média: {row['avg_confidence']:.1%}")
        
        return results_df


# Função auxiliar para integração com o pipeline existente
def evaluate_betting_strategies(predictor, X_test, y_test):
    """
    Avalia estratégias de apostas usando o modelo treinado
    
    Args:
        predictor: SoccerPredictor já treinado e com dados carregados
        X_test: Features de teste
        y_test: Target de teste
    """
    betting_pred = BettingStrategiesPredictor(predictor)
    
    # Comparar estratégias
    results_df = betting_pred.compare_strategies(
        X_test, y_test,
        thresholds=[0.50, 0.60, 0.65, 0.70]
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("🎯 MELHOR ESTRATÉGIA")
    logger.info(f"{'='*80}")
    
    best = results_df.iloc[0]
    logger.info(f"\n{best['strategy'].upper()}")
    logger.info(f"Limiar: {best['confidence_threshold']:.0%}")
    logger.info(f"ROI: {best['roi']:.2%}")
    logger.info(f"Win Rate: {best['win_rate']:.1%}")
    
    return results_df, betting_pred


if __name__ == "__main__":
    # Exemplo de uso
    from src.model_xgboost import SoccerPredictor
    from src.etl import load_and_preprocess_data
    from src.feature_engineering import create_features
    
    # Carregar modelo
    predictor = SoccerPredictor()
    if not predictor.load_model():
        logger.error("Erro ao carregar modelo. Execute 'python main.py --mode train' primeiro.")
        exit(1)
    
    # Carregar dados
    data = load_and_preprocess_data()
    master_df = create_features(data)
    
    # Preparar dados (sem balanceamento para avaliação)
    predictor.prepare_data(master_df, use_balancing=False)
    
    # Avaliar estratégias
    results_df, betting_pred = evaluate_betting_strategies(
        predictor,
        predictor.X_test,
        predictor.y_test
    )