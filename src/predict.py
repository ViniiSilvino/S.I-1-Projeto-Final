"""
Módulo para predição de resultados de novas partidas
"""
import pandas as pd
import numpy as np
from model_xgboost import SoccerPredictor
from utils import logger

class MatchPredictor:
    """Classe para fazer predições de partidas"""
    
    def __init__(self):
        self.predictor = SoccerPredictor()
        self.loaded = False
    
    def load_model(self):
        """Carrega o modelo treinado"""
        logger.info("Carregando modelo para predição...")
        success = self.predictor.load_model()
        
        if success:
            self.loaded = True
            logger.info("✓ Modelo carregado e pronto para predição")
        else:
            logger.error("✗ Falha ao carregar modelo")
        
        return success
    
    def predict_match(self, match_features):
        """
        Prediz o resultado de uma partida
        
        Args:
            match_features: DataFrame ou dict com features da partida
        
        Returns:
            dict com predição e probabilidades
        """
        if not self.loaded:
            logger.error("Modelo não carregado. Execute load_model() primeiro.")
            return None
        
        # Converter para DataFrame se for dict
        if isinstance(match_features, dict):
            match_features = pd.DataFrame([match_features])
        
        # Verificar se tem todas as features necessárias
        missing_features = set(self.predictor.feature_columns) - set(match_features.columns)
        if missing_features:
            logger.error(f"Features faltando: {missing_features}")
            return None
        
        # Selecionar apenas as features necessárias na ordem correta
        X = match_features[self.predictor.feature_columns]
        
        # Fazer predição
        predictions, probabilities = self.predictor.predict(X)
        
        # Preparar resultado
        class_names = ['Empate', 'Vitória Casa', 'Vitória Visitante']
        result = {
            'prediction': int(predictions[0]),
            'prediction_label': class_names[predictions[0]],
            'probabilities': {
                'empate': float(probabilities[0][0]),
                'vitoria_casa': float(probabilities[0][1]),
                'vitoria_visitante': float(probabilities[0][2])
            },
            'confidence': float(probabilities[0][predictions[0]])
        }
        
        return result
    
    def predict_multiple_matches(self, matches_df):
        """
        Prediz resultados para múltiplas partidas
        
        Args:
            matches_df: DataFrame com features de múltiplas partidas
        
        Returns:
            DataFrame com predições
        """
        if not self.loaded:
            logger.error("Modelo não carregado. Execute load_model() primeiro.")
            return None
        
        logger.info(f"Predizendo {len(matches_df)} partidas...")
        
        # Verificar features
        missing_features = set(self.predictor.feature_columns) - set(matches_df.columns)
        if missing_features:
            logger.error(f"Features faltando: {missing_features}")
            return None
        
        # Selecionar features
        X = matches_df[self.predictor.feature_columns]
        
        # Fazer predições
        predictions, probabilities = self.predictor.predict(X)
        
        # Criar DataFrame de resultados
        results_df = matches_df.copy()
        results_df['prediction'] = predictions
        results_df['prob_empate'] = probabilities[:, 0]
        results_df['prob_vitoria_casa'] = probabilities[:, 1]
        results_df['prob_vitoria_visitante'] = probabilities[:, 2]
        results_df['confidence'] = probabilities[np.arange(len(predictions)), predictions]
        
        # Mapear predições para labels
        class_map = {0: 'Empate', 1: 'Vitória Casa', 2: 'Vitória Visitante'}
        results_df['prediction_label'] = results_df['prediction'].map(class_map)
        
        logger.info("✓ Predições concluídas")
        
        return results_df
    
    def explain_prediction(self, match_features, top_n=10):
        """
        Explica uma predição mostrando as features mais importantes
        
        Args:
            match_features: Features da partida
            top_n: Número de features a mostrar
        
        Returns:
            DataFrame com explicação
        """
        if not self.loaded:
            logger.error("Modelo não carregado.")
            return None
        
        # Fazer predição
        result = self.predict_match(match_features)
        
        if result is None:
            return None
        
        # Pegar feature importance do modelo
        feature_importance = self.predictor.model.feature_importances_
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': self.predictor.feature_columns,
            'importance': feature_importance
        })
        
        # Adicionar valor da feature para esta partida
        if isinstance(match_features, dict):
            match_features = pd.DataFrame([match_features])
        
        importance_df['value'] = importance_df['feature'].map(
            lambda f: match_features[f].values[0] if f in match_features.columns else None
        )
        
        # Ordenar por importância
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"\nPredição: {result['prediction_label']}")
        logger.info(f"Confiança: {result['confidence']:.2%}")
        logger.info(f"\nTop {top_n} Features Mais Importantes:")
        logger.info(f"\n{importance_df.to_string(index=False)}")
        
        return importance_df, result

def predict_new_matches(matches_data):
    """
    Função principal para predizer novas partidas
    
    Args:
        matches_data: DataFrame ou dict com dados das partidas
    
    Returns:
        Predições
    """
    predictor = MatchPredictor()
    
    # Carregar modelo
    if not predictor.load_model():
        return None
    
    # Predizer
    if isinstance(matches_data, pd.DataFrame):
        results = predictor.predict_multiple_matches(matches_data)
    else:
        results = predictor.predict_match(matches_data)
    
    return results

def predict_from_team_ids(home_team_id, away_team_id, data_dict):
    """
    Prediz resultado a partir dos IDs dos times (cria features automaticamente)
    
    Args:
        home_team_id: ID do time da casa
        away_team_id: ID do time visitante
        data_dict: Dicionário com dados processados (standings, team_stats, etc)
    
    Returns:
        Predição do resultado
    """
    from feature_engineering import FeatureEngineer
    
    logger.info(f"\nPredizendo partida: Time {home_team_id} (casa) vs Time {away_team_id} (visitante)")
    
    # Criar um fixture fictício
    fake_fixture = pd.DataFrame([{
        'eventId': 999999,
        'homeTeamId': home_team_id,
        'awayTeamId': away_team_id,
        'date': pd.Timestamp.now(),
        'leagueId': 0,
        'statusId': 28,
        'result': np.nan
    }])
    
    # Adicionar ao data_dict
    data_dict['fixtures'] = fake_fixture
    
    # Criar features
    engineer = FeatureEngineer(data_dict)
    
    # Adicionar features manualmente (simplificado)
    try:
        engineer._initialize_master_df()
        engineer._add_form_features()
        engineer._add_performance_features()
        engineer._add_match_stats_features()
        engineer._add_lineup_features()
        engineer._add_derived_features()
        
        match_features = engineer.master_df.iloc[0].to_dict()
        
        # Remover colunas não-features
        for col in ['eventId', 'date', 'homeTeamId', 'awayTeamId', 'target']:
            match_features.pop(col, None)
        
        # Predizer
        result = predict_new_matches(match_features)
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao criar features: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    logger.info("=== TESTE DE PREDIÇÃO ===\n")
    
    # Exemplo 1: Predição com features prontas
    logger.info("Exemplo 1: Predição com features prontas")
    
    match_features = {
        'home_recent_wins': 3,
        'home_recent_draws': 1,
        'home_recent_losses': 1,
        'home_form_points': 10,
        'away_recent_wins': 2,
        'away_recent_draws': 2,
        'away_recent_losses': 1,
        'away_form_points': 8,
        'home_goals_per_game': 1.8,
        'away_goals_per_game': 1.5,
        'home_goals_against_per_game': 1.0,
        'away_goals_against_per_game': 1.2,
        'home_goal_difference': 8,
        'away_goal_difference': 3,
        'home_points': 45,
        'away_points': 38,
        'home_wins': 14,
        'away_wins': 11,
        'home_losses': 3,
        'away_losses': 6,
        'home_draws': 5,
        'away_draws': 5,
        'home_possession_avg': 55.0,
        'away_possession_avg': 48.0,
        'home_pass_accuracy': 82.0,
        'away_pass_accuracy': 78.0,
        'home_shot_accuracy': 35.0,
        'away_shot_accuracy': 32.0,
        'home_avg_age': 26.5,
        'away_avg_age': 27.0,
        'home_avg_height': 1.82,
        'away_avg_height': 1.80,
        'home_avg_weight': 77.0,
        'away_avg_weight': 76.0
    }
    
    predictor = MatchPredictor()
    if predictor.load_model():
        result = predictor.predict_match(match_features)
        
        if result:
            print(f"\n{'='*50}")
            print(f"RESULTADO DA PREDIÇÃO")
            print(f"{'='*50}")
            print(f"Predição: {result['prediction_label']}")
            print(f"Confiança: {result['confidence']:.2%}")
            print(f"\nProbabilidades:")
            print(f"  Empate: {result['probabilities']['empate']:.2%}")
            print(f"  Vitória Casa: {result['probabilities']['vitoria_casa']:.2%}")
            print(f"  Vitória Visitante: {result['probabilities']['vitoria_visitante']:.2%}")
            
            # Explicar predição
            print(f"\n{'='*50}")
            print(f"EXPLICAÇÃO DA PREDIÇÃO")
            print(f"{'='*50}")
            predictor.explain_prediction(match_features, top_n=10)