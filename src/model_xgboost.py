"""
Treinamento do modelo XGBoost MELHORADO para predição de resultados de futebol
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import MODEL_PARAMS, TRAIN_CONFIG, MODEL_FILES, ALL_FEATURES
from src.utils import logger, log_metrics, log_feature_importance, save_json

class SoccerPredictor:
    """Classe para treinamento e avaliação do modelo"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.use_balancing = True  # Flag para usar balanceamento
        
    def prepare_data(self, master_df, use_balancing=True):
        """
        Prepara os dados para treinamento com balanceamento de classes
        
        Args:
            master_df: DataFrame com todas as features
            use_balancing: Se True, aplica SMOTE+UnderSampling
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("\n" + "="*60)
        logger.info("PREPARANDO DADOS PARA TREINAMENTO")
        logger.info("="*60)
        
        self.use_balancing = use_balancing
        
        # Separar features e target
        feature_cols = [col for col in master_df.columns 
                       if col not in ['eventId', 'date', 'homeTeamId', 'awayTeamId', 
                                     'target', 'league_name']]
        
        X = master_df[feature_cols].copy()
        y = master_df['target'].copy()
        
        # Salvar nomes das features
        self.feature_columns = list(X.columns)
        
        logger.info(f"Features utilizadas: {len(self.feature_columns)}")
        logger.info(f"Total de amostras: {len(X):,}")
        logger.info(f"\nDistribuição ORIGINAL do target:")
        logger.info(f"  Empate (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
        logger.info(f"  Vitória Casa (1): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
        logger.info(f"  Vitória Visitante (2): {(y==2).sum():,} ({(y==2).sum()/len(y)*100:.1f}%)")
        
        # Split treino/teste estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TRAIN_CONFIG['test_size'],
            random_state=TRAIN_CONFIG['random_state'],
            stratify=y
        )
        
        logger.info(f"\n✓ Dados divididos:")
        logger.info(f"  Treino: {len(X_train):,} amostras")
        logger.info(f"  Teste: {len(X_test):,} amostras")
        
        # Aplicar balanceamento de classes no treino
        if use_balancing:
            logger.info("\n--- Aplicando Balanceamento de Classes ---")
            X_train, y_train = self._balance_classes(X_train, y_train)
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Converter de volta para DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        logger.info("✓ Features normalizadas")
        
        # Armazenar para uso posterior
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _balance_classes(self, X_train, y_train):
        """
        Balanceia classes usando combinação de SMOTE e UnderSampling
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        
        Returns:
            X_train_balanced, y_train_balanced
        """
        logger.info("Aplicando SMOTE + RandomUnderSampler...")
        
        # Estratégia: 
        # 1. SMOTE para aumentar classes minoritárias (especialmente Empate)
        # 2. UnderSampling leve para reduzir classe majoritária
        
        # Calcular distribuição alvo
        class_counts = y_train.value_counts().sort_index()
        max_count = class_counts.max()
        
        # SMOTE: aumentar classes minoritárias para 70% da majoritária
        smote_strategy = {
            0: int(max_count * 0.7),  # Empate
            1: max_count,              # Vitória Casa (majoritária)
            2: int(max_count * 0.7)   # Vitória Visitante
        }
        
        # UnderSampling: reduzir majoritária para 80% do original
        under_strategy = {
            0: int(max_count * 0.7),
            1: int(max_count * 0.8),
            2: int(max_count * 0.7)
        }
        
        # Aplicar balanceamento
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=5)
        under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
        
        # Pipeline de balanceamento
        pipeline = ImbPipeline([
            ('smote', smote),
            ('under', under)
        ])
        
        X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
        
        logger.info(f"\n✓ Balanceamento aplicado:")
        logger.info(f"  Antes: {len(X_train):,} amostras")
        logger.info(f"  Depois: {len(X_balanced):,} amostras")
        logger.info(f"\nDistribuição BALANCEADA do target:")
        logger.info(f"  Empate (0): {(y_balanced==0).sum():,} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)")
        logger.info(f"  Vitória Casa (1): {(y_balanced==1).sum():,} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)")
        logger.info(f"  Vitória Visitante (2): {(y_balanced==2).sum():,} ({(y_balanced==2).sum()/len(y_balanced)*100:.1f}%)")
        
        return X_balanced, y_balanced
    
    def train_model(self, X_train=None, y_train=None):
        """
        Treina o modelo XGBoost com parâmetros otimizados
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        
        Returns:
            modelo treinado
        """
        logger.info("\n" + "="*60)
        logger.info("TREINANDO MODELO XGBOOST")
        logger.info("="*60)
        
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        # Parâmetros otimizados para melhor performance em empates
        optimized_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 7,  # Aumentado de 6 para capturar mais complexidade
            'learning_rate': 0.05,  # Reduzido para aprendizado mais cuidadoso
            'n_estimators': 300,  # Aumentado de 200
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,  # Aumentado para regularização
            'min_child_weight': 3,  # Aumentado para evitar overfit em classes minoritárias
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': 1.0  # Peso para classes desbalanceadas
        }
        
        # Criar modelo
        self.model = xgb.XGBClassifier(**optimized_params)
        
        logger.info("\nParâmetros OTIMIZADOS do modelo:")
        for param, value in optimized_params.items():
            logger.info(f"  {param}: {value}")
        
        # Treinar
        logger.info("\nIniciando treinamento...")
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (self.X_test, self.y_test)],
            verbose=False
        )
        
        logger.info("✓ Treinamento concluído!")
        
        return self.model
    
    def cross_validate(self):
        """Realiza validação cruzada"""
        logger.info("\n--- Realizando validação cruzada ---")
        
        cv = StratifiedKFold(
            n_splits=TRAIN_CONFIG['cv_folds'],
            shuffle=True,
            random_state=TRAIN_CONFIG['random_state']
        )
        
        # Criar modelo temporário para CV
        temp_model = xgb.XGBClassifier(**MODEL_PARAMS)
        
        # Realizar CV
        cv_scores = cross_val_score(
            temp_model,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring='f1_macro',  # Mudado de accuracy para f1_macro
            n_jobs=-1
        )
        
        logger.info(f"✓ Validação cruzada concluída:")
        logger.info(f"  Scores (F1-Macro): {[f'{s:.4f}' for s in cv_scores]}")
        logger.info(f"  Média: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
        
        Returns:
            dict com métricas de avaliação
        """
        logger.info("\n" + "="*60)
        logger.info("AVALIANDO MODELO")
        logger.info("="*60)
        
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Fazer predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Métricas por classe
        class_names = ['Empate', 'Vitória Casa', 'Vitória Visitante']
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Log métricas
        log_metrics(metrics, 'Teste')
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nMatriz de Confusão:")
        logger.info(f"\n{cm}")
        logger.info(f"\nLinhas = Real, Colunas = Predito")
        logger.info(f"Classes: {class_names}")
        
        # Relatório de classificação
        logger.info("\nRelatório de Classificação:")
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=class_names,
            zero_division=0
        )
        logger.info(f"\n{report}")
        
        # Feature importance
        feature_importance = log_feature_importance(
            self.model, 
            self.feature_columns,
            top_n=20
        )
        
        # Análise de erros em empates
        self._analyze_draw_errors(y_test, y_pred, y_pred_proba)
        
        # Adicionar confusion matrix ao retorno
        metrics['confusion_matrix'] = cm.tolist()
        metrics['feature_importance'] = feature_importance.to_dict() if feature_importance is not None else None
        
        return metrics
    
    def _analyze_draw_errors(self, y_test, y_pred, y_pred_proba):
        """Analisa erros específicos na predição de empates"""
        logger.info("\n" + "="*60)
        logger.info("ANÁLISE DE ERROS - EMPATES")
        logger.info("="*60)
        
        # Empates reais
        draw_mask = y_test == 0
        draws_real = y_test[draw_mask]
        draws_pred = y_pred[draw_mask]
        draws_proba = y_pred_proba[draw_mask]
        
        if len(draws_real) > 0:
            # Análise
            correct_draws = (draws_pred == 0).sum()
            pred_as_home = (draws_pred == 1).sum()
            pred_as_away = (draws_pred == 2).sum()
            
            logger.info(f"\nDe {len(draws_real)} empates reais:")
            logger.info(f"  ✓ Acertou: {correct_draws} ({correct_draws/len(draws_real)*100:.1f}%)")
            logger.info(f"  ✗ Previu V. Casa: {pred_as_home} ({pred_as_home/len(draws_real)*100:.1f}%)")
            logger.info(f"  ✗ Previu V. Visitante: {pred_as_away} ({pred_as_away/len(draws_real)*100:.1f}%)")
            
            # Confiança média nas predições
            avg_confidence_draw = draws_proba[:, 0].mean()
            avg_confidence_home = draws_proba[:, 1].mean()
            avg_confidence_away = draws_proba[:, 2].mean()
            
            logger.info(f"\nConfiança média das probabilidades:")
            logger.info(f"  P(Empate): {avg_confidence_draw:.3f}")
            logger.info(f"  P(V. Casa): {avg_confidence_home:.3f}")
            logger.info(f"  P(V. Visitante): {avg_confidence_away:.3f}")
    
    def save_model(self):
        """Salva o modelo, scaler e feature columns"""
        logger.info("\n--- Salvando modelo ---")
        
        try:
            # Salvar modelo XGBoost
            self.model.save_model(MODEL_FILES['model'])
            logger.info(f"✓ Modelo salvo em: {MODEL_FILES['model']}")
            
            # Salvar scaler
            with open(MODEL_FILES['scaler'], 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"✓ Scaler salvo em: {MODEL_FILES['scaler']}")
            
            # Salvar feature columns
            save_json(self.feature_columns, MODEL_FILES['features'])
            logger.info(f"✓ Features salvas em: {MODEL_FILES['features']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self):
        """Carrega modelo, scaler e feature columns salvos"""
        logger.info("\n--- Carregando modelo ---")
        
        try:
            # Carregar modelo
            self.model = xgb.XGBClassifier()
            self.model.load_model(MODEL_FILES['model'])
            logger.info(f"✓ Modelo carregado de: {MODEL_FILES['model']}")
            
            # Carregar scaler
            with open(MODEL_FILES['scaler'], 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"✓ Scaler carregado de: {MODEL_FILES['scaler']}")
            
            # Carregar feature columns
            from src.utils import load_json
            self.feature_columns = load_json(MODEL_FILES['features'])
            logger.info(f"✓ Features carregadas de: {MODEL_FILES['features']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def predict(self, X):
        """
        Faz predições para novos dados
        
        Args:
            X: DataFrame com features
        
        Returns:
            predições e probabilidades
        """
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predizer
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

def train_and_evaluate(master_df, use_balancing=True):
    """
    Função principal para treinar e avaliar o modelo
    
    Args:
        master_df: DataFrame com features
        use_balancing: Se True, aplica balanceamento de classes
    
    Returns:
        predictor treinado e métricas
    """
    # Criar predictor
    predictor = SoccerPredictor()
    
    # Preparar dados (com balanceamento)
    predictor.prepare_data(master_df, use_balancing=use_balancing)
    
    # Validação cruzada
    cv_scores = predictor.cross_validate()
    
    # Treinar modelo
    predictor.train_model()
    
    # Avaliar
    metrics = predictor.evaluate()
    
    # Salvar modelo
    predictor.save_model()
    
    logger.info("\n" + "="*60)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("="*60)
    logger.info(f"\n📊 Métricas Finais:")
    logger.info(f"  Acurácia: {metrics['accuracy']:.2%}")
    logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Empate: {metrics['f1_Empate']:.4f}")
    logger.info(f"  Recall-Empate: {metrics['recall_Empate']:.2%}")
    
    return predictor, metrics

if __name__ == "__main__":
    # Teste do módulo
    from src.etl import load_and_preprocess_data
    from src.feature_engineering import create_features
    
    logger.info("Executando pipeline completo de treinamento...")
    
    # Carregar e processar dados
    data = load_and_preprocess_data()
    
    # Criar features
    master_df = create_features(data)
    
    # Treinar e avaliar
    predictor, metrics = train_and_evaluate(master_df, use_balancing=True)
    
    logger.info("\n✓ Pipeline de treinamento concluído!")