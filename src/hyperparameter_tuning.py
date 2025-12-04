"""
Otimização de Hiperparâmetros usando Optuna
"""
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from src.utils import logger
import json
import os

class HyperparameterTuner:
    """Classe para otimização de hiperparâmetros com Optuna"""
    
    def __init__(self, X_train, y_train, n_trials=50, cv_folds=5):
        """
        Inicializa o tuner
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            n_trials: Número de tentativas do Optuna
            cv_folds: Folds para validação cruzada
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.study = None
        
    def objective(self, trial):
        """
        Função objetivo para o Optuna otimizar
        
        Args:
            trial: Trial do Optuna
        
        Returns:
            F1-Score macro médio da validação cruzada
        """
        # Definir espaço de busca de hiperparâmetros
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'tree_method': 'hist',  # Mais rápido
            'verbosity': 0,
            
            # Hiperparâmetros a otimizar
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
        }
        
        # Criar modelo
        model = xgb.XGBClassifier(**params)
        
        # Validação cruzada estratificada
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Scorer customizado - F1 Macro (foco em todas as classes)
        scorer = make_scorer(f1_score, average='macro')
        
        # Avaliar
        scores = cross_val_score(
            model, 
            self.X_train, 
            self.y_train, 
            cv=cv, 
            scoring=scorer,
            n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize(self, timeout=None):
        """
        Executa a otimização
        
        Args:
            timeout: Tempo máximo em segundos (None = sem limite)
        
        Returns:
            Melhores hiperparâmetros encontrados
        """
        logger.info("\n" + "="*60)
        logger.info("INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS")
        logger.info("="*60)
        logger.info(f"\n📊 Configuração:")
        logger.info(f"  Trials: {self.n_trials}")
        logger.info(f"  CV Folds: {self.cv_folds}")
        logger.info(f"  Timeout: {timeout if timeout else 'Sem limite'}")
        logger.info(f"  Métrica: F1-Score Macro")
        logger.info(f"\n🔍 Espaço de busca:")
        logger.info(f"  max_depth: [4, 10]")
        logger.info(f"  learning_rate: [0.01, 0.3]")
        logger.info(f"  n_estimators: [100, 500]")
        logger.info(f"  min_child_weight: [1, 10]")
        logger.info(f"  gamma: [0, 0.5]")
        logger.info(f"  subsample: [0.6, 1.0]")
        logger.info(f"  colsample_bytree: [0.6, 1.0]")
        logger.info(f"  reg_alpha: [0, 1.0]")
        logger.info(f"  reg_lambda: [0, 2.0]")
        
        # Criar estudo
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='xgboost_optimization'
        )
        
        # Callback para logging
        def logging_callback(study, trial):
            logger.info(f"\n🔄 Trial {trial.number + 1}/{self.n_trials}")
            logger.info(f"  F1-Score: {trial.value:.4f}")
            logger.info(f"  Melhor até agora: {study.best_value:.4f}")
        
        # Otimizar
        logger.info(f"\n⚙️  Iniciando otimização...\n")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=timeout,
            callbacks=[logging_callback],
            show_progress_bar=True
        )
        
        # Resultados
        self.best_params = self.study.best_params
        
        logger.info("\n" + "="*60)
        logger.info("OTIMIZAÇÃO CONCLUÍDA")
        logger.info("="*60)
        logger.info(f"\n✅ Melhor F1-Score: {self.study.best_value:.4f}")
        logger.info(f"\n🎯 Melhores Hiperparâmetros:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        return self.best_params
    
    def get_best_model_params(self):
        """
        Retorna os parâmetros completos para criar o melhor modelo
        
        Returns:
            Dict com todos os parâmetros
        """
        if self.best_params is None:
            raise ValueError("Execute optimize() primeiro!")
        
        full_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            **self.best_params
        }
        
        return full_params
    
    def save_results(self, filepath='models/optuna_results.json'):
        """
        Salva os resultados da otimização
        
        Args:
            filepath: Caminho do arquivo
        """
        if self.best_params is None:
            raise ValueError("Execute optimize() primeiro!")
        
        results = {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                }
                for trial in self.study.trials
            ]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"\n💾 Resultados salvos em: {filepath}")
    
    def plot_optimization_history(self, save_path='models/optimization_history.png'):
        """
        Plota o histórico de otimização
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Gráfico de otimização salvo em: {save_path}")
            plt.close()
        except ImportError:
            logger.warning("matplotlib não instalado. Pulando visualização.")
    
    def plot_param_importances(self, save_path='models/param_importances.png'):
        """
        Plota importância dos hiperparâmetros
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Importância dos parâmetros salva em: {save_path}")
            plt.close()
        except ImportError:
            logger.warning("matplotlib não instalado. Pulando visualização.")

def run_hyperparameter_optimization(X_train, y_train, n_trials=50, timeout=None):
    """
    Função principal para executar otimização de hiperparâmetros
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        n_trials: Número de trials
        timeout: Tempo máximo em segundos
    
    Returns:
        Melhores parâmetros encontrados
    """
    # Criar tuner
    tuner = HyperparameterTuner(X_train, y_train, n_trials=n_trials)
    
    # Otimizar
    best_params = tuner.optimize(timeout=timeout)
    
    # Salvar resultados
    tuner.save_results()
    
    # Plotar resultados (se matplotlib disponível)
    tuner.plot_optimization_history()
    tuner.plot_param_importances()
    
    return best_params, tuner

def quick_tune(X_train, y_train, n_trials=20):
    """
    Otimização rápida com menos trials
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        n_trials: Número de trials (padrão: 20)
    
    Returns:
        Melhores parâmetros
    """
    logger.info("\n🚀 MODO RÁPIDO - Otimização com menos trials")
    return run_hyperparameter_optimization(X_train, y_train, n_trials=n_trials)

def extensive_tune(X_train, y_train, n_trials=100, timeout=3600):
    """
    Otimização extensiva com mais trials e timeout
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        n_trials: Número de trials (padrão: 100)
        timeout: Tempo máximo em segundos (padrão: 1 hora)
    
    Returns:
        Melhores parâmetros
    """
    logger.info("\n🔬 MODO EXTENSIVO - Otimização completa")
    return run_hyperparameter_optimization(X_train, y_train, n_trials=n_trials, timeout=timeout)

if __name__ == "__main__":
    # Teste do módulo
    from src.etl import load_and_preprocess_data
    from src.feature_engineering import create_features
    from src.model_xgboost import SoccerPredictor
    
    logger.info("=== TESTE DE OTIMIZAÇÃO DE HIPERPARÂMETROS ===\n")
    
    # Carregar dados
    logger.info("Carregando dados...")
    data = load_and_preprocess_data()
    master_df = create_features(data)
    
    # Preparar dados
    predictor = SoccerPredictor()
    predictor.prepare_data(master_df, use_balancing=True)
    
    # Otimização rápida (apenas para teste)
    logger.info("\nExecutando otimização rápida com 10 trials...")
    best_params, tuner = quick_tune(predictor.X_train, predictor.y_train, n_trials=10)
    
    logger.info("\n✓ Teste concluído!")
    logger.info(f"Melhores parâmetros: {best_params}")