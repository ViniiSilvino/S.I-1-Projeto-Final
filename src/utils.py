"""
Funções utilitárias para o projeto de predição de partidas de futebol
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from config import LOG_FILE, LOG_FORMAT, LOG_LEVEL, LBS_TO_KG, FEET_TO_METERS, INCHES_TO_METERS, FORM_POINTS

# ========== CONFIGURAÇÃO DE LOGGING ==========

def setup_logger():
    """Configura o logger do projeto"""
    # Criar diretório de logs se não existir
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configurar logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Criar logger global
logger = setup_logger()

# ========== FUNÇÕES DE CONVERSÃO ==========

def parse_weight(weight_str):
    """
    Converte peso de lbs para kg
    
    Args:
        weight_str: String com peso (ex: "180 lbs", "180")
    
    Returns:
        Peso em kg (float) ou None
    """
    if pd.isna(weight_str):
        return None
    
    try:
        # Remover "lbs" e espaços
        weight_str = str(weight_str).replace('lbs', '').strip()
        weight_lbs = float(weight_str)
        return round(weight_lbs * LBS_TO_KG, 2)
    except (ValueError, AttributeError):
        return None

def parse_height(height_str):
    """
    Converte altura de feet'inches" para metros
    
    Args:
        height_str: String com altura (ex: "6'2\"", "6'2")
    
    Returns:
        Altura em metros (float) ou None
    """
    if pd.isna(height_str):
        return None
    
    try:
        # Limpar string
        height_str = str(height_str).replace('"', '').replace("'", ' ').strip()
        
        # Separar feet e inches
        parts = height_str.split()
        
        if len(parts) == 2:
            feet = float(parts[0])
            inches = float(parts[1])
        elif len(parts) == 1:
            # Se só tem um valor, assumir que é feet
            feet = float(parts[0])
            inches = 0
        else:
            return None
        
        # Converter para metros
        total_meters = (feet * FEET_TO_METERS) + (inches * INCHES_TO_METERS)
        return round(total_meters, 2)
        
    except (ValueError, AttributeError, IndexError):
        return None

def calculate_bmi(weight_kg, height_m):
    """
    Calcula o BMI (Body Mass Index)
    
    Args:
        weight_kg: Peso em kg
        height_m: Altura em metros
    
    Returns:
        BMI (float) ou None
    """
    if pd.isna(weight_kg) or pd.isna(height_m) or height_m == 0:
        return None
    
    try:
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    except (TypeError, ZeroDivisionError):
        return None

def parse_form_string(form_str):
    """
    Parse da string de forma (ex: "WWDLL") e calcula estatísticas
    
    Args:
        form_str: String com resultados recentes (W=Win, D=Draw, L=Loss)
    
    Returns:
        Dict com wins, draws, losses, points
    """
    if pd.isna(form_str) or not isinstance(form_str, str):
        return {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
    
    form_str = str(form_str).upper().strip()
    
    wins = form_str.count('W')
    draws = form_str.count('D')
    losses = form_str.count('L')
    
    points = (wins * FORM_POINTS['W'] + 
              draws * FORM_POINTS['D'] + 
              losses * FORM_POINTS['L'])
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'points': points
    }

# ========== FUNÇÕES MATEMÁTICAS ==========

def safe_divide(numerator, denominator, default=0):
    """
    Divisão segura que evita divisão por zero
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se divisão não for possível
    
    Returns:
        Resultado da divisão ou valor padrão
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def calculate_percentage(part, total, decimals=2):
    """
    Calcula porcentagem
    
    Args:
        part: Parte
        total: Total
        decimals: Número de casas decimais
    
    Returns:
        Porcentagem ou 0
    """
    result = safe_divide(part, total, 0) * 100
    return round(result, decimals)

# ========== FUNÇÕES DE VALIDAÇÃO ==========

def validate_dataframe(df, required_columns, df_name="DataFrame"):
    """
    Valida se um DataFrame possui as colunas necessárias
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de colunas obrigatórias
        df_name: Nome do DataFrame (para logging)
    
    Returns:
        True se válido, False caso contrário
    """
    if df is None or df.empty:
        logger.error(f"{df_name} está vazio ou None")
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        logger.error(f"{df_name} faltam colunas: {missing_cols}")
        return False
    
    return True

def check_data_quality(df, df_name="DataFrame"):
    """
    Verifica qualidade dos dados e loga estatísticas
    
    Args:
        df: DataFrame a verificar
        df_name: Nome do DataFrame
    """
    logger.info(f"\nQualidade de dados - {df_name}:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info(f"  Valores nulos:")
        for col, count in null_counts[null_counts > 0].items():
            pct = (count / len(df)) * 100
            logger.info(f"    {col}: {count:,} ({pct:.1f}%)")
    else:
        logger.info(f"  ✓ Sem valores nulos")
    
    # Duplicatas
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"  ⚠ Linhas duplicadas: {duplicates:,}")
    else:
        logger.info(f"  ✓ Sem duplicatas")

def check_class_balance(y, class_names=None):
    """
    Verifica balanceamento das classes
    
    Args:
        y: Array com labels
        class_names: Nomes das classes (opcional)
    
    Returns:
        Dict com contagens e porcentagens
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    balance_info = {}
    
    logger.info("\nBalanceamento de classes:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        pct = (count / total) * 100
        class_name = class_names[i] if class_names else f"Classe {cls}"
        logger.info(f"  {class_name}: {count:,} ({pct:.1f}%)")
        
        balance_info[int(cls)] = {
            'count': int(count),
            'percentage': round(pct, 2)
        }
    
    return balance_info

# ========== FUNÇÕES DE ARQUIVO ==========

def save_json(data, filepath):
    """
    Salva dados em formato JSON
    
    Args:
        data: Dados a salvar
        filepath: Caminho do arquivo
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON em {filepath}: {e}")
        return False

def load_json(filepath):
    """
    Carrega dados de arquivo JSON
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        Dados carregados ou None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar JSON de {filepath}: {e}")
        return None

def ensure_dir(directory):
    """
    Garante que um diretório existe
    
    Args:
        directory: Caminho do diretório
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório criado: {directory}")

# ========== FUNÇÕES DE OTIMIZAÇÃO ==========

def reduce_mem_usage(df, verbose=True):
    """
    Reduz uso de memória de um DataFrame
    
    Args:
        df: DataFrame a otimizar
        verbose: Se True, loga informações
    
    Returns:
        DataFrame otimizado
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f'Memória reduzida: {start_mem:.2f}MB → {end_mem:.2f}MB ({reduction:.1f}% redução)')
    
    return df

# ========== FUNÇÕES DE LOGGING DE MÉTRICAS ==========

def log_metrics(metrics, dataset_name=''):
    """
    Loga métricas de avaliação de forma organizada
    
    Args:
        metrics: Dict com métricas
        dataset_name: Nome do dataset (ex: 'Treino', 'Teste')
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"MÉTRICAS - {dataset_name}")
    logger.info(f"{'='*60}")
    
    # Métricas gerais
    general_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                      'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    logger.info("\nMétricas Gerais:")
    for metric in general_metrics:
        if metric in metrics:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
    
    # Métricas por classe
    class_metrics = {k: v for k, v in metrics.items() 
                    if any(x in k for x in ['Empate', 'Vitória'])}
    
    if class_metrics:
        logger.info("\nMétricas por Classe:")
        for metric, value in class_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

def log_feature_importance(model, feature_names, top_n=20):
    """
    Loga importância das features
    
    Args:
        model: Modelo treinado
        feature_names: Lista com nomes das features
        top_n: Número de features a mostrar
    
    Returns:
        DataFrame com importâncias
    """
    try:
        # Obter importâncias
        importances = model.feature_importances_
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Ordenar por importância
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Logar top N
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP {top_n} FEATURES MAIS IMPORTANTES")
        logger.info(f"{'='*60}")
        
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Erro ao calcular feature importance: {e}")
        return None

def log_training_summary(metrics, cv_scores, training_time):
    """
    Loga resumo completo do treinamento
    
    Args:
        metrics: Métricas de avaliação
        cv_scores: Scores de validação cruzada
        training_time: Tempo de treinamento
    """
    logger.info("\n" + "="*60)
    logger.info("RESUMO DO TREINAMENTO")
    logger.info("="*60)
    
    logger.info(f"\nTempo de treinamento: {training_time:.2f} segundos")
    
    logger.info(f"\nValidação Cruzada:")
    logger.info(f"  Média: {cv_scores.mean():.4f}")
    logger.info(f"  Desvio Padrão: {cv_scores.std():.4f}")
    logger.info(f"  Min: {cv_scores.min():.4f}")
    logger.info(f"  Max: {cv_scores.max():.4f}")
    
    logger.info(f"\nDesempenho no Teste:")
    logger.info(f"  Acurácia: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")

# ========== FUNÇÕES DE FORMATAÇÃO ==========

def format_prediction_output(result, home_team_name='Casa', away_team_name='Visitante'):
    """
    Formata resultado da predição para exibição
    
    Args:
        result: Dict com resultado da predição
        home_team_name: Nome do time da casa
        away_team_name: Nome do time visitante
    
    Returns:
        String formatada
    """
    output = []
    output.append("\n" + "="*60)
    output.append(f"PREDIÇÃO: {home_team_name} vs {away_team_name}")
    output.append("="*60)
    
    output.append(f"\nResultado Previsto: {result['prediction_label']}")
    output.append(f"Confiança: {result['confidence']:.2%}")
    
    output.append("\nProbabilidades:")
    output.append(f"  Empate: {result['probabilities']['empate']:.2%}")
    output.append(f"  Vitória {home_team_name}: {result['probabilities']['vitoria_casa']:.2%}")
    output.append(f"  Vitória {away_team_name}: {result['probabilities']['vitoria_visitante']:.2%}")
    
    output.append("="*60)
    
    return "\n".join(output)

def format_time(seconds):
    """
    Formata tempo em segundos para string legível
    
    Args:
        seconds: Tempo em segundos
    
    Returns:
        String formatada (ex: "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# ========== FUNÇÕES DE TIMESTAMP ==========

def get_timestamp():
    """
    Retorna timestamp atual formatado
    
    Returns:
        String com timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_separator(title='', char='=', width=60):
    """
    Loga separador visual
    
    Args:
        title: Título (opcional)
        char: Caractere do separador
        width: Largura do separador
    """
    if title:
        logger.info(f"\n{char*width}")
        logger.info(f"{title.center(width)}")
        logger.info(f"{char*width}")
    else:
        logger.info(f"\n{char*width}")

# ========== FUNÇÕES DE DEBUG ==========

def debug_dataframe(df, name="DataFrame", n_rows=5):
    """
    Exibe informações de debug de um DataFrame
    
    Args:
        df: DataFrame
        name: Nome do DataFrame
        n_rows: Número de linhas a mostrar
    """
    logger.debug(f"\n{'='*60}")
    logger.debug(f"DEBUG: {name}")
    logger.debug(f"{'='*60}")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columns: {list(df.columns)}")
    logger.debug(f"Dtypes:\n{df.dtypes}")
    logger.debug(f"\nPrimeiras {n_rows} linhas:")
    logger.debug(f"\n{df.head(n_rows)}")
    logger.debug(f"\nInfo:")
    logger.debug(df.info())

if __name__ == "__main__":
    # Testes das funções
    logger.info("=== TESTE DO MÓDULO UTILS ===\n")
    
    # Teste conversões
    logger.info("Testando conversões:")
    logger.info(f"  180 lbs = {parse_weight('180 lbs')} kg")
    logger.info(f" 6'2\" = {parse_height('6\'2\"')} m")
    logger.info(f"  BMI (80kg, 1.80m) = {calculate_bmi(80, 1.80)}")
    
    # Teste parse form
    logger.info("\nTestando parse de forma:")
    form = parse_form_string("WWDLL")
    logger.info(f"  WWDLL = {form}")
    
    # Teste safe divide
    logger.info("\nTestando divisão segura:")
    logger.info(f"  10 / 2 = {safe_divide(10, 2)}")
    logger.info(f"  10 / 0 = {safe_divide(10, 0, default=0)}")
    
    logger.info("\n✓ Testes concluídos!")