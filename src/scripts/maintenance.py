"""
Scripts de manutenção e utilitários para o projeto
"""
import os
import sys
import shutil
import argparse
from datetime import datetime
import pandas as pd

# Adicionar src ao path
sys.path.append('../src')
from config import MODELS_DIR, LOGS_DIR, DATA_DIR

def clean_logs():
    """Limpa arquivos de log antigos"""
    print("🧹 Limpando logs...")
    
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.txt') or f.endswith('.log')]
    
    if not log_files:
        print("  Nenhum log encontrado.")
        return
    
    for log_file in log_files:
        filepath = os.path.join(LOGS_DIR, log_file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"  • {log_file} ({size_mb:.2f} MB)")
        
        response = input("    Deletar? (s/n): ")
        if response.lower() == 's':
            os.remove(filepath)
            print("    ✓ Deletado")
        else:
            print("    ✗ Mantido")

def backup_models():
    """Faz backup dos modelos treinados"""
    print("\n💾 Fazendo backup de modelos...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join('backups', f'models_{timestamp}')
    
    if not os.path.exists(MODELS_DIR):
        print("  ✗ Diretório de modelos não existe")
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    
    model_files = [f for f in os.listdir(MODELS_DIR) 
                   if f.endswith(('.json', '.pkl', '.h5', '.pt'))]
    
    if not model_files:
        print("  ✗ Nenhum modelo encontrado")
        return
    
    for model_file in model_files:
        src = os.path.join(MODELS_DIR, model_file)
        dst = os.path.join(backup_dir, model_file)
        shutil.copy2(src, dst)
        print(f"  ✓ {model_file} copiado")
    
    print(f"\n✓ Backup salvo em: {backup_dir}")

def compress_data():
    """Comprime arquivos CSV para economizar espaço"""
    print("\n🗜️  Comprimindo dados...")
    
    total_saved = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                
                # Ler CSV
                df = pd.read_csv(filepath)
                
                # Tamanho original
                original_size = os.path.getsize(filepath)
                
                # Salvar comprimido
                compressed_path = filepath.replace('.csv', '.csv.gz')
                df.to_csv(compressed_path, compression='gzip', index=False)
                
                # Tamanho comprimido
                compressed_size = os.path.getsize(compressed_path)
                
                # Economia
                saved = original_size - compressed_size
                saved_pct = (saved / original_size) * 100
                
                print(f"  • {file}:")
                print(f"    Original: {original_size / 1024:.1f} KB")
                print(f"    Comprimido: {compressed_size / 1024:.1f} KB")
                print(f"    Economia: {saved_pct:.1f}%")
                
                total_saved += saved
                
                # Perguntar se quer deletar original
                response = input("    Deletar original? (s/n): ")
                if response.lower() == 's':
                    os.remove(filepath)
                    print("    ✓ Original deletado")
                else:
                    os.remove(compressed_path)
                    print("    ✗ Compressão revertida")
    
    print(f"\n✓ Total economizado: {total_saved / (1024*1024):.1f} MB")

def check_data_quality():
    """Verifica qualidade dos dados"""
    print("\n🔍 Verificando qualidade dos dados...")
    
    from etl import load_and_preprocess_data
    
    try:
        data = load_and_preprocess_data()
        
        print("\n📊 Relatório de Qualidade:")
        
        for name, df in data.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                print(f"\n{name.upper()}:")
                print(f"  • Linhas: {len(df):,}")
                print(f"  • Colunas: {len(df.columns)}")
                
                # Valores faltantes
                null_cols = df.isnull().sum()
                null_cols = null_cols[null_cols > 0]
                
                if len(null_cols) > 0:
                    print(f"  • Colunas com valores faltantes:")
                    for col, count in null_cols.items():
                        pct = (count / len(df)) * 100
                        print(f"    - {col}: {count} ({pct:.1f}%)")
                else:
                    print(f"  • ✓ Sem valores faltantes")
                
                # Duplicatas
                dups = df.duplicated().sum()
                if dups > 0:
                    print(f"  • ⚠️  {dups} linhas duplicadas")
                else:
                    print(f"  • ✓ Sem duplicatas")
        
        print("\n✓ Verificação concluída")
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")

def update_features():
    """Recria features com novos dados"""
    print("\n🔧 Atualizando features...")
    
    from etl import load_and_preprocess_data
    from feature_engineering import create_features
    
    try:
        # Carregar dados
        print("  1. Carregando dados...")
        data = load_and_preprocess_data()
        
        # Criar features
        print("  2. Criando features...")
        master_df = create_features(data)
        
        # Salvar
        print("  3. Salvando...")
        cache_dir = 'cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(cache_dir, f'features_{timestamp}.csv')
        
        master_df.to_csv(filepath, index=False)
        
        print(f"\n✓ Features salvas em: {filepath}")
        print(f"  Shape: {master_df.shape}")
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")

def generate_report():
    """Gera relatório completo do sistema"""
    print("\n📄 Gerando relatório...")
    
    report = []
    report.append("="*70)
    report.append("RELATÓRIO DO SISTEMA - Soccer Match Prediction")
    report.append("="*70)
    report.append(f"\nData/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Modelos
    report.append("\n--- MODELOS ---")
    if os.path.exists(MODELS_DIR):
        model_files = os.listdir(MODELS_DIR)
        if model_files:
            for f in model_files:
                filepath = os.path.join(MODELS_DIR, f)
                size = os.path.getsize(filepath) / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                report.append(f"  • {f} ({size:.1f} KB) - Modificado: {mtime}")
        else:
            report.append("  Nenhum modelo encontrado")
    
    # Logs
    report.append("\n--- LOGS ---")
    if os.path.exists(LOGS_DIR):
        log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.txt') or f.endswith('.log')]
        if log_files:
            for f in log_files:
                filepath = os.path.join(LOGS_DIR, f)
                size = os.path.getsize(filepath) / 1024
                report.append(f"  • {f} ({size:.1f} KB)")
        else:
            report.append("  Nenhum log encontrado")
    
    # Dados
    report.append("\n--- DADOS ---")
    for name, path in [('Base Data', os.path.join(DATA_DIR, 'base_data'))]:
        if os.path.exists(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            report.append(f"  {name}: {len(csv_files)} arquivo(s) CSV")
    
    report.append("\n" + "="*70)
    
    # Imprimir relatório
    report_text = "\n".join(report)
    print(report_text)
    
    # Salvar relatório
    report_file = os.path.join(LOGS_DIR, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Relatório salvo em: {report_file}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Scripts de manutenção')
    parser.add_argument(
        'action',
        choices=['clean-logs', 'backup-models', 'compress-data', 
                'check-quality', 'update-features', 'report'],
        help='Ação a executar'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*25 + "MANUTENÇÃO")
    print("="*70)
    
    if args.action == 'clean-logs':
        clean_logs()
    elif args.action == 'backup-models':
        backup_models()
    elif args.action == 'compress-data':
        compress_data()
    elif args.action == 'check-quality':
        check_data_quality()
    elif args.action == 'update-features':
        update_features()
    elif args.action == 'report':
        generate_report()
    
    print("\n✓ Manutenção concluída!")

if __name__ == "__main__":
    main()