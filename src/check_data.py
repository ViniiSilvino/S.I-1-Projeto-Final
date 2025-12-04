"""
Script para verificar a integridade e estrutura dos dados
"""
import os
import sys
import pandas as pd
from config import DATA_PATHS, BASE_FILES

def check_file_exists(filepath, filename):
    """Verifica se um arquivo existe"""
    if os.path.exists(filepath):
        print(f"  ✓ {filename} encontrado")
        return True
    else:
        print(f"  ✗ {filename} NÃO encontrado")
        return False

def check_csv_structure(filepath, filename, required_cols):
    """Verifica a estrutura de um CSV"""
    try:
        df = pd.read_csv(filepath, nrows=5)
        
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            print(f"    ⚠️  Colunas faltando: {missing_cols}")
            return False
        else:
            print(f"    ✓ Todas as colunas necessárias presentes")
            print(f"    📊 Linhas totais: {len(pd.read_csv(filepath)):,}")
            return True
    except Exception as e:
        print(f"    ✗ Erro ao ler arquivo: {e}")
        return False

def main():
    """Função principal de verificação"""
    print("\n" + "="*70)
    print(" "*20 + "VERIFICAÇÃO DE DADOS")
    print("="*70)
    
    all_ok = True
    
    # 1. Verificar diretórios
    print("\n📁 Verificando estrutura de diretórios...")
    for name, path in DATA_PATHS.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} NÃO EXISTE")
            all_ok = False
    
    # 2. Verificar arquivos base
    print("\n📄 Verificando arquivos base...")
    base_path = DATA_PATHS['base']
    
    required_columns = {
        'fixtures': ['eventId', 'date', 'homeTeamId', 'awayTeamId', 'statusId'],
        'standings': ['teamId', 'leagueId', 'points', 'gamesPlayed', 'form'],
        'teamStats': ['eventId', 'teamId', 'possessionPct'],
        'players': ['athleteId', 'displayName'],
        'teams': ['teamId', 'displayName'],
        'leagues': ['leagueId', 'name'],
        'status': ['statusId', 'description']
    }
    
    for key, filename in BASE_FILES.items():
        print(f"\n{filename}:")
        filepath = os.path.join(base_path, filename)
        
        if check_file_exists(filepath, filename):
            if key in required_columns:
                check_csv_structure(filepath, filename, required_columns[key])
        else:
            all_ok = False
    
    # 3. Verificar pastas opcionais
    print("\n📂 Verificando pastas opcionais...")
    
    optional_dirs = {
        'lineup': DATA_PATHS['lineup'],
        'playerStats': DATA_PATHS['playerStats'],
        'keyEvents': DATA_PATHS['keyEvents']
    }
    
    for name, path in optional_dirs.items():
        if os.path.exists(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if csv_files:
                print(f"  ✓ {name}: {len(csv_files)} arquivo(s) CSV encontrado(s)")
            else:
                print(f"  ⚠️  {name}: pasta existe mas não contém CSVs")
        else:
            print(f"  ⚠️  {name}: pasta não existe")
    
    # 4. Resumo
    print("\n" + "="*70)
    if all_ok:
        print(" "*25 + "✅ TUDO OK!")
        print("="*70)
        print("\n👍 Seus dados estão prontos!")
        print("Execute: python main.py --mode train")
        return 0
    else:
        print(" "*20 + "⚠️  PROBLEMAS ENCONTRADOS")
        print("="*70)
        print("\n❌ Alguns arquivos estão faltando ou com problemas.")
        print("Verifique a estrutura dos dados em 'estrutura_data.docx'")
        return 1

if __name__ == "__main__":
    sys.exit(main())