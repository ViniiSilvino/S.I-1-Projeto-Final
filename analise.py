"""
Script para verificar as colunas reais dos CSVs
"""

import pandas as pd
import os

BASE_PATH = r'C:\Users\Rafaribas\Desktop\Faculdade\Curso\6º período\SI\Projeto-Final\kaggle_data\data\base_data'

print("="*80)
print("🔍 VERIFICANDO COLUNAS DOS CSVs")
print("="*80)

# Verificar cada CSV
csvs = ['leagues.csv', 'teams.csv', 'fixtures.csv', 'standings.csv', 'teamStats.csv']

for csv_name in csvs:
    print(f"\n📄 {csv_name}")
    print("-"*80)
    try:
        df = pd.read_csv(os.path.join(BASE_PATH, csv_name), nrows=3)
        print(f"Colunas ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\nExemplo de dados (primeira linha):")
        if len(df) > 0:
            for col in df.columns[:10]:  # Mostrar apenas primeiras 10 colunas
                val = df[col].iloc[0]
                print(f"   {col}: {val}")
    except Exception as e:
        print(f"❌ Erro: {e}")

print("\n" + "="*80)