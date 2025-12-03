import pandas as pd
import os

path = r"C:\Users\Rafaribas\Desktop\Faculdade\Curso\6º período\SI\Projeto-Final\kaggle_data\data"
leagues = pd.read_csv(os.path.join(path, 'base_data', 'leagues.csv'))

print("Colunas em leagues:")
for col in leagues.columns:
    print(f"  - {col}")

print(f"\nPrimeiras linhas:")
print(leagues.head())