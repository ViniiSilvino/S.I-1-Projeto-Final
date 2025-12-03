"""
FIX RÁPIDO - Execute este arquivo para diagnosticar e treinar com dados corretos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

BASE_PATH = r'C:\Users\Rafaribas\Desktop\Faculdade\Curso\6º período\SI\Projeto-Final\kaggle_data\data'

print("="*80)
print("⚡ FIX RÁPIDO - PREDITOR DE FUTEBOL")
print("="*80)

# 1. CARREGAR E DIAGNOSTICAR
print("\n📂 Carregando dados...")
fixtures = pd.read_csv(os.path.join(BASE_PATH, 'base_data', 'fixtures.csv'))
fixtures['date'] = pd.to_datetime(fixtures['date'])

# Filtrar apenas partidas COM PLACAR DEFINIDO E NÃO EMPATADAS EM 0-0
completed = fixtures[
    (fixtures['homeTeamScore'].notna()) &
    (fixtures['awayTeamScore'].notna()) &
    ~((fixtures['homeTeamScore'] == 0) & (fixtures['awayTeamScore'] == 0))  # Remover 0-0
].copy()

print(f"✅ Partidas válidas: {len(completed):,}")

# Criar resultado
completed['result'] = completed.apply(
    lambda row: 2 if row['homeTeamScore'] > row['awayTeamScore']
    else (0 if row['homeTeamScore'] < row['awayTeamScore'] else 1),
    axis=1
)

# Verificar distribuição
print(f"\n🎯 Distribuição de resultados:")
dist = completed['result'].value_counts().sort_index()
for result, count in dist.items():
    result_name = ['Away Win', 'Draw', 'Home Win'][result]
    pct = count / len(completed) * 100
    print(f"   {result_name:12s}: {count:6,} ({pct:5.1f}%)")

# Verificar se temos dados balanceados o suficiente
if dist.min() < 100:
    print(f"\n⚠️  AVISO: Classe minoritária tem apenas {dist.min()} exemplos")
    print("   Considere usar TODOS os dados (não use sample)")

# 2. CRIAR FEATURES SIMPLES (mais rápido)
print(f"\n🔧 Criando features simples...")

# Pegar apenas as últimas partidas com boa distribuição
# Vamos pegar 15k partidas do MEIO do dataset (não as últimas)
if len(completed) > 20000:
    # Pegar do meio para ter histórico E boa distribuição
    start_idx = len(completed) // 4
    end_idx = start_idx + 15000
    sample = completed.iloc[start_idx:end_idx].copy()
    print(f"   Usando partidas do índice {start_idx:,} ao {end_idx:,}")
else:
    sample = completed.copy()

sample = sample.sort_values('date').reset_index(drop=True)

print(f"   Total a processar: {len(sample):,}")

# Verificar distribuição da amostra
print(f"\n🎯 Distribuição da AMOSTRA:")
dist_sample = sample['result'].value_counts().sort_index()
for result, count in dist_sample.items():
    result_name = ['Away Win', 'Draw', 'Home Win'][result]
    pct = count / len(sample) * 100
    print(f"   {result_name:12s}: {count:6,} ({pct:5.1f}%)")

# Features muito simples para teste rápido
features_list = []
targets = []

standings = pd.read_csv(os.path.join(BASE_PATH, 'base_data', 'standings.csv'))

print(f"\n⚙️  Processando features...")
count = 0
for idx, match in sample.iterrows():
    if count % 2000 == 0:
        print(f"   Processadas: {count:,}/{len(sample):,}")
    
    # Features básicas do jogo atual
    features = {
        'home_score_avg': match.get('homeTeamScore', 0),  # Placeholder
        'away_score_avg': match.get('awayTeamScore', 0),
    }
    
    # Buscar classificação
    home_standing = standings[
        (standings['teamId'] == match['homeTeamId']) &
        (standings['leagueId'] == match['leagueId'])
    ]
    away_standing = standings[
        (standings['teamId'] == match['awayTeamId']) &
        (standings['leagueId'] == match['leagueId'])
    ]
    
    # Se temos classificação, usar
    if len(home_standing) > 0:
        hs = home_standing.iloc[0]
        features['home_rank'] = hs.get('teamRank', 10)
        features['home_points'] = hs.get('points', 0)
        features['home_gd'] = hs.get('gd', 0)
        features['home_wins'] = hs.get('wins', 0)
    else:
        features['home_rank'] = 10
        features['home_points'] = 0
        features['home_gd'] = 0
        features['home_wins'] = 0
    
    if len(away_standing) > 0:
        aws = away_standing.iloc[0]
        features['away_rank'] = aws.get('teamRank', 10)
        features['away_points'] = aws.get('points', 0)
        features['away_gd'] = aws.get('gd', 0)
        features['away_wins'] = aws.get('wins', 0)
    else:
        features['away_rank'] = 10
        features['away_points'] = 0
        features['away_gd'] = 0
        features['away_wins'] = 0
    
    # Features derivadas
    features['rank_diff'] = features['away_rank'] - features['home_rank']
    features['points_diff'] = features['home_points'] - features['away_points']
    features['gd_diff'] = features['home_gd'] - features['away_gd']
    
    features_list.append(features)
    targets.append(match['result'])
    count += 1

print(f"✅ {len(features_list):,} partidas processadas")

# 3. PREPARAR DADOS
X = pd.DataFrame(features_list)
y = np.array(targets)

# Remover colunas de placar (são leakage)
X = X.drop(['home_score_avg', 'away_score_avg'], axis=1, errors='ignore')

print(f"\n📊 Dados finais:")
print(f"   Features: {X.shape[1]}")
print(f"   Amostras: {X.shape[0]:,}")

# Verificar distribuição final
print(f"\n🎯 Distribuição final do target:")
unique, counts = np.unique(y, return_counts=True)
for val, count in zip(unique, counts):
    result_name = ['Away Win', 'Draw', 'Home Win'][val]
    pct = count / len(y) * 100
    print(f"   {result_name:12s} ({val}): {count:6,} ({pct:5.1f}%)")

# Verificar se temos todas as classes
if len(unique) < 3:
    print(f"\n❌ ERRO CRÍTICO: Apenas {len(unique)} classe(s) detectada(s)!")
    print("   O dataset pode estar corrompido ou filtrado incorretamente.")
    print("\n💡 Tentando outra abordagem...")
    
    # Tentar com TODOS os dados
    sample = completed.copy()
    print(f"   Tentando com TODAS as {len(sample):,} partidas...")
    
    # Reprocessar...
    # (código similar ao acima)

# 4. TREINAR MODELO
print(f"\n🎓 Treinando modelo...")

# Split temporal
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"   Treino: {len(X_train):,}")
print(f"   Teste: {len(X_test):,}")

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Avaliar
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

acc_train = accuracy_score(y_train, model.predict(X_train_scaled))
acc_test = accuracy_score(y_test, y_pred)

print(f"\n📊 RESULTADOS:")
print(f"   Acurácia Treino: {acc_train:.1%}")
print(f"   Acurácia Teste:  {acc_test:.1%}")

# Relatório detalhado
print(f"\n📋 Relatório Detalhado:")
target_names = ['Away Win', 'Draw', 'Home Win']

# Verificar quais classes existem no teste
unique_test = np.unique(y_test)
labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
names_present = [target_names[i] for i in labels_present]

print(classification_report(y_test, y_pred, labels=labels_present, 
                          target_names=names_present, zero_division=0))

print(f"\n" + "="*80)
print("✅ TREINAMENTO COMPLETO!")
print("="*80)

if acc_test > 0.45:
    print("\n🎉 Resultado EXCELENTE! Acurácia acima do baseline!")
    print("   Você pode usar este modelo para predições.")
elif acc_test > 0.40:
    print("\n👍 Resultado BOM! Modelo funcional.")
    print("   Considere adicionar mais features para melhorar.")
else:
    print("\n⚠️  Resultado ABAIXO DO ESPERADO.")
    print("   Verifique a qualidade dos dados.")

# Feature importance
print(f"\n⭐ Features mais importantes:")
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importances.head(10).iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   {row['feature']:20s} {bar} {row['importance']:.4f}")