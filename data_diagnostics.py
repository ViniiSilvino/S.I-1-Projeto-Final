"""
Script de Diagnóstico - Verificar qualidade dos dados
Execute antes de treinar o modelo para identificar problemas
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_PATH = Path(__file__).parent / "data"

print("="*80)
print("🔍 DIAGNÓSTICO DE DADOS - SOCCER DATASET")
print("="*80)

# Carregar dados
print("\n📂 Carregando arquivos...")
fixtures = pd.read_csv(os.path.join(BASE_PATH, 'base_data', 'fixtures.csv'))
fixtures['date'] = pd.to_datetime(fixtures['date'])

print(f"✅ Total de partidas: {len(fixtures):,}")

# Análise de placares
print("\n" + "="*80)
print("📊 ANÁLISE DE PLACARES")
print("="*80)

print(f"\nTotal de partidas: {len(fixtures):,}")
print(f"Com homeTeamScore não nulo: {fixtures['homeTeamScore'].notna().sum():,}")
print(f"Com awayTeamScore não nulo: {fixtures['awayTeamScore'].notna().sum():,}")

# Partidas completas
completed = fixtures[
    (fixtures['homeTeamScore'].notna()) & 
    (fixtures['awayTeamScore'].notna())
].copy()

print(f"\nPartidas com ambos placares: {len(completed):,} ({len(completed)/len(fixtures)*100:.1f}%)")

# Verificar se placares são todos 0-0
print("\n📈 Análise detalhada de placares:")
print(f"   Placares 0-0: {((completed['homeTeamScore'] == 0) & (completed['awayTeamScore'] == 0)).sum():,}")
print(f"   Home > 0: {(completed['homeTeamScore'] > 0).sum():,}")
print(f"   Away > 0: {(completed['awayTeamScore'] > 0).sum():,}")

# Estatísticas de placares
print(f"\n📊 Estatísticas de gols:")
print(f"   Home Score - Média: {completed['homeTeamScore'].mean():.2f}, Máx: {completed['homeTeamScore'].max():.0f}")
print(f"   Away Score - Média: {completed['awayTeamScore'].mean():.2f}, Máx: {completed['awayTeamScore'].max():.0f}")

# Criar target
completed['result'] = completed.apply(
    lambda row: 2 if row['homeTeamScore'] > row['awayTeamScore']
    else (0 if row['homeTeamScore'] < row['awayTeamScore'] else 1),
    axis=1
)

print("\n🎯 DISTRIBUIÇÃO DE RESULTADOS:")
print("="*80)
dist = completed['result'].value_counts().sort_index()
total = len(completed)

for result, count in dist.items():
    result_name = ['Away Win', 'Draw', 'Home Win'][result]
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f"   {result_name:12s} ({result}): {count:6,} ({pct:5.1f}%) {bar}")

# Análise por período
print("\n📅 DISTRIBUIÇÃO POR PERÍODO:")
print("="*80)

completed = completed.sort_values('date')
print(f"\nPrimeira partida: {completed['date'].min()}")
print(f"Última partida: {completed['date'].max()}")

# Dividir em quartis
completed['quarter'] = pd.qcut(completed.index, q=4, labels=['Q1 (mais antigo)', 'Q2', 'Q3', 'Q4 (mais recente)'])

print("\nDistribuição de resultados por período:")
for quarter in ['Q1 (mais antigo)', 'Q2', 'Q3', 'Q4 (mais recente)']:
    quarter_data = completed[completed['quarter'] == quarter]
    if len(quarter_data) > 0:
        dist = quarter_data['result'].value_counts()
        print(f"\n{quarter}: {len(quarter_data):,} partidas")
        for result in [0, 1, 2]:
            count = dist.get(result, 0)
            pct = count / len(quarter_data) * 100 if len(quarter_data) > 0 else 0
            result_name = ['Away Win', 'Draw', 'Home Win'][result]
            print(f"   {result_name:12s}: {count:6,} ({pct:5.1f}%)")

# Últimas 10k partidas
print("\n🔍 ANÁLISE DAS ÚLTIMAS 10.000 PARTIDAS:")
print("="*80)

recent = completed.tail(10000)
if len(recent) > 0:
    dist = recent['result'].value_counts()
    print(f"\nTotal: {len(recent):,} partidas")
    print(f"Período: {recent['date'].min()} até {recent['date'].max()}")
    print("\nDistribuição:")
    for result in [0, 1, 2]:
        count = dist.get(result, 0)
        pct = count / len(recent) * 100 if len(recent) > 0 else 0
        result_name = ['Away Win', 'Draw', 'Home Win'][result]
        print(f"   {result_name:12s}: {count:6,} ({pct:5.1f}%)")

# Verificar se há problema
print("\n⚠️  VERIFICAÇÃO DE PROBLEMAS:")
print("="*80)

issues = []

# Problema 1: Poucos dados
if len(completed) < 1000:
    issues.append("❌ Poucos dados completos (< 1000 partidas)")

# Problema 2: Muito desbalanceado
dist = completed['result'].value_counts()
max_pct = dist.max() / len(completed) * 100
if max_pct > 80:
    issues.append(f"❌ Distribuição muito desbalanceada ({max_pct:.1f}% em uma classe)")

# Problema 3: Última amostra desbalanceada
recent = completed.tail(10000)
if len(recent) > 0:
    dist_recent = recent['result'].value_counts()
    max_pct_recent = dist_recent.max() / len(recent) * 100
    if max_pct_recent > 80:
        issues.append(f"❌ Últimas 10k partidas muito desbalanceadas ({max_pct_recent:.1f}%)")

# Problema 4: Muitos 0-0
zero_zero = ((completed['homeTeamScore'] == 0) & (completed['awayTeamScore'] == 0)).sum()
zero_zero_pct = zero_zero / len(completed) * 100
if zero_zero_pct > 50:
    issues.append(f"❌ Muitos jogos 0-0 ({zero_zero_pct:.1f}%)")

if issues:
    print("\n⚠️  PROBLEMAS ENCONTRADOS:")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n💡 SUGESTÕES:")
    print("   1. Verifique se a coluna de placares está correta")
    print("   2. Não use sample_size pequeno (use mais dados)")
    print("   3. Remova partidas futuras/agendadas sem placar")
    print("   4. Considere filtrar apenas ligas principais")
else:
    print("\n✅ Dados parecem estar OK para treinamento!")

# Exemplo de placares
print("\n📋 AMOSTRA DE PLACARES (10 partidas aleatórias):")
print("="*80)
sample = completed.sample(min(10, len(completed)))
for _, match in sample.iterrows():
    result_name = ['Away Win', 'Draw', 'Home Win'][match['result']]
    print(f"   {match['date'].date()} | {match['homeTeamScore']:.0f}-{match['awayTeamScore']:.0f} | {result_name}")

print("\n" + "="*80)
print("✅ Diagnóstico completo!")
print("="*80)