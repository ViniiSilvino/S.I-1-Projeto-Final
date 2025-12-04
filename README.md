# ⚽ Soccer Match Prediction System

Sistema completo de Machine Learning para predição de resultados de partidas de futebol usando XGBoost.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Pipeline de Dados](#-pipeline-de-dados)
- [Features Criadas](#-features-criadas)
- [Modelo](#-modelo)
- [Resultados](#-resultados)
- [Troubleshooting](#-troubleshooting)

## 🎯 Visão Geral

Este projeto implementa um sistema de predição de resultados de partidas de futebol (Vitória Casa, Empate, Vitória Visitante) utilizando:

- **Dados**: 30.000+ partidas de 400+ ligas da temporada 2024-2025
- **Fonte**: ESPN Soccer Data API
- **Modelo**: XGBoost Classifier
- **Features**: 30+ variáveis preditivas (forma recente, performance, estatísticas de jogo, escalação)

### Principais Características

✅ Pipeline completo e automatizado de ML  
✅ Feature Engineering avançado  
✅ Validação cruzada estratificada  
✅ Normalização de features  
✅ Explicabilidade das predições  
✅ Interface de linha de comando intuitiva  

## 📁 Estrutura do Projeto

```
S.I-1-Projeto-Final/
│
├── data/                          # Dados brutos
│   ├── base_data/                 # Dados principais
│   │   ├── fixtures.csv           # Partidas
│   │   ├── standings.csv          # Classificações
│   │   ├── teamStats.csv          # Estatísticas dos times
│   │   ├── players.csv            # Jogadores
│   │   └── ...
│   ├── lineup_data/               # Escalações
│   ├── playerStats_data/          # Estatísticas de jogadores
│   └── ...
│
├── src/                           # Código-fonte
│   ├── config.py                  # Configurações
│   ├── utils.py                   # Funções auxiliares
│   ├── etl.py                     # ETL
│   ├── feature_engineering.py     # Criação de features
│   ├── model_xgboost.py          # Treinamento
│   └── predict.py                 # Predições
│
├── models/                        # Modelos treinados
│   ├── best_model.json           # Modelo XGBoost
│   ├── scaler.pkl                # Normalizador
│   └── feature_columns.json      # Features usadas
│
├── logs/                          # Logs de execução
│   └── training_log.txt
│
├── notebooks/                     # Análises exploratórias
│   ├── EDA.ipynb
│   └── Debug_Features.ipynb
│
├── main.py                        # Pipeline principal
└── README.md                      # Este arquivo
```

## 🚀 Instalação

### Requisitos

- Python 3.8+
- pip

### Dependências

```bash
pip install pandas numpy xgboost scikit-learn
```

Ou crie um arquivo `requirements.txt`:

```
pandas>=1.5.0
numpy>=1.23.0
xgboost>=2.0.0
scikit-learn>=1.2.0
```

E instale:

```bash
pip install -r requirements.txt
```

## 💻 Como Usar

### 1. Treinar o Modelo

Execute o pipeline completo (ETL → Features → Treinamento):

```bash
python main.py --mode train
```

Ou simplesmente:

```bash
python main.py
```

**Saída esperada:**
- Modelo treinado salvo em `models/best_model.json`
- Scaler salvo em `models/scaler.pkl`
- Features salvas em `models/feature_columns.json`
- Logs em `logs/training_log.txt`

### 2. Fazer Predições

Modo interativo para predizer resultados:

```bash
python main.py --mode predict
```

Você será solicitado a informar:
- Estatísticas do time da casa
- Estatísticas do time visitante

**Exemplo de interação:**

```
--- Time da Casa ---
Vitórias recentes (últimos 5 jogos): 3
Empates recentes: 1
Derrotas recentes: 1
Média de gols por jogo: 1.8
Média de gols sofridos por jogo: 1.0
Pontos na tabela: 45

--- Time Visitante ---
Vitórias recentes (últimos 5 jogos): 2
Empates recentes: 2
Derrotas recentes: 1
Média de gols por jogo: 1.5
Média de gols sofridos por jogo: 1.2
Pontos na tabela: 38

🎯 RESULTADO DA PREDIÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Resultado Previsto: Vitória Casa
📊 Confiança: 65.3%

📈 Probabilidades:
   • Empate: 18.2%
   • Vitória Casa: 65.3%
   • Vitória Visitante: 16.5%
```

### 3. Avaliar o Modelo

Avalia o desempenho do modelo em todos os dados:

```bash
python main.py --mode evaluate
```

## 🔄 Pipeline de Dados

### Etapa 1: ETL (etl.py)

**Processo:**
1. Carrega todos os arquivos CSV
2. Converte datas para datetime
3. Converte unidades (lbs → kg, ft'in" → metros)
4. Calcula BMI dos jogadores
5. Trata valores faltantes
6. Filtra apenas partidas completas

**Saída:** Dicionário com DataFrames processados

### Etapa 2: Feature Engineering (feature_engineering.py)

**Processo:**
1. Inicializa master_df com fixtures
2. Cria variável target (0=Empate, 1=Casa, 2=Visitante)
3. Adiciona features de forma recente
4. Adiciona features de performance
5. Adiciona estatísticas de jogo
6. Adiciona qualidade da escalação
7. Cria features derivadas

**Saída:** DataFrame com 30+ features

### Etapa 3: Treinamento (model_xgboost.py)

**Processo:**
1. Separa treino/teste (80/20 estratificado)
2. Normaliza features (StandardScaler)
3. Realiza validação cruzada
4. Treina XGBoost
5. Avalia métricas
6. Salva modelo

**Saída:** Modelo treinado + métricas

## 📊 Features Criadas

### Forma Recente (últimos 5 jogos)
- `home_recent_wins` / `away_recent_wins`
- `home_recent_draws` / `away_recent_draws`
- `home_recent_losses` / `away_recent_losses`
- `home_form_points` / `away_form_points`

### Performance Geral
- `home_goals_per_game` / `away_goals_per_game`
- `home_goals_against_per_game` / `away_goals_against_per_game`
- `home_goal_difference` / `away_goal_difference`
- `home_points` / `away_points`
- `home_wins` / `away_wins`
- `home_draws` / `away_draws`
- `home_losses` / `away_losses`

### Estatísticas de Jogo
- `home_possession_avg` / `away_possession_avg`
- `home_pass_accuracy` / `away_pass_accuracy`
- `home_shot_accuracy` / `away_shot_accuracy`

### Qualidade da Escalação
- `home_avg_age` / `away_avg_age`
- `home_avg_height` / `away_avg_height`
- `home_avg_weight` / `away_avg_weight`

### Features Derivadas
- `points_difference`: Diferença de pontos na tabela
- `form_difference`: Diferença de forma recente
- `attack_difference`: Diferença ofensiva
- `defense_difference`: Diferença defensiva

## 🤖 Modelo

### XGBoost Classifier

**Hiperparâmetros:**
```python
{
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### Processo de Treinamento

1. **Split**: 80% treino, 20% teste (estratificado)
2. **Normalização**: StandardScaler
3. **Validação Cruzada**: 5-fold estratificado
4. **Early Stopping**: 50 rounds
5. **Métrica**: Multi-class Log Loss

## 📈 Resultados

### Métricas Esperadas

| Métrica | Valor Típico |
|---------|--------------|
| Acurácia | 50-60% |
| Precision (macro) | 45-55% |
| Recall (macro) | 45-55% |
| F1-Score (macro) | 45-55% |

**Nota:** Futebol é inerentemente difícil de prever. Acurácias de 50-60% são consideradas boas no domínio.

### Distribuição Típica de Classes

- **Vitória Casa**: ~45%
- **Empate**: ~27%
- **Vitória Visitante**: ~28%

### Feature Importance

As features mais importantes geralmente são:
1. `points_difference`
2. `home_form_points`
3. `home_goals_per_game`
4. `away_goals_per_game`
5. `form_difference`

## 🔧 Troubleshooting

### Erro: "Arquivo não encontrado"

**Solução:** Verifique se os arquivos CSV estão nas pastas corretas:
```
data/base_data/fixtures.csv
data/base_data/standings.csv
etc.
```

### Erro: "Colunas faltando"

**Solução:** Certifique-se de que os CSVs têm todas as colunas necessárias. Veja `estrutura_data.docx`.

### Modelo com baixa acurácia

**Possíveis causas:**
1. Dados insuficientes
2. Features não representativas
3. Hiperparâmetros não otimizados

**Soluções:**
1. Adicione mais dados históricos
2. Crie novas features (form home/away separado)
3. Faça hyperparameter tuning

### Memória insuficiente

**Solução:** Use a função `reduce_mem_usage()` em `utils.py`:

```python
from utils import reduce_mem_usage
df = reduce_mem_usage(df)
```

## 📝 Logs

Todos os logs são salvos em `logs/training_log.txt` com informações sobre:
- Carregamento de dados
- Pré-processamento
- Feature engineering
- Treinamento
- Métricas de avaliação
- Feature importance

## 🤝 Contribuindo

Para adicionar novas features:

1. Edite `feature_engineering.py`
2. Adicione a nova feature em `FEATURE_GROUPS` no `config.py`
3. Atualize `ALL_FEATURES`
4. Retreine o modelo

## 📄 Licença

Este projeto é para fins educacionais.

## 👥 Autores

Sistema de Inteligência - Projeto Final

---

**Última atualização:** Dezembro 2024