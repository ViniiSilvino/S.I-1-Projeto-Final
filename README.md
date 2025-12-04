# ⚽ Soccer Match Prediction System

Sistema completo de Machine Learning para predição de resultados de partidas de futebol usando XGBoost.

## 📋 Índice

- Visão Geral
- Estrutura do Projeto
- Instalação
- Como Usar
- Pipeline de Dados
- Features Criadas
- Modelo
- Resultados
- Melhorias Futuras

## 🎯 Visão Geral

Este projeto implementa um sistema de predição de resultados de partidas de futebol (Vitória Casa, Empate, Vitória Visitante) utilizando:

- **Dados**: ~55.000 partidas processadas (de um total de 67.000+) de diversas ligas.
- **Fonte**: Dados históricos incluindo *fixtures*, *standings* e *team stats*.
- **Modelo**: XGBoost Classifier (Otimizado com Optuna).
- **Features**: 30+ variáveis preditivas focadas em forma recente, estatísticas de jogo e qualidade de escalação.


## 📂 Estrutura do Projeto

```

S.I-1-Projeto-Final/
│
├── data/                      \# Dados brutos e processados (ignorados pelo git)
├── logs/                      \# Logs de execução e histórico de treinamento
│   └── training\_log.txt       \# Log detalhado do último treino
│
├── models/                    \# Artefatos gerados pelo modelo
│   ├── best\_model.json        \# Modelo XGBoost treinado
│   ├── scaler.pkl             \# Objeto de normalização (StandardScaler)
│   ├── feature\_columns.json   \# Lista oficial de features usadas
│   └── draw\_threshold.json    \# Limiar otimizado para predição de empates
│
├── notebooks/                 \# Análises exploratórias e testes
│   └── EDA.ipynb              \# Notebook de Análise Exploratória de Dados
│
├── src/                       \# Código-fonte do sistema
│   ├── config.py              \# Configurações globais e hiperparâmetros
│   ├── utils.py               \# Funções utilitárias (logs, memória)
│   ├── etl.py                 \# Pipeline de Extração e Limpeza
│   ├── feature\_engineering.py \# Criação e transformação de variáveis
│   ├── model\_xgboost.py       \# Lógica de treinamento e avaliação
│   ├── hyperparameter\_tuning.py \# Otimização com Optuna
│   └── predict.py             \# Motor de inferência para novos jogos
│
├── main.py                    \# Arquivo principal (Ponto de entrada)
├── requirements.txt           \# Lista de dependências do projeto
└── README.md                  \# Documentação oficial

```

## 🚀 Instalação

### Requisitos

- Python 3.8+
- pip

### Instalação das Dependências
```
pip install -r requirements.txt
```
## Como usar
### Treinar modelo
```
python main.py --mode train
```
### Otimizar hiperparametros
```
python main.py --mode tune
```
### Avaliar performance
```
python main.py --mode evaluate
```
### Fazer predições 
```
python main.py --mode predict
```
## 🔄 Pipeline de Dados

O sistema segue um fluxo linear de processamento:

1.  **ETL (`src/etl.py`):**
    * Carregamento de CSVs brutos (Fixtures, Players, TeamStats, Standings).
    * Conversão de unidades imperiais (lbs/ft) para métricas (kg/m).
    * Imputação de valores nulos utilizando a mediana da liga/time.
    * Filtragem de partidas canceladas ou sem placar.

2.  **Feature Engineering (`src/feature_engineering.py`):**
    * Criação de janelas temporais (ex: aproveitamento nos últimos 5 jogos).
    * Cálculo de métricas diferenciais (`home_stat` - `away_stat`).
    * Agregação de estatísticas físicas da escalação titular (idade, altura, peso).

3.  **Pré-processamento (`src/model_xgboost.py`):**
    * **Normalização:** Aplicação de `StandardScaler` nas features numéricas.
    * **Balanceamento:** Uso de **SMOTE** (para criar exemplos sintéticos de Empates/Visitantes) combinado com **RandomUnderSampler** (para reduzir a classe majoritária Casa).

## 📊 Features Criadas

O modelo utiliza mais de 30 variáveis explicativas divididas em grupos:

* **Forma Recente (5 jogos):** `home_recent_wins`, `away_recent_losses`, `form_points`.
* **Performance Geral:** `goals_per_game`, `goal_difference`, `points_table`.
* **Estatísticas de Jogo:** `possession_avg`, `pass_accuracy`, `shot_accuracy`.
* **Qualidade da Escalação:** `avg_age` (experiência), `avg_height` (bola aérea), `avg_weight` (força).
* **Features Derivadas (As mais importantes):**
    * `points_difference`: Diferença de pontuação na tabela.
    * `form_difference`: Comparação de momento vivido pelas equipes.
    * `defense_difference`: Solidez defensiva relativa.

## 🤖 Modelo

### XGBoost Classifier (Configuração Otimizada)

O modelo final foi ajustado via **Optuna** para maximizar o F1-Score Macro. Os hiperparâmetros resultantes foram:

```python
{
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 7,            # Profundidade da árvore (captura complexidade)
    'learning_rate': 0.05,     # Taxa de aprendizado (mais lento e preciso)
    'n_estimators': 300,       # Número de árvores de decisão
    'subsample': 0.8,          # Amostragem de linhas por árvore
    'colsample_bytree': 0.8,   # Amostragem de colunas por árvore
    'gamma': 0.1,              # Redução mínima de perda para divisão
    'min_child_weight': 3,     # Peso mínimo para criar um nó filho
    'eval_metric': 'mlogloss'
}
```
## 🔮 4. Melhorias Futuras
Para evoluir o projeto e reduzir os conflitos entre Vitória Casa e Empate identificados na Matriz de Confusão, propõe-se:

### Mercado "Chance Dupla" (Double Chance):

Alterar o alvo do modelo para prever classes binárias: "Vitória ou Empate" vs "Derrota". Isso aumenta drasticamente a assertividade para estratégias de aversão ao risco.

### Mercado "Empate Anula" (Draw No Bet):

Treinar um modelo que ignora o empate como target, focando puramente na superioridade técnica. Se a probabilidade de empate for alta, o sistema sugere não apostar.