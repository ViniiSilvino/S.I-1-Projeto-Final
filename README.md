# ⚽ Sistema de Predição de Resultados de Futebol

Sistema intermediário de Machine Learning para prever resultados de partidas de futebol (Vitória Casa / Empate / Vitória Visitante) usando Random Forest e Feature Engineering avançado.

---

## 📋 Índice

1. [Sobre o Projeto](#sobre-o-projeto)
2. [Instalação](#instalação)
3. [Estrutura dos Arquivos](#estrutura-dos-arquivos)
4. [Como Usar](#como-usar)
5. [Features do Modelo](#features-do-modelo)
6. [Resultados Esperados](#resultados-esperados)
7. [Exemplos](#exemplos)

---

## 🎯 Sobre o Projeto

Este sistema utiliza dados históricos de mais de 30.000 partidas de futebol da temporada 2024-2025 para prever o resultado de futuras partidas.

### Características:

- **Target**: 3 classes (Home Win / Draw / Away Win)
- **Algoritmo**: Random Forest com 200 árvores
- **Features**: 40+ features engineered
- **Validação**: Time Series Split (validação temporal)
- **Acurácia esperada**: 50-55% (melhor que baseline ~33%)

---

## 🔧 Instalação

### 1. Pré-requisitos

```bash
Python 3.8+
```

### 2. Instalar dependências

```bash
pip install pandas numpy scikit-learn
```

### 3. Estrutura de pastas esperada

```
seu_projeto/
│
├── kaggle_data/
│   └── data/
│       ├── base_data/
│       │   ├── fixtures.csv
│       │   ├── standings.csv
│       │   ├── teamStats.csv
│       │   ├── teams.csv
│       │   └── leagues.csv
│       │
│       ├── commentary_data/
│       ├── keyEvents_data/
│       └── ...
│
├── soccer_predictor.py
├── data_explorer.py
└── main_runner.py
```

---

## 📁 Estrutura dos Arquivos

### 1. `soccer_predictor.py`
Classe principal do preditor com:
- Carregamento de dados
- Feature engineering
- Treinamento do modelo
- Predições

### 2. `data_explorer.py`
Ferramenta para explorar o dataset:
- Buscar times e ligas
- Ver classificações
- Análise de estatísticas
- Encontrar próximas partidas

### 3. `main_runner.py`
Script de execução com menu interativo

---

## 🚀 Como Usar

### Opção 1: Menu Interativo (Recomendado)

```bash
python main_runner.py
```

Você verá um menu com opções:
1. **Explorar dados** - Conhecer times, ligas e IDs
2. **Treinar modelo** - Treinar o algoritmo preditivo
3. **Fazer predições** - Prever resultados de partidas
4. **Pipeline completo** - Treinar e prever

### Opção 2: Linha de Comando

```bash
# Explorar dados
python main_runner.py --explore

# Treinar modelo
python main_runner.py --train

# Fazer predições
python main_runner.py --predict
```

### Opção 3: Uso Programático

```python
from soccer_predictor import SoccerMatchPredictor

# Configurar caminho
BASE_PATH = r'C:\caminho\para\seus\dados'

# Inicializar
predictor = SoccerMatchPredictor(BASE_PATH)

# Carregar dados
predictor.load_data()

# Criar features
predictor.engineer_features(sample_size=10000)

# Treinar
results = predictor.train_model()

# Prever partida
predictor.predict_match(
    home_team_id=86,      # Real Madrid
    away_team_id=83,      # Barcelona
    league_id=140,        # La Liga
    season_type=2
)
```

---

## 📊 Features do Modelo

O modelo utiliza mais de 40 features divididas em categorias:

### 1. **Forma Recente** (últimos 5 jogos)
- Pontos por jogo (PPG)
- Gols marcados/sofridos médios
- Taxa de vitórias
- Para mandante e visitante

### 2. **Confrontos Diretos (H2H)**
- Taxa de vitórias históricas
- Taxa de empates
- Número de confrontos

### 3. **Classificação (Standings)**
- Posição na tabela
- Pontos totais
- Vitórias/Empates/Derrotas
- Saldo de gols
- PPG da temporada
- Taxa de vitórias

### 4. **Estatísticas do Time**
- Posse de bola média
- Chutes totais/a gol
- Escanteios
- Cartões
- Faltas
- Defesas

### 5. **Features Derivadas**
- Diferença de posição na tabela
- Diferença de pontos
- Diferença de forma recente
- Diferença de saldo de gols

---

## 📈 Resultados Esperados

### Métricas Típicas:

| Métrica | Valor Esperado |
|---------|----------------|
| **Acurácia Geral** | 50-55% |
| **Precisão Home Win** | 55-60% |
| **Precisão Draw** | 30-35% |
| **Precisão Away Win** | 45-50% |

### Por que essas métricas?

- **Baseline aleatório**: 33% (1 em 3 chances)
- **Nosso modelo**: 50-55% = **melhoria de 50-65%**
- Empates são mais difíceis de prever (menos padrão)
- Vitórias do mandante são mais previsíveis (vantagem de casa)

### Features Mais Importantes (típicas):

1. Forma recente (PPG)
2. Diferença de posição na tabela
3. Confrontos diretos
4. Saldo de gols
5. Pontos na classificação

---

## 💡 Exemplos

### Exemplo 1: Explorar Dados

```python
from data_explorer import SoccerDataExplorer

explorer = SoccerDataExplorer(BASE_PATH)
explorer.load_all_data()

# Buscar time
explorer.search_team('Real Madrid')
# Output: ID: 86, Nome: Real Madrid, País: Spain

# Buscar liga
explorer.search_league('Champions')
# Output: ID: 2, Nome: UEFA Champions League

# Ver classificação
explorer.get_league_standings(140)  # La Liga
```

### Exemplo 2: Treinar Modelo Rápido

```python
from soccer_predictor import SoccerMatchPredictor

predictor = SoccerMatchPredictor(BASE_PATH)
predictor.load_data()

# Usar amostra pequena para teste rápido
predictor.engineer_features(sample_size=5000)
results = predictor.train_model()

print(f"Acurácia: {results['test_accuracy']:.1%}")
```

### Exemplo 3: Predição Completa

```python
# Exemplo: Real Madrid vs Barcelona
result = predictor.predict_match(
    home_team_id=86,      # Real Madrid
    away_team_id=83,      # Barcelona  
    league_id=140,        # La Liga
    season_type=2
)

# Output:
# 🔮 PREDIÇÃO:
#    Resultado previsto: Home Win
#    Probabilidades:
#       Away Win: 25.3%
#       Draw:     22.1%
#       Home Win: 52.6%
```

### Exemplo 4: Próximas Partidas de um Time

```python
explorer = SoccerDataExplorer(BASE_PATH)
explorer.load_all_data()

# Ver próximas 5 partidas do Real Madrid
explorer.find_upcoming_matches(team_id=86, limit=5)

# Output com IDs para usar no predictor
```

---

## 🎓 Workflow Recomendado

### Para Primeiro Uso:

1. **Execute o explorador** para conhecer o dataset
   ```bash
   python main_runner.py --explore
   ```

2. **Busque os times** que você quer prever
   - Anote os `teamId`
   - Anote os `leagueId`

3. **Treine o modelo** (comece com amostra média)
   ```bash
   python main_runner.py --train
   ```
   - Escolha opção 2 (10.000 partidas)
   - Aguarde ~2-5 minutos

4. **Faça predições**
   ```bash
   python main_runner.py --predict
   ```
   - Use os IDs anotados
   - Analise as probabilidades

---

## 🔍 Troubleshooting

### Erro: "File not found"
**Solução**: Verifique o caminho em `BASE_PATH` no código

### Erro: "Team ID not found"
**Solução**: Use o explorador para encontrar IDs válidos

### Baixa acurácia (< 45%)
**Possíveis causas**:
- Amostra muito pequena (use mais dados)
- Liga com poucos dados históricos
- Partidas muito imprevisíveis (copas, amistosos)

### Processamento lento
**Soluções**:
- Use `sample_size` menor para testes
- Processe menos features
- Use máquina mais potente

---

## 📚 Próximos Passos (Melhorias Futuras)

### Nível Avançado:

1. **Ensemble Methods**
   - Combinar Random Forest + XGBoost + LightGBM
   - Voting Classifier

2. **Deep Learning**
   - LSTM para sequências temporais
   - Neural Networks com embeddings

3. **Features Adicionais**
   - Lineups (escalações)
   - Player stats (estatísticas individuais)
   - Weather data (clima)
   - Odds de casas de apostas

4. **Calibração de Probabilidades**
   - Platt Scaling
   - Isotonic Regression

5. **Análise por Liga**
   - Modelos especializados por campeonato

---

## 📞 Suporte

### Erros Comuns:

| Erro | Solução |
|------|---------|
| `KeyError` | Verificar nomes das colunas no CSV |
| `ValueError` | Verificar tipos de dados (int vs float) |
| `MemoryError` | Reduzir `sample_size` |
| `IndexError` | Verificar se há dados suficientes |

---

## 📄 Licença

Este projeto é para fins educacionais.

---

## 🙏 Créditos

- **Dataset**: ESPN Soccer API (via Kaggle)
- **Algoritmo**: Random Forest (scikit-learn)
- **Desenvolvido para**: Projeto Final - 6º Período

---

## ✅ Checklist de Verificação

Antes de começar, certifique-se:

- [ ] Python 3.8+ instalado
- [ ] Bibliotecas instaladas (`pandas`, `numpy`, `scikit-learn`)
- [ ] Dataset baixado e descompactado
- [ ] Caminho `BASE_PATH` configurado corretamente
- [ ] CSVs principais presentes (`fixtures.csv`, `standings.csv`, `teamStats.csv`)

---

**Versão**: 1.0  
**Última atualização**: Dezembro 2024  
**Status**: ✅ Pronto para uso

🎯 **Boa sorte com suas predições!**