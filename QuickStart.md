# ⚡ Guia Rápido de Início

Este guia vai te ajudar a começar a usar o sistema em **5 minutos**.

## 📦 Passo 1: Instalação (2 min)

```bash
# Clone ou baixe o projeto
cd S.I-1-Projeto-Final/

# Instale as dependências
pip install pandas numpy xgboost scikit-learn

# Ou use o requirements.txt
pip install -r requirements.txt
```

## 📂 Passo 2: Verifique os Dados (1 min)

Certifique-se de que você tem os arquivos CSV nas pastas corretas:

```
data/
├── base_data/
│   ├── fixtures.csv       ✓
│   ├── standings.csv      ✓
│   ├── teamStats.csv      ✓
│   ├── players.csv        ✓
│   ├── teams.csv          ✓
│   ├── leagues.csv        ✓
│   └── status.csv         ✓
└── ...
```

**Verificação rápida:**

```bash
# Contar arquivos em base_data
ls data/base_data/*.csv | wc -l
# Deve retornar 7 ou mais
```

## 🚀 Passo 3: Treinar o Modelo (2 min)

```bash
python main.py --mode train
```

**O que acontece:**
1. ✅ Carrega ~30.000 partidas
2. ✅ Cria 30+ features preditivas
3. ✅ Treina modelo XGBoost
4. ✅ Salva modelo em `models/`

**Tempo estimado:** 1-3 minutos dependendo do hardware.

## 🎯 Passo 4: Fazer Predição (1 min)

```bash
python main.py --mode predict
```

**Entrada de exemplo:**

```
--- Time da Casa ---
Vitórias recentes: 3
Empates recentes: 1
Derrotas recentes: 1
Média de gols: 2.0
Média de gols sofridos: 1.0
Pontos: 50

--- Time Visitante ---
Vitórias recentes: 2
Empates recentes: 2
Derrotas recentes: 1
Média de gols: 1.5
Média de gols sofridos: 1.3
Pontos: 42
```

**Saída:**

```
🎯 RESULTADO DA PREDIÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Resultado Previsto: Vitória Casa
📊 Confiança: 68.2%

📈 Probabilidades:
   • Empate: 16.5%
   • Vitória Casa: 68.2%
   • Vitória Visitante: 15.3%
```

## 📊 Passo 5: Avaliar Modelo (opcional)

```bash
python main.py --mode evaluate
```

Mostra métricas detalhadas do modelo.

---

## 🆘 Problemas Comuns

### "Arquivo não encontrado"

```bash
# Verifique se está na pasta correta
pwd
# Deve terminar em: .../S.I-1-Projeto-Final

# Verifique se os dados existem
ls data/base_data/
```

### "ModuleNotFoundError"

```bash
# Reinstale as dependências
pip install --upgrade pandas numpy xgboost scikit-learn
```

### "KeyError" ou "Coluna não encontrada"

Seus CSVs podem estar em formato diferente. Verifique a documentação em `estrutura_data.docx`.

---

## 🎓 Próximos Passos

1. **Explorar os Dados**: Abra `notebooks/EDA.ipynb`
2. **Ajustar Modelo**: Edite `src/config.py` → `MODEL_PARAMS`
3. **Adicionar Features**: Edite `src/feature_engineering.py`
4. **Otimizar**: Use Grid Search para hyperparameter tuning

---

## 📚 Documentação Completa

Leia o `README.md` para entender o projeto em detalhes.

---

## 💡 Dicas

- **Primeiro treino**: Pode demorar alguns minutos
- **Treinos subsequentes**: Mais rápidos (cache de dados)
- **Dados grandes**: Use `reduce_mem_usage()` em `utils.py`
- **Logs**: Sempre verifique `logs/training_log.txt`

---

**Pronto! Você agora tem um sistema de predição de futebol funcionando! ⚽🚀**