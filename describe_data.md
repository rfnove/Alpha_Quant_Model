
---

# 📊 Dataset: Previsão de Tendência de Longo Prazo (MVP)

## Visão Geral

Este dataset foi construído para treinar um modelo de *Machine Learning* (XGBoost) com o objetivo de prever a **tendência direcional (Alta ou Baixa)** de uma ação específica em múltiplos horizontes de tempo (3, 6, 9 e 12 meses).

Diferente de abordagens focadas em ruído de curto prazo, este projeto adota uma visão de *Position/Swing Trading*, combinando Análise Técnica profunda com dados Macroeconômicos para capturar a tendência real do ativo.

* **Ativo Alvo (MVP):** Santander Brasil (`SANB11.SA`)
* **Período Histórico:** 01/01/2016 a 01/01/2026 (10 Anos)
* **Frequência:** Diária (Dias úteis de pregão na B3)

---

## Fontes de Dados e Engenharia

1. **Yahoo Finance (`yfinance`):** Utilizado para extrair o fechamento ajustado (`Adj Close`) do ativo alvo, do Ibovespa e da cotação do Dólar Comercial. O uso do fechamento ajustado garante que eventos corporativos (dividendos, desdobramentos) não poluam o modelo com quedas artificiais de preço.
2. **Banco Central do Brasil (`python-bcb` / SGS):** Utilizado para extrair a Meta da Taxa Selic (Código SGS: 432), injetando o contexto do custo de oportunidade e ciclo de juros brasileiro no modelo.
3. **Tratamento de Calendário (Sincronização):** O calendário da **B3 (Ibovespa) atua como calendário mestre**. Feriados nacionais onde não houve pregão foram removidos. Feriados internacionais (que afetam o Dólar, mas não a B3) tiveram seus valores preenchidos com o último dado válido via *Forward Fill* (`ffill()`).

---

## Dicionário de Variáveis (Features)

### 1. Dados Base (Cotações e Índices)

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `SANB11.SA` | Float | Preço de Fechamento Ajustado da ação alvo (Santander). |
| `^BVSP` | Float | Pontuação de Fechamento Ajustado do Índice Bovespa (Benchmark). |
| `BRL=X` | Float | Cotação de Fechamento do Dólar Comercial frente ao Real. |
| `SELIC` | Float | Meta da Taxa Selic anualizada (em %). |

### 2. Features Técnicas (Ação Alvo)

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `return_1d`, `_5d`, `_20d` | Float | Retorno percentual da ação em 1 dia, 5 dias (semana) e 20 dias (mês). |
| `sma_20`, `_50`, `_200` | Float | Média Móvel Simples (Simple Moving Average) de 20, 50 e 200 dias. |
| `dist_sma_20`, `_50`, `_200` | Float | Distância percentual do preço atual para suas respectivas médias móveis. Indica níveis de sobrecompra/sobrevenda. |
| `volatility_20d` | Float | Volatilidade anualizada calculada com base no desvio padrão dos retornos dos últimos 20 dias. |
| `rsi_14` | Float | Índice de Força Relativa (Relative Strength Index) de 14 dias. Oscila entre 0 e 100. |

### 3. Features Macroeconômicas (Contexto de Mercado)

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `retorno_ibov_20d` | Float | Retorno percentual do Ibovespa nos últimos 20 dias úteis. |
| `retorno_dolar_20d` | Float | Retorno percentual do Dólar nos últimos 20 dias úteis. |
| `variacao_selic_20d` | Float | Variação absoluta da taxa Selic nos últimos 20 dias úteis. |
| `alpha_ibov_20d` | Float | Força relativa do ativo contra o benchmark (`return_20d` - `retorno_ibov_20d`). Valores positivos indicam que a ação superou o mercado. |

---

## Variáveis Alvo (Os Targets)

Estas colunas representam o **futuro**. Elas foram construídas usando a técnica de deslocamento temporal negativo (`shift(-N)`) para trazer a resposta futura para a linha do presente. **Estas colunas não podem ser usadas como features no modelo, apenas como *labels* de treinamento.**

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `retorno_futuro_3m` | Float | Retorno percentual real que a ação obteve 63 dias úteis após a data atual. |
| `retorno_futuro_6m` | Float | Retorno percentual real que a ação obteve 126 dias úteis após a data atual. |
| `retorno_futuro_9m` | Float | Retorno percentual real que a ação obteve 189 dias úteis após a data atual. |
| `retorno_futuro_12m` | Float | Retorno percentual real que a ação obteve 252 dias úteis após a data atual. |
| **`target_3m`** | Binário | **1** se o `retorno_futuro_3m` for > 0 (Tendência de Alta). **0** caso contrário. |
| **`target_6m`** | Binário | **1** se o `retorno_futuro_6m` for > 0 (Tendência de Alta). **0** caso contrário. |
| **`target_9m`** | Binário | **1** se o `retorno_futuro_9m` for > 0 (Tendência de Alta). **0** caso contrário. |
| **`target_12m`** | Binário | **1** se o `retorno_futuro_12m` for > 0 (Tendência de Alta). **0** caso contrário. |

---

## ⚠️ Limitações e Cuidados com Vazamento de Dados (Data Leakage)

1. **Perda de Dados Iniciais (Warm-up):** Devido ao cálculo da média móvel de 200 dias (`sma_200`), os primeiros 200 pregões do dataset original são descartados automaticamente por conterem valores `NaN` nas features.
2. **Zona Cega do Futuro (Inferência):** As últimas linhas do dataset (ex: os últimos 126 pregões para a janela de 6 meses) possuem as colunas `target_` preenchidas com `NaN`. Isso ocorre porque o futuro dessas datas ainda não aconteceu no limite temporal de extração dos dados. **Essas linhas devem ser isoladas na etapa de treino para não corromper o modelo, sendo utilizadas exclusivamente para a etapa final de inferência preditiva.**
3. **Embargo de Treino/Teste:** Para evitar *overfitting* devido à sobreposição de janelas de previsão (*overlapping*), o particionamento temporal para validação cruzada ou split de teste exige um período de "embargo" (gap) equivalente à janela de previsão (ex: 6 meses).

