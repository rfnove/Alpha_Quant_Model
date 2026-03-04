#%%
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
import datetime
import mlflow
from utils import TechnicalFeatures

def backtest_alpha_seeker(
    run_id: str,
    ticker: str = "VALE3.SA",
    initial_capital: float = 100000.0,
    commission: float = 0.0003,  # Taxa B3 de ~0.03% (Corretagem Zero)
    test_start_date: str = '2023-01-01'
) -> dict:
    """
    Simulador Event-Driven para o modelo Alpha Seeker.
    Estratégia Híbrida: Comprado na ação ou rendendo CDI.
    """
    print(f"📥 Baixando dados e conectando ao MLflow (Run: {run_id[:8]})...")
    
    # 1. CARREGAR O MODELO DO MLFLOW
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    model_uri = f"runs:/{run_id}/modelo_random_forest"
    model = mlflow.sklearn.load_model(model_uri)
    features_esperadas = model.feature_names_in_

    # 2. ENGENHARIA DE DADOS (Igual ao seu app.py)
    hoje = datetime.datetime.today()
    cinco_anos_atras = hoje - datetime.timedelta(days=365 * 5)
    
    tickers_busca = [ticker, '^BVSP', 'BRL=X']
    df_yahoo = yf.download(tickers_busca, start=cinco_anos_atras, end=hoje, multi_level_index=False)
    df_final = df_yahoo['Close'].copy()
    
    data_selic = sgs.get(432, start=cinco_anos_atras.strftime('%Y-%m-%d'), end=hoje.strftime('%Y-%m-%d'))
    data_selic.index = pd.to_datetime(data_selic.index).tz_localize(None)
    df_final.index = pd.to_datetime(df_final.index).tz_localize(None)
    
    df_final['SELIC META ANUAL'] = data_selic['432']
    df_final = df_final.ffill().dropna()
    df_final['selic_diaria'] = (1 + (df_final['SELIC META ANUAL'] / 100)) ** (1/252) - 1
    
    # Aplicar as suas features
    df_master = TechnicalFeatures.add_all(df_final, coluna_alvo=ticker)
    
    # 3. SPLIT DE TREINO/TESTE PARA O LIMIAR SEGURO (Prevenindo Leakage)
    df_train = df_master[df_master.index < test_start_date].copy()
    df_test = df_master[df_master.index >= test_start_date].copy()
    
    # Descobre o percentil 60% apenas no TREINO
    y_train_proba = model.predict_proba(df_train[features_esperadas])[:, 1]
    safe_threshold = np.percentile(y_train_proba, 60)
    print(f"🎯 Threshold Seguro Calibrado (Treino): {safe_threshold:.4f}")
    
    # 4. SIMULAÇÃO DE TRADING (O Motor Event-Driven)
    capital = initial_capital
    position = 0
    shares = 0
    
    trades = []
    equity_curve = [{'Date': df_test.index[0], 'Equity': initial_capital}]
    
    # Fazemos a inferência de todo o teste de uma vez por eficiência
    df_test['proba'] = model.predict_proba(df_test[features_esperadas])[:, 1]
    
    print("Iniciando simulação dia a dia...")
    for i in range(len(df_test) - 1):
        hoje_data = df_test.index[i]
        amanha_data = df_test.index[i+1]
        
        # O modelo olha para hoje e decide o que fazer amanhã (Shift Realista)
        proba_hoje = df_test['proba'].iloc[i]
        preco_amanha = df_test[ticker].iloc[i+1]
        selic_amanha = df_test['selic_diaria'].iloc[i+1]
        
        # Sinal
        sinal_compra = proba_hoje >= safe_threshold
        
        # Execução
        if position == 0:
            if sinal_compra:
                # COMPRAR (Gasta o dinheiro e adquire ações)
                investment = capital * 0.99  # Deixa 1% de margem de segurança
                shares = int(investment / preco_amanha)
                cost = shares * preco_amanha * (1 + commission)
                capital -= cost
                position = 1
                trades.append({"type": "buy", "date": amanha_data, "price": preco_amanha, "shares": shares})
            else:
                # CONTINUAR FORA (Rende o CDI diário)
                capital *= (1 + selic_amanha)
                
        elif position == 1:
            if not sinal_compra:
                revenue = shares * preco_amanha * (1 - commission)
                capital += revenue
                shares = 0
                position = 0
                capital *= (1 + selic_amanha) # Já ganha o CDI do dia
                trades.append({"type": "sell", "date": amanha_data, "price": preco_amanha, "shares": 0, "pnl": revenue})
            else:
                # CONTINUAR COMPRADO (Patrimônio flutua com a ação)
                pass 
                
        # Calcula o patrimônio total do dia
        equity_amanha = capital + (shares * preco_amanha if position == 1 else 0)
        equity_curve.append({'Date': amanha_data, 'Equity': equity_amanha})

    # 5. FECHAMENTO DO CAIXA (Métricas)
    final_equity = equity_curve[-1]['Equity']
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Buy and hold base (Comprando no primeiro dia de teste e vendendo no último)
    preco_inicial_teste = df_test[ticker].iloc[0]
    preco_final_teste = df_test[ticker].iloc[-1]
    buy_hold_return = (preco_final_teste / preco_inicial_teste - 1) * 100
    
    # Drawdown Máximo
    df_equity = pd.DataFrame(equity_curve).set_index('Date')
    peak = df_equity['Equity'].expanding().max()
    drawdown = (df_equity['Equity'] - peak) / peak
    max_dd = drawdown.min() * 100

    return {
        "capital_final": final_equity,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_dd,
        "n_trades": len([t for t in trades if t["type"] == "buy"])
    }
#%%
# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    
    # SUBSTITUA PELO SEU RUN ID DO RANDOM FOREST NO MLFLOW
    MEU_RUN_ID = "99d968011aa04da0ade48790cde102a2"
    
    resultados = backtest_alpha_seeker(run_id=MEU_RUN_ID)
    
    print("\n" + "="*40)
    print("📈 EXTRATO FINANCEIRO - CONTA CORRENTE")
    print("="*40)
    print(f"Saldo Inicial:     R$ 100,000.00")
    print(f"Saldo Final:       R$ {resultados['capital_final']:,.2f}")
    print("-" * 40)
    print(f"Retorno do Robô:   {resultados['total_return']:+.2f}%")
    print(f"Retorno do Mercado:{resultados['buy_hold_return']:+.2f}%")
    print(f"Excesso (Alpha):   {resultados['total_return'] - resultados['buy_hold_return']:+.2f}%")
    print(f"Drawdown Máximo:   {resultados['max_drawdown']:.2f}%")
    print(f"Total de Compras:  {resultados['n_trades']}")
    print("="*40)
# %%
