#%%
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
#%%
df=pd.read_csv('abt.csv')
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#SAMPLE
split_date = pd.to_datetime('2023-01-01') 
df['Date'] = pd.to_datetime(df['Date'])  
df_test = df[df['Date'] >= split_date].copy() 
df_test = df_test.dropna(subset=['target_6m'])
df_train=df[df['Date']<split_date].copy()
new_split_date=df_train['Date'].iloc[-126]
df_train=(df[df['Date']<new_split_date])
features=['BRL=X', 'VALE3.SA', '^BVSP', 'SELIC META ANUAL', 'return_1d',
       'return_5d', 'return_20d', 'sma_20', 'dist_sma_20', 'sma_50',
       'dist_sma_50', 'sma_200', 'dist_sma_200', 'volatility_20d', 'rsi_14',
       'retorno_ibov_20d', 'retorno_dolar_20d', 'variacao_selic_20d',
       'alpha_ibov_20d']
target='target_6m'
X_train,y_train,X_test,y_test=df_train[features],df_train[target],df_test[features],df_test[target]
#%%
#EXPLORE
print("Taxa variável resposta geral:",df.dropna(subset=[target])[target].mean())
print("Taxa variável resposta Treino:", y_train.mean())
print("Taxa variável resposta Test:", y_test.mean())

df_analise = X_train.copy()
df_analise[target] = y_train
summario = df_analise.groupby(by=target).agg(["mean", "median"]).T
summario['diff_abs'] = summario[0] - summario[1]
summario['diff_rel'] = summario[0] / summario[1]
summario.sort_values(by=['diff_rel'], ascending=False)
#MODIFY
#%%
#MODEL
mlflow.set_experiment("Previsao_Tendencia_VALE3.SA")
# params={
#     'max_depth':2,            
#     'learning_rate':0.01,     
#     'subsample':0.8,          
#     'colsample_bytree':0.8,
#     'n_estimators':300,
#     'min_child_weight':5,
#     'reg_lambda':2.0, 
#     'scale_pos_weight':2.0,
#     'random_state':42
# }
params={'n_estimators':100,
    'max_depth':3,            # Limita o crescimento para não decorar ruído
    'min_samples_leaf':100,  # Exige mais dados para criar uma regra
    'max_features':'sqrt',  
    'random_state':42,
    'n_jobs':-1}
# param_grid = {
#     'max_depth': [2, 3, 4],               
#     'learning_rate': [0.01, 0.05],        
#     'min_child_weight': [2, 5],           
#     'scale_pos_weight': [1.0, 2.0]        
# }
# tscv = TimeSeriesSplit(n_splits=3,gap=126)
# xgb_base = XGBClassifier(n_estimators=200, reg_lambda=2.0, random_state=42)
# grid_search = GridSearchCV(
#     estimator=xgb_base,
#     param_grid=params,
#     scoring='roc_auc',  #melhor auc
#     cv=tscv,           I
#     verbose=1,          
#     n_jobs=-1         
# )

with mlflow.start_run(run_name="Iteracao_grid_best_model"):
    mlflow.sklearn.autolog()
    #mlflow.log_params(params)
    best_model=ensemble.RandomForestClassifier(**params)
    # best_model = XGBClassifier(**params)
    best_model.fit(X_train, y_train)
    #ASSES
    y_train_predict = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)[:,1]
    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)

    y_test_predict = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:,1]
    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Acurácia Test:", acc_test)
    print("AUC Test:", auc_test)

    df_test['sinal_robo']=y_test_predict
    df_test['retorno_estrategia']=df_test['retorno_futuro_6m']*df_test['sinal_robo']

    
    
    # O Relatório da Verdade
    df_audit = df_test[['retorno_futuro_6m', 'sinal_robo', 'retorno_estrategia']].copy()
    conditions = [
            (df_audit['sinal_robo'] == 1) & (df_audit['retorno_futuro_6m'] > 0), # Acertou Alta (Lucro)
            (df_audit['sinal_robo'] == 0) & (df_audit['retorno_futuro_6m'] < 0), # Acertou Baixa (Livrou a pele)
            (df_audit['sinal_robo'] == 0) & (df_audit['retorno_futuro_6m'] > 0), # Errou: Perdeu Alta (Custo Oportunidade)
            (df_audit['sinal_robo'] == 1) & (df_audit['retorno_futuro_6m'] < 0)  # Errou: Comprou na Baixa (Prejuízo)
        ]
    choices = ['Lucro Real', 'Proteção (Salvou)', 'Deixou de Ganhar', 'Prejuízo Real']

    # Classifica o resultado de cada trade
    df_audit['tipo_trade'] = np.select(conditions, choices, default='Neutro')
    print(df_audit['tipo_trade'].value_counts())
    print("\n--- Média de Retorno por Tipo de Erro ---")
    print(df_audit.groupby('tipo_trade')['retorno_futuro_6m'].mean())

    limiar_forçado = np.percentile(y_train_proba, 60) 
    
    df_test['sinal_agressivo'] = (y_test_proba >= limiar_forçado).astype(int)
    df_test['retorno_estrategia_agressiva'] = df_test['retorno_futuro_6m'] * df_test['sinal_agressivo']

        # 1. Transformando a Selic Anual em Retorno Fixo de 6 meses (126 dias úteis / 252)
    # Assumime que a coluna 'SELIC' está em formato percentual (ex: 11.25). 
    df_test['retorno_rf_6m'] = (1 + (df_test['SELIC META ANUAL'] / 100)) ** (126 / 252) - 1

    # Se o robô diz '1', assumime o risco da ação. Se diz '0', ganha a Selic!
    df_test['retorno_estrategia_com_rf'] = np.where(
        df_test['sinal_agressivo'] == 1, 
        df_test['retorno_futuro_6m'], 
        df_test['retorno_rf_6m']
    )

    soma_mercado = df_test['retorno_futuro_6m'].sum()
    soma_robo_agressivo = df_test['retorno_estrategia_agressiva'].sum()
    soma_robo_hibrido = df_test['retorno_estrategia_com_rf'].sum()
    soma_robo=df_test['retorno_estrategia'].sum()
    print("\n--- 🏁 O PLACAR DEFINITIVO (SCORE DE JANELAS) ---")
    print(f"Mercado (Buy & Hold Cego): {soma_mercado:.2f} pontos")
    print(f"Robô (Apenas Ação):        {soma_robo_agressivo:.2f} pontos")
    print(f"Robô Híbrido (Ação + CDI): {soma_robo_hibrido:.2f} pontos")    # Com CDI
    print(f"Robô normal:               {soma_robo:.2f} pontos")



# 1. Taxa Selic Diária (Juros Compostos: transformando anual em diária)
    # Assume 252 dias úteis no ano
    df_test['selic_diaria'] = (1 + (df_test['SELIC META ANUAL'] / 100)) ** (1/252) - 1

    # 2. O Atraso da Realidade (Shift)
    # O modelo gera o sinal no fechamento de hoje. Nós o executamos para capturar o rendimento de AMANHÃ.
    df_test['sinal_execucao'] = df_test['sinal_robo'].shift(1)

    # 3. O Retorno Diário da Estratégia
    df_test['retorno_diario_robo'] = np.where(
        df_test['sinal_execucao'] == 1,
        df_test['return_1d'],     # Se comprado, ganha/perde a variação da ação no dia
        df_test['selic_diaria']   # Se fora, ganha a Selic do dia
    )

    # Remove o primeiro dia do teste que ficou com NaN por causa do shift
    df_backtest = df_test.dropna(subset=['sinal_execucao', 'return_1d']).copy()

    # O .cumprod() simula o dinheiro rolando e acumulando dia após dia
    df_backtest['carteira_mercado'] = (1 + df_backtest['return_1d']).cumprod()
    df_backtest['carteira_robo'] = (1 + df_backtest['retorno_diario_robo']).cumprod()

    #Placar Financeiro Definitivo (Subtraindo 1 para ter a porcentagem de lucro limpo)
    lucro_mercado_real = df_backtest['carteira_mercado'].iloc[-1] - 1
    lucro_robo_real = df_backtest['carteira_robo'].iloc[-1] - 1

    print("\n--- 🏁 BACKTEST DE EXECUÇÃO REAL (MARCAÇÃO A MERCADO) ---")
    print(f"Rentabilidade Real do Mercado (VALE3): {lucro_mercado_real:.2%}")
    print(f"Rentabilidade Real do Robô Híbrido:    {lucro_robo_real:.2%}")




    
    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "retorno_robo":soma_robo,
    'retorno_robo_agressivo':soma_robo_agressivo,
    'soma_robo_hibrido':soma_robo_hibrido,
    "retorno_mercado":soma_mercado,
    'rentabilidade_real_mercado':lucro_mercado_real,
    'rentabilidade_real_robo_hibrido': lucro_robo_real

    })
    

    # mlflow.xgboost.log_model(best_model, 'modelo_xgboost')
    #print("Run do MLflow finalizada com sucesso!")
    mlflow.sklearn.log_model(best_model, "modelo_random_forest")
# %%
#mlflow ui
# %%
import matplotlib.pyplot as plt
    
# Plotando as duas carteiras (A linha de base começa em 1.0)
plt.figure(figsize=(12, 6))  
plt.plot(df_backtest['Date'], df_backtest['carteira_mercado'], label='Mercado (Buy & Hold VALE3)', color='red', alpha=0.7)
plt.plot(df_backtest['Date'], df_backtest['carteira_robo'], label='Robô Híbrido (Ação + CDI)', color='green', linewidth=2)   
plt.fill_between(df_backtest['Date'], df_backtest['carteira_mercado'], df_backtest['carteira_robo'], 
                     where=(df_backtest['carteira_robo'] > df_backtest['carteira_mercado']), 
                     interpolate=True, color='green', alpha=0.1, label='Alpha Gerado')   
plt.title('Curva de Patrimônio: Robô Híbrido vs Mercado (VALE3 - Out of Sample)', fontsize=14, fontweight='bold')
plt.xlabel('Data', fontsize=12)
plt.ylabel('Crescimento do Capital (1.0 = Início)', fontsize=12)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5) # Linha do zero a zero
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
nome_grafico = "curva_patrimonio_vale3.png"
plt.savefig(nome_grafico, dpi=300)
# %%
# %%
