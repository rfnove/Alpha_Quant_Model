#%%
import pandas as pd 
import numpy as np
import yfinance as yf
from bcb import sgs
from utils import TechnicalFeatures, criar_targets_tendencia
#%%
#import dados do santander ibovespa e dolar
ticker=['VALE3.SA','^BVSP','BRL=X']
start='2016-01-01'
end='2026-01-01'
df=yf.download(ticker,start=start,end=end,multi_level_index=False)
df
#%%
#importando dados da selic diretamente do banco central
cod_selic = 432
data_selic = sgs.get(cod_selic, start=start, end=end)
data_selic
#%%
# juntando tudo em um unico data frame
df_final=df['Close'].copy()
df_final['SELIC META ANUAL']=data_selic['432']
df_final
#%%
df_final=df_final.dropna(subset="^BVSP")
df_final=df_final.ffill()
df_final=df_final.dropna()

#%%
#%%    
# Aplicando a função no DataFrame limpo
df_master = TechnicalFeatures.add_all(df_final, coluna_alvo='VALE3.SA')
df_features = criar_targets_tendencia(df_master, coluna_preco='VALE3.SA')
print("\n--- Verificando o Target de 6 Meses ---")
print(df_features.dropna(subset=['target_6m'])[['VALE3.SA', 'retorno_futuro_6m', 'target_6m']].tail(5))
#%%
df_features
#%%
df_features.to_csv('abt.csv') #salvar logo caso de ruim na biblioteca do banco central
# %%
