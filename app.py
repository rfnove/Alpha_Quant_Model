import streamlit as st
import mlflow.xgboost
import mlflow.sklearn
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
import datetime
from utils import TechnicalFeatures
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
st.set_page_config(page_title="QuantTrend - IA de Investimentos", page_icon="📈", layout="centered")

st.title("📈 QuantTrend: IA para Position Trading")
st.markdown("Previsão de tendência direcional para **6 meses** utilizando Machine Learning Quantitativo.")

#RUN_ID = "b9da8738ac20402cb4d919978c74dd08"
RUN_ID ="122ae014f4334679800c9dc976a026c4"
# MODEL_URI = f"runs:/{RUN_ID}/modelo_xgboost"
MODEL_URI = f"runs:/{RUN_ID}/modelo_random_forest"
@st.cache_resource 
def load_quant_model():
    return mlflow.sklearn.load_model(MODEL_URI)

try:
    model = load_quant_model()
    st.sidebar.success("🤖 Modelo Quantitativo Ativo!")
    st.sidebar.info(f"Run ID: {RUN_ID[:8]}...")
except Exception as e:
    st.sidebar.error("Erro ao carregar o modelo. Verifique o RUN_ID.")
    st.stop()

st.write("---")
col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input("Digite o Ticker da Ação (Padrão Yahoo):", value="VALE3.SA").upper()

with col2:
    st.write("") 
    st.write("")
    analisar_btn = st.button("🔮 Analisar Tendência", use_container_width=True)

if analisar_btn:
    st.info(f"Iniciando varredura ao vivo para {ticker_input}...")
    
    try:
        with st.spinner('Extraindo cotações da B3 e Wall Street (Último 1 ano)...'):
            tickers_busca = [ticker_input, '^BVSP', 'BRL=X']
            df_yahoo = yf.download(tickers_busca, period="1y", multi_level_index=False)
            df_final = df_yahoo['Close'].copy()
            
        with st.spinner('Buscando Taxa Selic no Banco Central...'):
            hoje = datetime.datetime.today()
            um_ano_atras = hoje - datetime.timedelta(days=365)
            
            # Puxa os dados usando start e end (formato YYYY-MM-DD)
            data_selic = sgs.get(432, start=um_ano_atras.strftime('%Y-%m-%d'), end=hoje.strftime('%Y-%m-%d'))
            
            # Garantimos que a data_selic tenha o índice de data correto e a mesma timezone (UTC-0 ou None)
            data_selic.index = pd.to_datetime(data_selic.index).tz_localize(None)
            df_final.index = pd.to_datetime(df_final.index).tz_localize(None)
            
            # Junta a Selic ao DataFrame
            df_final['SELIC META ANUAL'] = data_selic['432']
            
            # Preenche feriados e finais de semana igual no treino
            df_final = df_final.ffill().dropna()

        with st.spinner('Calculando RSI, Médias e Contexto Macro...'):
            df_master = TechnicalFeatures.add_all(df_final, coluna_alvo=ticker_input)
            df_hoje = df_master.tail(1)
            
            
            features_esperadas = model.feature_names_in_
            X_live = df_hoje[features_esperadas]
        with st.spinner('Consultando o Random Forest...'):
            probabilidade_alta = model.predict_proba(X_live)[0][1]
            
        st.success("Análise Quantitativa concluída com sucesso!")
        st.markdown("### Veredito do Robô")
        
        col_metrica, col_decisao = st.columns(2)
        
        with col_metrica:
            st.metric(label="Probabilidade de Alta (6 Meses)", value=f"{probabilidade_alta:.1%}")
        
        with col_decisao:
           
            limiar_agressivo = 0.50 
            
            if probabilidade_alta >= limiar_agressivo:
                st.success("SINAL VERDE: COMPRAR 🟢")
                st.write("Assimetria de risco positiva. Potencial de superar o CDI.")
            else:
                st.warning("SINAL VERMELHO: FICAR NO CDI 🔴")
                st.write("Risco elevado. O modelo sugere proteção na Renda Fixa.")

    except Exception as e:
        st.error(f"Erro na execução da Pipeline: {e}")
        st.write("Verifique se o Ticker foi digitado corretamente.")

#python -m streamlit run app.py