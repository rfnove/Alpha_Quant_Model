import numpy as np
import pandas as pd
class TechnicalFeatures:

    @staticmethod
    def add_returns(df, col):
        df["return_1d"] = df[col].pct_change(1)
        df["return_5d"] = df[col].pct_change(5)
        df["return_20d"] = df[col].pct_change(20) # Retorno mensal
        return df

    @staticmethod
    def add_moving_averages(df, col):
        for window in [20, 50, 200]:
            df[f"sma_{window}"] = df[col].rolling(window, min_periods=window).mean()
            # Distância do preço atual para a média 
            df[f"dist_sma_{window}"] = (df[col] / df[f"sma_{window}"]) - 1
        return df

    @staticmethod
    def add_volatility(df, col):
        # Volatilidade baseada no desvio padrão dos retornos
        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
        return df

    @staticmethod
    def add_rsi(df, col, period=14):
        delta = df[col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macro_features(df):
        """
        Como o ativo se comporta em relação ao mercado.
        """
        # Como está o mercado geral (Ibovespa)?
        df["retorno_ibov_20d"] = df["^BVSP"].pct_change(20)
        
        # Como está o câmbio (Dólar)?
        df["retorno_dolar_20d"] = df["BRL=X"].pct_change(20)
        
        # A Selic está subindo ou caindo no último mês?
        df["variacao_selic_20d"] = df["SELIC META ANUAL"].diff(20)
        
        # Força Relativa: A ação subiu mais ou menos que o Ibov no mês?
        if "return_20d" in df.columns:
            df["alpha_ibov_20d"] = df["return_20d"] - df["retorno_ibov_20d"]
            
        return df

    @staticmethod
    def add_all(df, coluna_alvo):
        df = df.copy()
        
        # Aplicando as features
        df = TechnicalFeatures.add_returns(df, coluna_alvo)
        df = TechnicalFeatures.add_moving_averages(df, coluna_alvo)
        df = TechnicalFeatures.add_volatility(df, coluna_alvo)
        df = TechnicalFeatures.add_rsi(df, coluna_alvo)
        df = TechnicalFeatures.add_macro_features(df)
        
        #  LIMPEZA OBRIGATÓRIA 
        # A média móvel de 200 dias (sma_200) vai criar 200 linhas de 'NaN' no começo do dataset.
        # dropar as linhas que não têm features para o modelo não quebrar.
        # IMPORTANTE: Não dropar os NaNs das colunas 'target_' ainda!
        features = [c for c in df.columns if not c.startswith('target_') and not c.startswith('retorno_futuro_')]
        df = df.dropna(subset=features)
        
        return df

# ==========================================
# ENGENHARIA DE TARGETS (MÚLTIPLOS HORIZONTES)
# ==========================================

def criar_targets_tendencia(df, coluna_preco):
    """
    Cria alvos binários para prever se a ação vai subir (1) ou cair (0)
    em janelas de 3, 6, 9 e 12 meses.
    """
    df = df.copy()
    
    # Assumindo média de 21 dias úteis por mês na B3
    horizontes = {
        '3m': 63,   # 3 meses
        '6m': 126,  # 6 meses
        '9m': 189,  # 9 meses
        '12m': 252  # 12 meses
    }
    
    for label, dias in horizontes.items():
        # Desloca o preço 'N' dias para trás (traz o futuro para a linha atual)
        preco_futuro = df[coluna_preco].shift(-dias)
        
        # Opcional: Calcula o percentual exato de ganho/perda (útil para o backtest depois)
        df[f'retorno_futuro_{label}'] = (preco_futuro / df[coluna_preco]) - 1
        
        # TARGET BINÁRIO: 1 se o preço futuro for MAIOR que o atual; 0 caso contrário
        df[f'target_{label}'] = (preco_futuro > df[coluna_preco]).astype(int)
        
        # Os últimos dias do dataset não têm futuro.
        # Ex: Não sabemos o preço do final de 2026 ainda. O Pandas preencheria com 0.
        # Precisamos forçar para NaN para o XGBoost não aprender mentiras.
        df.loc[preco_futuro.isna(), f'target_{label}'] = np.nan
        
    return df