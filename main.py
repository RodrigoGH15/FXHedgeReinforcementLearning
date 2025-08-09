from datetime import date
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

"""
Modelo de Aprendizaje por Refuerzo (Q-Learning) para gestión de
coberturas FX USD→CLP.

Contexto financiero:
En la gestión de portafolios de inversión expuestos a moneda
extranjera, decidir cuándo cubrir el riesgo cambiario es crítico. Una
cobertura excesiva puede limitar retornos, mientras que una cobertura
insuficiente puede exponer al portafolio a pérdidas significativas.
Este programa entrena un agente usando Q-learning para aprender una
política óptima de cobertura basada en datos históricos.

Motivación laboral:
El modelo se puede integrar en un flujo real de monitoreo de fondos con
exposición USD→CLP para sugerir o ejecutar coberturas automáticas según
condiciones de mercado, optimizando la relación riesgo-retorno neta de
costos de cobertura.

Warning:
Misma metodología de la tarea anterior. Trata de leer un archivo xlsx
para buscar los tipos de cambio y si no lo encuentra recurre a yfinance.
El formato del xlsx tiene que ser una columna Date y una Adj_Close.
"""

# -------------------------------
# Parámetros del modelo
# -------------------------------
ALPHA = 0.1  # Tasa de aprendizaje
GAMMA = 0.95  # Factor de descuento
EPSILON = 0.1  # Probabilidad de explorar
N_ATTEMPTS = 500  # Número de intentos de entrenamiento
HEDGE_COST = 0.0005  # Costo de cobertura por operación (0.05%)

# -------------------------------
# Funciones auxiliares
# -------------------------------


def _get_fx_data(ticker: str, start: date, end: date) -> pd.Series:
    """Obtiene datos de tipo de cambio desde archivo local o yfinance.
    Similar a la lógica usada para acciones en la tarea anterior.
    """
    # --- 1. Intentar leer desde archivo local
    try:
        fx_data = pd.read_excel("fx_data.xlsx")
        print("Se leyeron los datos de FX desde archivo local...")
        if "Date" in fx_data.columns:
            fx_data.set_index("Date", inplace=True)
        return fx_data["Adj_Close"]
    except FileNotFoundError:
        print("Archivo local no encontrado. Descargando desde yfinance...")

    # --- 2. Descargar desde yfinance
    try:
        df_fx = yf.download(ticker, start=start, end=end, auto_adjust=False)
    except Exception as e:
        print(f"Error al descargar datos de yfinance: {e}")
        raise e
    if df_fx is None:
        raise ValueError("Check df_fx for problems.")
    if "Date" in df_fx.columns:
        df_fx.set_index("Date", inplace=True)
    return df_fx["Adj Close"].squeeze()  # type: ignore


def load_fx_data(filepath: str) -> pd.DataFrame:
    """Carga datos históricos de USDCLP desde un CSV local."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=[0])
    df = df.ffill()  # Si no tengo un día lleno con el día anterior
    return df


def calculate_state_variables(prices: pd.Series) -> pd.DataFrame:
    """Calcula retornos, volatilidad y tendencia como variables de estado."""
    returns = cast(pd.Series, np.log(prices / prices.shift(1)))
    vol = returns.rolling(window=5).std()
    trend = prices.rolling(window=5).mean() - prices.rolling(window=20).mean()

    # Convertimos variables continuas en bins (-1, 0, 1)
    # Esto reduce el espacio de estados y permite que q-learning funcione en
    # datos financieeros
    def discretize(series):
        return pd.qcut(series, q=3, labels=[-1, 0, 1])

    return pd.DataFrame(
        {
            "return_bin": discretize(returns),
            "vol_bin": discretize(vol),
            "trend_bin": discretize(trend),
        }
    ).dropna()


def step_value_clp(
    action: int, fx_ret: float, include_hedge_cost: bool = True
) -> float:
    """
    Calcula el cambio en valor CLP del portafolio dado:
    - acción (0 = no cubrir, 1 = cubrir 100%)
    - retorno FX del día
    Aplica costo de cobertura si corresponde.
    """
    if action == 0:  # No cubrir
        pnl = fx_ret
    else:
        pnl = 0 - HEDGE_COST if include_hedge_cost else 0
        #  si uno se cubre el costo en el pnl baja a cero y se paga solo
        # el hedge cost
    return pnl


# -------------------------------
# Entrenamiento Q-learning
# -------------------------------


def train_q_learning(
    states: pd.DataFrame,
    fx_returns: pd.Series,
    include_hedge_cost: bool = True,
) -> dict:
    """
    Entrena un agente Q-learning para decidir la cobertura óptima.
    Devuelve una tabla Q como diccionario: {estado: [Q_no_cover, Q_cover]}.
    """
    q_table = {}
    state_list = list(states.itertuples(index=False, name=None))

    for _ in range(N_ATTEMPTS):
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i + 1]
            reward = step_value_clp(0, fx_returns.iloc[i])
            # el base es no cubrir
            if state not in q_table:
                q_table[state] = [0, 0]

            # Epsilon-greedy: explorar o explotar
            if np.random.rand() < EPSILON:
                action = np.random.randint(0, 2)
            else:
                action = cast(int, np.argmax(q_table[state]))

            # Recalcular recompensa con acción elegida
            reward = step_value_clp(
                action,
                fx_returns.iloc[i],
                include_hedge_cost=include_hedge_cost,
            )

            # Inicializar siguiente estado si no existe
            if next_state not in q_table:
                q_table[next_state] = [0, 0]

            # Actualizar el  Q-learning
            q_old = q_table[state][action]
            q_table[state][action] = q_old + ALPHA * (
                reward + GAMMA * max(q_table[next_state]) - q_old
            )

    return q_table


def derive_policy(q_table: dict) -> dict:
    """Genera la política óptima (acción con mayor Q) para cada estado."""
    return {state: np.argmax(actions) for state, actions in q_table.items()}


# -------------------------------
# Ejecución principal
# -------------------------------


def main():
    start_date = date(2022, 1, 2)
    end_date = date(2024, 12, 30)
    include_hedge_cost = True

    # 1. Cargar datos
    fx_data = _get_fx_data("USDCLP=X", start=start_date, end=end_date)

    file_path = Path("fx_data.xlsx")
    if not file_path.is_file():
        # File does not exist, so create it
        pd.DataFrame(fx_data).to_excel(file_path, header=["Adj_Close"])
        print(f"File '{file_path}' created successfully.")
    else:
        print(f"File '{file_path}' already exists.")

    # 2. Calcular estados y retornos
    states = calculate_state_variables(fx_data)
    fx_returns = cast(pd.Series, np.log(fx_data / fx_data.shift(1)))
    fx_returns = fx_returns.fillna(0)

    # 3. Entrenar agente
    q_table = train_q_learning(
        states, fx_returns, include_hedge_cost=include_hedge_cost
    )

    # 4. Derivar política
    policy = derive_policy(q_table)

    # 5. Interpretación de resultados
    cover_count = list(policy.values()).count(1)
    no_cover_count = list(policy.values()).count(0)

    print("\n--- Política aprendida ---")
    print(f"Estados totales: {len(policy)}")
    print(f"Recomendación de cubrir: {cover_count} estados")
    print(f"Recomendación de no cubrir: {no_cover_count} estados")
    print("\nInterpretación:")
    if include_hedge_cost:
        print(
            f"Costo de cobertura modelado: {HEDGE_COST * 100:.3f}% por trade\n"
        )
    else:
        print("No se está utilizando costo de cobertura.\n")

    # Ejemplo: graficar volatilidad vs recomendación
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(policy)), list(policy.values()))
    plt.xlabel("Estado")
    plt.ylabel("Acción (0=No cubrir, 1=Cubrir)")
    plt.title("Política óptima aprendida")
    plt.show()
    """
    Corrí el modelo 10 veces (con costos de cobertura)
    De los 27 estados totales (3 ** 3), el modelo recomendó hacer hedge
    de tipo de cambio en:
    20, 21, 21, 21, 21, 18, 21, 18, 22, 20
    de los 27 estados totales.
    Conclusiones:
    El Dominio de la Cautela 🛡️
    - El modelo tiende a recomendar cobertura en la mayoría de los estados,
      incluso con un costo de cobertura del 0.05%. Esto sugiere que, bajo
      las condiciones y parámetros definidos, el riesgo de no cubrir es
      generalmente mayor que el costo de la cobertura.
    - La consistencia en el número de estados cubiertos (entre 18 y 22 de
      27) a lo largo de múltiples corridas indica una política
      relativamente estable.
    - Sería interesante analizar qué combinaciones de 'return_bin',
      'vol_bin' y 'trend_bin' llevan a la decisión de no cubrir, para
      entender mejor las condiciones de mercado bajo las cuales el modelo
      considera que la exposición es preferible.
    - La visualización de la política óptima (gráfico de dispersión)
      muestra la distribución de las decisiones de cobertura a través de
      los estados, pero una visualización más detallada que mapee las
      decisiones a las variables de estado discretizadas podría ofrecer
      más insights.
    Finalmente, el agente aprendió a no ser un especulador, sino un gestor
    precavido de riesgo.
    """


if __name__ == "__main__":
    from rich import print

    print()
    main()
