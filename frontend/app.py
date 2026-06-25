import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="PreSIADA - Deteccion de Fraude",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── LIGHT MODE ── */
    [data-theme="light"] .stApp {
        background-color: #f0f4f8;
    }

    [data-theme="light"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a56db 0%, #1e40af 100%);
    }
    [data-theme="light"] [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
    }
    [data-theme="light"] [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255,255,255,0.15);
    }
    [data-theme="light"] [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > label {
        color: #ffffff !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #60a5fa !important;
        background-color: transparent !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div > div:first-child {
        border-color: #ffffff !important;
        background-color: #ffffff !important;
    }

    [data-theme="light"] [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
    }
    [data-theme="light"] [data-testid="metric-container"] label {
        color: #374151 !important;
    }
    [data-theme="light"] [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    [data-theme="light"] div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    [data-theme="light"] div[data-testid="stForm"] label {
        color: #1f2937 !important;
    }
    [data-theme="light"] div[data-testid="stForm"] input {
        color: #111827 !important;
    }
    [data-theme="light"] div[data-testid="stForm"] .stSelectbox label {
        color: #1f2937 !important;
    }

    [data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3,
    [data-theme="light"] h4, [data-theme="light"] p, [data-theme="light"] li,
    [data-theme="light"] span, [data-theme="light"] label, [data-theme="light"] div {
        color: #1e293b;
    }

    [data-theme="light"] .stMarkdown {
        color: #334155;
    }

    /* ── DARK MODE ── */
    [data-theme="dark"] .stApp {
        background-color: #0f172a;
    }

    [data-theme="dark"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #1e293b 100%);
    }
    [data-theme="dark"] [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-theme="dark"] [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
    }
    [data-theme="dark"] [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(59,130,246,0.2);
    }
    [data-theme="dark"] [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > label {
        color: #e2e8f0 !important;
    }
    [data-theme="dark"] [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #60a5fa !important;
        background-color: transparent !important;
    }
    [data-theme="dark"] [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div > div:first-child {
        border-color: #60a5fa !important;
        background-color: #60a5fa !important;
    }

    [data-theme="dark"] [data-testid="metric-container"] {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.3);
        border-left: 4px solid #3b82f6;
    }
    [data-theme="dark"] [data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    [data-theme="dark"] [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }

    [data-theme="dark"] div[data-testid="stForm"] {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
    }
    [data-theme="dark"] div[data-testid="stForm"] label {
        color: #e2e8f0 !important;
    }
    [data-theme="dark"] div[data-testid="stForm"] input {
        color: #f1f5f9 !important;
        background-color: #0f172a !important;
    }
    [data-theme="dark"] div[data-testid="stForm"] .stSelectbox label {
        color: #e2e8f0 !important;
    }

    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3,
    [data-theme="dark"] h4, [data-theme="dark"] p, [data-theme="dark"] li,
    [data-theme="dark"] span, [data-theme="dark"] label, [data-theme="dark"] div {
        color: #e2e8f0;
    }

    [data-theme="dark"] .stMarkdown {
        color: #cbd5e1;
    }

    [data-theme="dark"] [data-testid="stExpander"] {
        border-color: #334155;
    }
    [data-theme="dark"] [data-testid="stExpander"] summary span {
        color: #e2e8f0 !important;
    }

    [data-theme="dark"] [data-testid="stSpinner"] > div {
        border-top-color: #3b82f6;
    }

    /* ── SHARED ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
    }

    [data-testid="stHorizontalBlock"] > div {
        color: inherit;
    }

    [data-testid="stAlert"] p {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# PreSIADA")
    st.caption("Sistema de Deteccion de Fraude Financiero")
    st.markdown("---")
    page = st.radio("Navegacion", ["Dashboard del Modelo", "Simulador de Riesgo"])

if page == "Dashboard del Modelo":
    st.title("Dashboard del Modelo")
    st.markdown(
        "Clasificador **Random Forest** para deteccion de fraudes en transacciones "
    )
    st.markdown("---")

    st.subheader("Metricas de Rendimiento")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", "99.7%", help="Porcentaje de predicciones correctas")
    with c2:
        st.metric("Precision", "98.0%", help="Fraudes detectados que son reales")
    with c3:
        st.metric("Recall", "99.6%", help="Fraudes reales que fueron detectados")

    st.markdown("---")
    st.subheader("Matriz de Confusion")

    fig, ax = plt.subplots(figsize=(6, 5))

    tn = 552409
    fp = 30
    fn = 6
    tp = 1637

    labels = np.array([
        [f"TN\n{tn:,}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp:,}"],
    ])

    colors = np.array([["#d1fae5", "#fee2e2"], ["#fef3c7", "#dbeafe"]])

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=colors[i][j], edgecolor="white", linewidth=3))
            ax.text(j + 0.5, 1.5 - i, labels[i][j], ha="center", va="center",
                    fontsize=18, fontweight="bold", color="#1e293b")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: Normal", "Pred: Fraude"], fontsize=12, fontweight="bold")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Real: Fraude", "Real: Normal"], fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)

    col_chart, col_stats = st.columns([3, 2])
    with col_chart:
        st.pyplot(fig)
    with col_stats:
        st.markdown("#### Resumen")
        total_fraudes = tp + fn
        st.markdown(f"- **Fraudes reales:** {total_fraudes:,}")
        st.markdown(f"- **Fraudes detectados (TP):** {tp:,}")
        st.markdown(f"- **Falsos negativos (FN):** {fn}")
        st.markdown(f"- **Falsos positivos (FP):** {fp}")
        st.markdown(f"- **Tasa de deteccion:** {tp / total_fraudes * 100:.1f}%")
        st.markdown("---")
        st.markdown("#### Interpretacion")
        st.markdown(
            "El modelo detecta el **99.6%** de los fraudes reales con solo "
            "**30** falsas alarmas sobre mas de 552.409 transacciones legitimas."
        )

else:
    st.title("Simulador de Riesgo")
    st.markdown(
        "Ingrese los datos de una transaccion para evaluar su riesgo de fraude en tiempo real."
    )
    st.markdown("---")

    with st.form("form_transaccion"):
        st.markdown("### Datos de la Transaccion")

        col_a, col_b = st.columns(2)

        with col_a:
            transaction_type = st.selectbox(
                "Tipo de Transaccion",
                ["CASH_OUT", "TRANSFER"],
                help="Solo se analizan CASH_OUT y TRANSFER",
            )
            amount = st.number_input(
                "Monto (amount)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
            )
            oldbalanceOrg = st.number_input(
                "Saldo anterior origen (oldbalanceOrg)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
            )
            newbalanceOrig = st.number_input(
                "Saldo nuevo origen (newbalanceOrig)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
            )

        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            oldbalanceDest = st.number_input(
                "Saldo anterior destino (oldbalanceDest)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
            )
            newbalanceDest = st.number_input(
                "Saldo nuevo destino (newbalanceDest)", min_value=0.0, value=0.0, step=1000.0, format="%.2f"
            )

        submitted = st.form_submit_button("Analizar Transaccion")

    if submitted:
        payload = {
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalanceOrg),
            "newbalanceOrig": float(newbalanceOrig),
            "oldbalanceDest": float(oldbalanceDest),
            "newbalanceDest": float(newbalanceDest),
            "type_TRANSFER": transaction_type == "TRANSFER",
        }

        try:
            with st.spinner("Consultando API de prediccion..."):
                response = requests.post(API_URL, json=payload, timeout=15)

            if response.status_code == 422:
                st.error("Error de validacion. Verifique que los datos sean correctos.")
            elif not response.ok:
                detail = response.json().get("detail", response.text)
                st.error(f"Error del servidor ({response.status_code}): {detail}")
            else:
                result = response.json()
                prob = result["fraud_probability"]
                nivel = result["nivel_alerta"]

                st.markdown("---")
                st.markdown("### Resultado del Analisis")

                if nivel == "Improbable":
                    st.success(f"{nivel} - Probabilidad de fraude: {prob:.2f}%")
                elif nivel == "Sospechoso":
                    st.warning(f"{nivel} - Probabilidad de fraude: {prob:.2f}%")
                else:
                    st.error(f"{nivel} - Probabilidad de fraude: {prob:.2f}%")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probabilidad", f"{prob:.2f}%")
                with col2:
                    st.metric("Nivel de Riesgo", nivel)
                with col3:
                    st.metric("Tipo", transaction_type)

                with st.expander("Ver detalles tecnicos"):
                    error_orig = payload["newbalanceOrig"] + payload["amount"] - payload["oldbalanceOrg"]
                    error_dest = payload["oldbalanceDest"] + payload["amount"] - payload["newbalanceDest"]
                    vacia = 1 if payload["amount"] == payload["oldbalanceOrg"] else 0
                    st.markdown("**Features calculados por el backend:**")
                    st.json({
                        "error_balance_orig": round(error_orig, 2),
                        "error_balance_dest": round(error_dest, 2),
                        "vacia_cuenta": vacia,
                        "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0,
                    })

        except requests.exceptions.ConnectionError:
            st.error(
                "No se pudo conectar con el backend. "
                "Asegurese de que la API este corriendo en `localhost:8000`."
            )
        except requests.exceptions.Timeout:
            st.error("La solicitud excedio el tiempo de espera.")
        except Exception as e:
            st.error(f"Error inesperado: {e}")
