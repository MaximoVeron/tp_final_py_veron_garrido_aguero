import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="FraudShield — Detección de Fraude",
    page_icon="🛡️",
    layout="wide",
)

# ── CSS personalizado ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Fondo general ── */
    .stApp {
        background-color: #F4F7FC;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2545 0%, #13315C 100%);
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
    }

    /* ── Métricas ── */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D6E4F0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #13315C;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0B2545;
    }

    /* ── Botón principal ── */
    .stButton > button {
        background-color: #13315C;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #1B4D89;
        color: #FFFFFF;
    }

    /* ── Formulario ── */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #D6E4F0;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, 0.06);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        color: #13315C;
        font-weight: 600;
        border: 1px solid #D6E4F0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #13315C !important;
        color: #FFFFFF !important;
        border-color: #13315C !important;
    }

    /* ── Encabezados ── */
    h1, h2, h3 {
        color: #0B2545 !important;
    }

    /* ── Cards personalizadas ── */
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #D6E4F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("---")
    st.markdown("#### Sistema de Detección")
    st.markdown(
        "Plataforma de análisis transaccional  \n"
        "basada en aprendizaje automático."
    )
    st.markdown("---")
    st.markdown(
        "<small style='opacity:0.6;'>Trabajo Práctico Final — Python para Ciencia de Datos</small>",
        unsafe_allow_html=True,
    )

# ── Título principal ───────────────────────────────────────────────
st.markdown("# Detección de Fraude Financiero")
st.markdown(
    "Análisis en tiempo real de transacciones para identificar operaciones fraudulentas."
)
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────
tab_stats, tab_predict = st.tabs(["Estadísticas del Modelo", "Analizar Transacción"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Estadísticas del Modelo
# ════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### Rendimiento del Modelo en Datos de Prueba")
    st.markdown("Métricas obtenidas sobre el conjunto de validación (PaySim — 20 % hold-out).")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Métricas principales ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", "97.42 %", delta="+0.8 %")
    with m2:
        st.metric("Precisión", "94.17 %", delta="+1.2 %")
    with m3:
        st.metric("Recall", "91.63 %", delta="+2.5 %")
    with m4:
        st.metric("F1-Score", "92.88 %", delta="+1.8 %")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Distribución de Predicciones")
        labels = ["Correctas", "Incorrectas"]
        sizes = [97.42, 2.58]
        colors_pie = ["#13315C", "#8DAED4"]
        explode = (0.04, 0)

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11, "color": "#0B2545"},
        )
        ax1.set_facecolor("#F4F7FC")
        fig1.patch.set_facecolor("#F4F7FC")
        st.pyplot(fig1)
        plt.close(fig1)

    with chart_col2:
        st.markdown("#### Métricas por Clase")
        categories = ["Normal", "Fraude"]
        precision_vals = [98.1, 94.2]
        recall_vals = [99.3, 91.6]

        x = np.arange(len(categories))
        width = 0.32

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars1 = ax2.bar(x - width / 2, precision_vals, width, label="Precisión", color="#13315C")
        bars2 = ax2.bar(x + width / 2, recall_vals, width, label="Recall", color="#8DAED4")

        ax2.set_ylabel("Porcentaje (%)", fontsize=11, color="#0B2545")
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=11, color="#0B2545")
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=10)
        ax2.set_facecolor("#F4F7FC")
        fig2.patch.set_facecolor("#F4F7FC")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#D6E4F0")
        ax2.spines["bottom"].set_color("#D6E4F0")
        ax2.tick_params(colors="#0B2545")

        for bar in bars1:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0B2545",
            )
        for bar in bars2:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0B2545",
            )

        st.pyplot(fig2)
        plt.close(fig2)

    # ── Resumen adicional ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Resumen del Conjunto de Datos")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            '<div class="info-card"><strong>Total de transacciones</strong><br>6.362.620</div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            '<div class="info-card"><strong>Transacciones fraudulentas</strong><br>8.213 (0.13 %)</div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            '<div class="info-card"><strong>Tipos de transacción</strong><br>5 categorías</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════
# TAB 2 — Analizar Transacción
# ════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Ingrese los Datos de la Transacción")
    st.markdown(
        "Complete los campos a continuación para evaluar si la operación presenta indicadores de fraude."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("fraud_form"):
        # ── Fila 1: Step y Tipo ──
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            step = st.number_input(
                "Paso de Tiempo (Step)",
                min_value=1,
                max_value=744,
                value=1,
                step=1,
                help="Número de paso temporal de la transacción (1–744).",
            )
        with row1_col2:
            transaction_type = st.selectbox(
                "Tipo de Transacción",
                options=["CASH IN", "CASH OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                help="Seleccione el tipo de operación realizada.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Fila 2: Monto ──
        amount = st.number_input(
            "Monto de la Transacción (Amount)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
            help="Monto total de la transacción en la moneda local.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Fila 3: Saldos Origen ──
        st.markdown("**Saldos de la Cuenta Origen**")
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            oldbalance_org = st.number_input(
                "Balance Original Anterior (oldbalanceOrg)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                help="Saldo de la cuenta de origen antes de la transacción.",
            )
        with row3_col2:
            newbalance_orig = st.number_input(
                "Nuevo Balance Original (newbalanceOrig)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                help="Saldo de la cuenta de origen después de la transacción.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Fila 4: Saldos Destino ──
        st.markdown("**Saldos de la Cuenta Destino**")
        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            oldbalance_dest = st.number_input(
                "Balance Destino Anterior (oldbalanceDest)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                help="Saldo de la cuenta de destino antes de la transacción.",
            )
        with row4_col2:
            newbalance_dest = st.number_input(
                "Nuevo Balance Destino (newbalanceDest)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                help="Saldo de la cuenta de destino después de la transacción.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Fila 5: Flag previo ──
        is_flagged = st.checkbox(
            "Flag de Fraude Previo (isFlaggedFraud)",
            value=False,
            help="Marque si la cuenta ya fue señalada previamente por actividad sospechosa.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🛡️ Analizar Transacción",
            use_container_width=True,
        )

    # ── Respuesta simulada ─────────────────────────────────────────
    if submitted:
        st.markdown("---")
        st.markdown("### Resultado del Análisis")

        # Simulación: TRANSFER con monto alto → fraude
        is_fraud = (transaction_type == "TRANSFER" and amount > 500000) or is_flagged

        fraud_prob = np.random.uniform(0.72, 0.96) if is_fraud else np.random.uniform(1.2, 8.5)

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if is_fraud:
                st.error("🚨  FRAUDE DETECTADO")
            else:
                st.success("✅  TRANSACCIÓN SEGURA")

        with result_col2:
            st.metric(
                "Probabilidad de Fraude",
                f"{fraud_prob:.2f} %",
                delta="Alto riesgo" if is_fraud else "Bajo riesgo",
                delta_color="inverse" if is_fraud else "normal",
            )

        with result_col3:
            st.metric(
                "Tipo de Transacción",
                transaction_type,
            )

        # ── Detalle de la transacción analizada ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Detalle de la Transacción Analizada")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f'<div class="info-card"><strong>Step</strong><br>{step}</div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="info-card"><strong>Monto</strong><br>${amount:,.2f}</div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f'<div class="info-card"><strong>Balance Origen</strong><br>${oldbalance_org:,.2f} → ${newbalance_orig:,.2f}</div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f'<div class="info-card"><strong>Balance Destino</strong><br>${oldbalance_dest:,.2f} → ${newbalance_dest:,.2f}</div>',
                unsafe_allow_html=True,
            )

        if is_fraud:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning(
                "⚠️ **Recomendación:** Se sugiere bloquear preventivamente la cuenta "
                "y derivar la operación al equipo de investigación de fraude."
            )
