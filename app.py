import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# PALETTE GRAFICI
# -----------------------------
px.defaults.color_discrete_sequence = [
    "#1F4FD8",  # blu aziendale
    "#0EA5E9",  # azzurro
    "#10B981",  # verde soft
    "#F59E0B",  # ambra
    "#EF4444",  # rosso
]

# -----------------------------
# COLOR SCALE 
# -----------------------------

SCALE_TAB1 = px.colors.sequential.Blues     
SCALE_TAB2 = px.colors.sequential.Greens    
SCALE_TAB3 = px.colors.sequential.Purples   
SCALE_TAB4 = px.colors.sequential.Oranges   
MONEY_FMT = ",.0f"

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

# -----------------------------
# STILE GLOBALE CSS
# -----------------------------
st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 2.5rem; }

    html, body, [data-testid="stApp"] {
        background-color: #FFFFFF;
        color: #0F172A;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    h1 { font-size: 2.0rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.35rem; margin-top: 1.0rem; }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #475569;
    }

    button[role="tab"] { font-size: 0.95rem; padding: 0.55rem 1rem; }

    div[data-testid="stPlotlyChart"] { margin-top: 0.25rem; margin-bottom: 0.9rem; }

    section[data-testid="stSidebar"] { width: 300px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FUNZIONE STILE PLOTLY 
# -----------------------------
def style_plotly(fig, title: str | None = None, x_title: str | None = None, y_title: str | None = None):
    fig.update_layout(
        title=title,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=14, color="#0F172A"),
        title_font=dict(size=22, color="#0F172A"),
        margin=dict(l=55, r=35, t=70, b=60),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        legend_title_text="",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.10)",
        zeroline=False,
        showline=True,
        linecolor="rgba(15, 23, 42, 0.25)",
        title=x_title,
        ticks="outside",
        ticklen=6,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.10)",
        zeroline=False,
        showline=True,
        linecolor="rgba(15, 23, 42, 0.25)",
        title=y_title,
        ticks="outside",
        ticklen=6,
    )

    x_type = fig.layout.xaxis.type
    y_type = fig.layout.yaxis.type

    if x_type not in ("category", "multicategory"):
        fig.update_xaxes(tickformat=MONEY_FMT)

    if y_type not in ("category", "multicategory"):
        fig.update_yaxes(tickformat=MONEY_FMT)

    fig.update_traces(hovertemplate="%{y}: %{x:,.0f}<extra></extra>", selector=dict(type="bar"))

    return fig


@st.cache_data(show_spinner=True, ttl=3600)
def load_data(path1: str, path2: str) -> pd.DataFrame:
    usecols = [
        "date", "store_nbr", "sales", "onpromotion", "transactions",
        "family", "state", "holiday_type", "store_type"
    ]

    dtypes = {
        "store_nbr": "Int64",
        "sales": "float32",
        "onpromotion": "Int64",
        "transactions": "float32",
        "family": "category",
        "state": "category",
        "holiday_type": "category",
        "store_type": "category",
    }

    df1 = pd.read_csv(path1, compression="gzip", usecols=usecols, dtype=dtypes)
    df2 = pd.read_csv(path2, compression="gzip", usecols=usecols, dtype=dtypes)

    df = pd.concat([df1, df2], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["year"] = df["date"].dt.year.astype("Int16")
    df["month"] = df["date"].dt.month.astype("Int8")
    df["week"] = df["date"].dt.isocalendar().week.astype("Int16")
    df["day_of_week"] = df["date"].dt.day_name().astype("category")

    df["sales"] = df["sales"].fillna(0)
    df["onpromotion"] = df["onpromotion"].fillna(0)

    return df

@st.cache_data(ttl=3600)
def tab1_top_products(df):
    return (
        df.groupby("family", as_index=False)["sales"]
          .sum()
          .sort_values("sales", ascending=False)
          .head(10)
    )

@st.cache_data(ttl=3600)
def tab1_store_sales(df):
    return (
        df.groupby("store_nbr", as_index=False)["sales"]
          .sum()
          .dropna(subset=["store_nbr"])
    )

@st.cache_data(ttl=3600)
def tab1_promo_sales_store(df):
    return (
        df.loc[df["onpromotion"] > 0]
          .groupby("store_nbr", as_index=False)["sales"]
          .sum()
          .sort_values("sales", ascending=False)
          .head(10)
    )

@st.cache_data(ttl=3600)
def tab1_seasonality(df):
    # identico a quello che fai ora: mean su sales per day_of_week
    dow = df.groupby("day_of_week", as_index=False)["sales"].mean()

    # identico: prima somma per (year, week), poi media per week
    weekly = df.groupby(["year", "week"], as_index=False)["sales"].sum()
    weekly_mean = weekly.groupby("week", as_index=False)["sales"].mean().sort_values("week")

    # identico: prima somma per (year, month), poi media per month
    monthly = df.groupby(["year", "month"], as_index=False)["sales"].sum()
    monthly_mean = monthly.groupby("month", as_index=False)["sales"].mean().sort_values("month")

    return dow, weekly_mean, monthly_mean

df = load_data("parte_1.csv.gz", "parte_2.csv.gz")
df_tx = (
    df[["date", "store_nbr", "state", "transactions", "year"]]
    .groupby(["date", "store_nbr", "state", "year"], as_index=False)["transactions"]
    .max()
)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.title("Dashboard de ventas")
st.caption("KPIs y análisis para dirección")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Pestaña 1 - Visión global",
     "Pestaña 2 - Análisis por tienda",
     "Pestaña 3 - Análisis por estado",
     "Pestaña 4 - Extra"]
)

# -----------------------------
# PESTAÑA 1 - VISIÓN GLOBAL
# -----------------------------
with tab1:
    SCALE = SCALE_TAB1
    st.subheader("KPIs generales")

    total_stores = int(df["store_nbr"].nunique(dropna=True))
    total_products = int(df["family"].nunique(dropna=True))
    total_states = int(df["state"].nunique(dropna=True))

    df_months = df.dropna(subset=["date"]).assign(ym=df["date"].dt.to_period("M").astype(str))
    total_months = int(df_months["ym"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Número total de tiendas", f"{total_stores:,}")
    c2.metric("Número total de productos (family)", f"{total_products:,}")
    c3.metric("Número de estados", f"{total_states:,}")
    c4.metric("Número de meses con datos", f"{total_months:,}")

    st.divider()
    st.subheader("Análisis y distribuciones")

    # Top 10 productos por ventas totales
    prod_sales = tab1_top_products(df)

    fig_top_prod = px.bar(
        prod_sales,
        x="sales",
        y="family",
        orientation="h",
        color="sales",
        color_continuous_scale=SCALE,
        text_auto=".2s",
        labels={"sales": "Ventas", "family": "Producto (family)"},
    )
    fig_top_prod.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig_top_prod.update_traces(textposition="outside", cliponaxis=False)
    fig_top_prod = style_plotly(fig_top_prod, "Top 10 productos por ventas totales", "Ventas", "Producto (family)")
    st.plotly_chart(fig_top_prod, use_container_width=True)

    # Distribución ventas totales por tienda
    store_sales = tab1_store_sales(df)

    NBINS = 60
    counts, edges = np.histogram(store_sales["sales"].values, bins=NBINS)
    bin_left = edges[:-1]
    bin_right = edges[1:]
    bin_mid = (bin_left + bin_right) / 2

    hist_df = pd.DataFrame({
        "bin_left": bin_left,
        "bin_right": bin_right,
        "bin_mid": bin_mid,
        "count": counts
    })

    fig_dist_store = px.bar(
        hist_df,
        x="bin_mid",
        y="count",
        color="count", 
        color_continuous_scale=SCALE,
        labels={"bin_mid": "Ventas totales por tienda", "count": "Frecuencia"},
    )

    fig_dist_store.update_traces(
        width=(hist_df["bin_right"] - hist_df["bin_left"]),
        hovertemplate=(
            "Rango ventas: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}"
            "<br>Frecuencia: %{y}<extra></extra>"
        ),
        customdata=np.stack([hist_df["bin_left"], hist_df["bin_right"]], axis=1),
    )

    fig_dist_store = style_plotly(
        fig_dist_store,
        "Distribución de ventas totales por tienda",
        "Ventas totales por tienda",
        "Frecuencia"
    )

    fig_dist_store.update_xaxes(tickformat="~s")

    fig_dist_store.update_layout(coloraxis_colorbar=dict(title="Frecuencia"))

    st.plotly_chart(fig_dist_store, use_container_width=True)


    # Top 10 tiendas por ventas en promoción
    promo_sales_store = tab1_promo_sales_store(df).copy()
    promo_sales_store["store_nbr"] = promo_sales_store["store_nbr"].astype(str)
    order = promo_sales_store.sort_values("sales", ascending=True)["store_nbr"].tolist()

    fig_top_promo = px.bar(
        promo_sales_store,
        x="sales",
        y="store_nbr",
        orientation="h",
        color="sales",
        color_continuous_scale=SCALE,
        text_auto=".2s",
        labels={"sales": "Ventas en promoción", "store_nbr": "Tienda (store_nbr)"},
    )
    fig_top_promo.update_traces(textposition="outside", cliponaxis=False)
    fig_top_promo.update_yaxes(type="category")
    fig_top_promo.update_layout(yaxis=dict(categoryorder="array", categoryarray=order))

    fig_top_promo = style_plotly(
        fig_top_promo,
        "Top 10 tiendas por ventas en promoción",
        "Ventas en promoción",
        "Tienda (store_nbr)",
    )
    st.plotly_chart(fig_top_promo, use_container_width=True)

    st.divider()
    st.subheader("Estacionalidad")
    
   
    dow, weekly_mean, monthly_mean = tab1_seasonality(df)

    
    colA, colB = st.columns(2)

   
    # Ventas medias por día de la semana
    dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=DAY_ORDER, ordered=True)
    dow = dow.sort_values("day_of_week")

    fig_dow = px.bar(
        dow,
        x="day_of_week",
        y="sales",
        color="sales",
        color_continuous_scale=SCALE,
        text_auto=".2s",
        labels={"day_of_week": "Día de la semana", "sales": "Ventas medias"},
    )
    fig_dow.update_xaxes(tickformat=None)
    fig_dow.update_traces(textposition="outside", cliponaxis=False)
    fig_dow = style_plotly(fig_dow, "Ventas medias por día de la semana", "Día de la semana", "Ventas medias")
    fig_dow.update_layout(coloraxis_colorbar=dict(title="Ventas"))

    with colA:
        st.plotly_chart(fig_dow, use_container_width=True)
   
    # Ventas medias por semana del año

    fig_week = px.line(
        weekly_mean,
        x="week",
        y="sales",
        markers=True,
        labels={"week": "Semana (ISO)", "sales": "Ventas medias"},
    )
    fig_week.update_traces(hovertemplate="Semana %{x}<br>Ventas %{y:,.0f}<extra></extra>")
    fig_week = style_plotly(fig_week, "Ventas medias por semana del año", "Semana (ISO)", "Ventas medias")
    with colB:
        st.plotly_chart(fig_week, use_container_width=True)

    # Ventas medias por mes 
    
    fig_month = px.line(
        monthly_mean,
        x="month",
        y="sales",
        markers=True,
        labels={"month": "Mes", "sales": "Ventas medias"},
    )
    fig_month.update_traces(hovertemplate="Mes %{x}<br>Ventas %{y:,.0f}<extra></extra>")
    fig_month = style_plotly(fig_month, "Ventas medias por mes", "Mes", "Ventas medias")
    st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------
# PESTAÑA 2 - ANÁLISIS POR TIENDA
# -----------------------------
with tab2:
    SCALE = SCALE_TAB2
    st.subheader("Análisis por tienda")

    stores = df["store_nbr"].dropna().sort_values().unique()
    store_sel = st.selectbox("Selecciona tienda (store_nbr)", stores)

    df_s = df[df["store_nbr"] == store_sel]

    sales_year = df_s.groupby("year", as_index=False)["sales"].sum().sort_values("year")
    sales_year["year"] = sales_year["year"].astype(str)
    fig_store_year = px.bar(sales_year, x="year", y="sales", color="sales", color_continuous_scale=SCALE)
    fig_store_year.update_xaxes(tickformat=None)
    fig_store_year = style_plotly(fig_store_year, f"Tienda {store_sel} - ventas totales por año", "Año", "Ventas")

    fig_store_year.update_xaxes(type="category")
    fig_store_year.update_xaxes(tickformat=None)

    st.plotly_chart(fig_store_year, use_container_width=True)

    # Totale "prodotti venduti" = quantità venduta (somma sales)
    total_productos_vendidos = float(df_s["sales"].sum())

    # Totale "prodotti venduti in promozione" = quantità venduta quando onpromotion > 0
    total_productos_vendidos_promo = float(df_s.loc[df_s["onpromotion"] > 0, "sales"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Número total de productos vendidos", f"{total_productos_vendidos:,.0f}")
    c2.metric("Número total de productos vendidos en promoción", f"{total_productos_vendidos_promo:,.0f}")


# -----------------------------
# PESTAÑA 3 - ANÁLISIS POR ESTADO
# -----------------------------
with tab3:
    SCALE = SCALE_TAB3
    st.subheader("Análisis por estado")

    states = df["state"].dropna().sort_values().unique()
    state_sel = st.selectbox("Selecciona estado", states)

    df_state = df[df["state"] == state_sel]
    df_state_tx = df_tx[df_tx["state"] == state_sel]

    tx_year = df_state_tx.groupby("year", as_index=False)["transactions"].sum().sort_values("year")
    tx_year["year"] = tx_year["year"].astype(str)
    fig_tx = px.bar(tx_year, x="year", y="transactions", color="transactions", color_continuous_scale=SCALE)
    fig_tx.update_xaxes(tickformat=None)
    fig_tx = style_plotly(
    fig_tx,
    f"{state_sel} - transacciones totales por año",
    "Año",
    "Transacciones"
    )

    fig_tx.update_xaxes(type="category")
    fig_tx.update_xaxes(tickformat=None)

    st.plotly_chart(fig_tx, use_container_width=True)


    rank_store = (
        df_state.groupby("store_nbr", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(15)
        .copy()
    )

    rank_store["store_nbr"] = rank_store["store_nbr"].astype(str)
    order = rank_store["store_nbr"].tolist()

    fig_rank = px.bar(
        rank_store,
        x="sales",
        y="store_nbr",
        orientation="h",
        color="sales",
        color_continuous_scale=SCALE,
        text_auto=".2s",
        category_orders={"store_nbr": order},
    )
    fig_rank.update_yaxes(type="category")
    fig_rank.update_traces(textposition="outside", cliponaxis=False)

    fig_rank = style_plotly(
        fig_rank,
        f"{state_sel} - ranking de tiendas por ventas",
        "Ventas",
        "Tienda (store_nbr)",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    if len(rank_store) > 0:
        top_store = str(rank_store.iloc[0]["store_nbr"])
        top_prod = (
            df_state[df_state["store_nbr"].astype(str) == top_store]
            .groupby("family", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
            .head(1)
        )
        if len(top_prod) > 0:
            st.info(
                f"En el estado {state_sel}, la tienda con más ventas es {top_store} "
                f"y el producto más vendido es {top_prod.iloc[0]['family']}."
            )

# -----------------------------
# PESTAÑA 4 - EXTRA
# -----------------------------
with tab4:
    SCALE = SCALE_TAB4
    st.subheader("Análisis adicional")
        
    df_flag = df[["sales", "onpromotion"]].copy()
    df_flag["promo_flag"] = np.where(df_flag["onpromotion"] > 0, "Promoción", "Sin promoción")
    sample = df_flag.sample(min(20_000, len(df_flag)), random_state=7)

    sample_box = sample[sample["sales"] > 0]

    fig_box = px.box(
        sample_box,
        x="promo_flag",
        y="sales",
        points=False,
        color="promo_flag",
        color_discrete_map={
            "Promoción": "#EA580C",      
            "Sin promoción": "#FDBA74",  
        }
    )
    fig_box.update_xaxes(tickformat=None)
    fig_box.update_yaxes(type="log")

    fig_box = style_plotly(
        fig_box,
        "Ventas: promoción vs sin promoción (escala logarítmica)",
        "Grupo",
        "Ventas (log)"
    )

    fig_box.update_layout(showlegend=False)

    st.plotly_chart(fig_box, use_container_width=True)

    st.caption(
        "Se utiliza escala logarítmica para facilitar la comparación debido a la presencia de valores extremos."
    )

    col1, col2 = st.columns(2)

    with col1:
        if df["holiday_type"].notna().any():
            holiday = (
                df.groupby("holiday_type", as_index=False)["sales"]
                .mean()
                .sort_values("sales", ascending=False)
            )
            fig_holiday = px.bar(
                holiday, x="sales", y="holiday_type", orientation="h",
                color="sales", color_continuous_scale=SCALE
            )
            fig_holiday.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig_holiday = style_plotly(fig_holiday, "Ventas medias por tipo de festivo", "Ventas medias", "Tipo de festivo")
            st.plotly_chart(fig_holiday, use_container_width=True)

    with col2:
        if df["store_type"].notna().any():
            stype = (
                df.groupby("store_type", as_index=False)["sales"]
                .sum()
                .sort_values("sales", ascending=False)
            )
            fig_stype = px.bar(
                stype, x="store_type", y="sales",
                color="sales", color_continuous_scale=SCALE
            )
            fig_stype.update_xaxes(tickformat=None)
            fig_stype = style_plotly(fig_stype, "Ventas totales por tipo de tienda", "Tipo de tienda", "Ventas")
            st.plotly_chart(fig_stype, use_container_width=True)

   
    # Heatmap: ventas medias por mes y día de la semana
    heat = df.groupby(["month", "day_of_week"], as_index=False)["sales"].mean()

    DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat["day_of_week"] = pd.Categorical(heat["day_of_week"], categories=DAY_ORDER, ordered=True)

    heat_pivot = heat.pivot(index="day_of_week", columns="month", values="sales")

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=SCALE,
        labels=dict(x="Mes", y="Día de la semana", color="Ventas medias"),
        aspect="auto"
    )

    fig_heat = style_plotly(
        fig_heat,
        "Mapa de calor: ventas medias por mes y día de la semana",
        "Mes",
        "Día de la semana"
    )

    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("El mapa de calor permite identificar patrones temporales de ventas y días con mayor potencial comercial.")
