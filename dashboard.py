"""
Dashboard EDA — informe de ventas
Funciona con informe_limpio.xlsx (Wide_Maestra) O con informe.xlsx original.
Ejecutar: python dashboard.py
Abrir:    http://127.0.0.1:8050
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ─── LOCATE FILE ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def find_xlsx():
    for name in ('informe_limpio.xlsx', 'informe.xlsx'):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró informe_limpio.xlsx ni informe.xlsx en " + BASE)

XLSX = find_xlsx()
print(f"✅ Cargando: {XLSX}")

# ─── MONTH DETECTION & NORMALIZATION ─────────────────────────────────────────
MONTHS_CLEAN = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
                'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

MONTH_ALIASES = {m.lower().strip(): m for m in MONTHS_CLEAN}
MONTH_ALIASES.update({'enero':'Enero','febrero':'Febrero','marzo':'Marzo',
                       'abril':'Abril','mayo':'Mayo','junio':'Junio',
                       'julio':'Julio','agosto':'Agosto','septiembre':'Septiembre',
                       'octubre':'Octubre','noviembre':'Noviembre','diciembre':'Diciembre'})

def normalize_month_cols(df):
    rename = {}
    for c in df.columns:
        key = str(c).lower().strip()
        if key in MONTH_ALIASES:
            rename[c] = MONTH_ALIASES[key]
    return df.rename(columns=rename)

# ─── LOAD & ENRICH ───────────────────────────────────────────────────────────
def load_and_enrich():
    # Try Wide_Maestra first, fall back to first sheet
    xl = pd.ExcelFile(XLSX)
    sheet = 'Wide_Maestra' if 'Wide_Maestra' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(XLSX, sheet_name=sheet)
    print(f"   Hoja: '{sheet}' — {df.shape[0]} filas × {df.shape[1]} columnas")

    # Normalize month column names (handles spaces, caps)
    df = normalize_month_cols(df)

    # Normalize derived column display names → internal names
    display_to_internal = {
        'Total Anual':      'Total_anual',
        'Promedio Mensual': 'Promedio_mensual',
        'Máx Mensual':      'Max_mensual',
        'Max Mensual':      'Max_mensual',
        'Meses Activos':    'Meses_activos',
        'CV (σ/μ)':         'CV',
        'Mes Pico':         'Mes_pico',
        'Negativos':        'Negativos_count',
        'Activo':           'Flag_activo',
        'Tipo Cliente':     'Tipo_cliente',
    }
    df.rename(columns=display_to_internal, inplace=True)

    # Find which month columns are present
    months = [m for m in MONTHS_CLEAN if m in df.columns]
    if not months:
        raise ValueError("No se encontraron columnas de meses en el archivo.")
    print(f"   Meses detectados: {len(months)}")

    # Force numeric on month cols
    for m in months:
        df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)

    nums = df[months]

    # Compute derived columns if missing
    if 'Total_anual' not in df.columns:
        df['Total_anual'] = nums.sum(axis=1)
    if 'Promedio_mensual' not in df.columns:
        df['Promedio_mensual'] = nums.mean(axis=1).round(0)
    if 'Max_mensual' not in df.columns:
        df['Max_mensual'] = nums.max(axis=1)
    if 'Meses_activos' not in df.columns:
        df['Meses_activos'] = (nums > 0).sum(axis=1)
    if 'Negativos_count' not in df.columns:
        df['Negativos_count'] = (nums < 0).sum(axis=1)
    if 'Flag_activo' not in df.columns:
        df['Flag_activo'] = (df['Total_anual'] > 0).astype(int)
    if 'CV' not in df.columns:
        df['CV'] = nums.apply(lambda r: round(r.std()/r.mean(), 4) if r.mean() != 0 else None, axis=1)
    if 'Mes_pico' not in df.columns:
        df['Mes_pico'] = nums.idxmax(axis=1)
        df.loc[df['Flag_activo'] == 0, 'Mes_pico'] = None
    if 'Tipo_cliente' not in df.columns:
        corp = r'S\.A\.S\.|S\.A\.|LTDA|SAS|COMERCIALIZADORA|REPRESENTACIONES|GROUP|DROGAS|IMPORTADOS|CONCENTRADOS'
        df['Tipo_cliente'] = df['Nombre'].apply(
            lambda x: 'Empresa' if re.search(corp, str(x), re.I) else 'Persona Natural')
    if 'Ciudad' not in df.columns:
        df['Ciudad'] = 'Sin ciudad'

    # Clean city names
    city_fixes = {'cali':'Cali','garzón':'Garzon','garzon':'Garzon',
                  'itagui ':'Itagui','jamundi':'Jamundi','la virginia':'La Virginia'}
    df['Ciudad'] = df['Ciudad'].astype(str).str.strip()
    df['Ciudad'] = df['Ciudad'].apply(lambda x: city_fixes.get(str(x).lower(), str(x)) if pd.notna(x) else 'Sin ciudad')

    # Force numeric on derived cols
    for col in ['Total_anual','Promedio_mensual','Max_mensual','Meses_activos','CV']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Flag_activo'] = df['Flag_activo'].fillna(0).astype(int)

    print(f"   Activas: {df['Flag_activo'].sum()} | Total: {len(df)}")
    return df, months

def build_long(df, months):
    dl = df[['Nombre','Ciudad','Tipo_cliente','Flag_activo'] + months].melt(
        id_vars=['Nombre','Ciudad','Tipo_cliente','Flag_activo'],
        value_vars=months, var_name='Mes', value_name='Ventas')
    mo = {m: i+1 for i, m in enumerate(months)}
    dl['Mes_num'] = dl['Mes'].map(mo)
    dl['Ventas'] = pd.to_numeric(dl['Ventas'], errors='coerce').fillna(0)
    return dl.sort_values(['Nombre','Mes_num']).reset_index(drop=True)

df_wide, MONTHS = load_and_enrich()
df_long = build_long(df_wide, MONTHS)

ALL_CITIES = sorted(df_wide['Ciudad'].dropna().unique())
ALL_TYPES  = sorted(df_wide['Tipo_cliente'].dropna().unique())
NUMERIC_COLS = MONTHS + ['Total_anual','Promedio_mensual','Max_mensual','Meses_activos','CV']

# ─── STYLE ───────────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#2563EB', 'secondary': '#7C3AED',
    'success':   '#059669', 'warning':   '#D97706',
    'danger':    '#DC2626', 'bg':        '#F8FAFC',
    'card':      '#FFFFFF', 'border':    '#E2E8F0',
    'text':      '#1E293B', 'muted':     '#64748B',
}
PALETTE = px.colors.qualitative.Set2

def fmt_cop(v):
    if pd.isna(v): return "—"
    if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def card_style(border_color=None):
    s = {'background':COLORS['card'],'borderRadius':'12px','padding':'16px 20px',
         'boxShadow':'0 1px 4px rgba(0,0,0,.08)','border':f'1px solid {COLORS["border"]}','height':'100%'}
    if border_color: s['borderTop'] = f'4px solid {border_color}'
    return s

def fig_layout(fig, title='', height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS['text']), x=0),
        height=height, plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['card'],
        font=dict(family='Inter, Arial, sans-serif', color=COLORS['text'], size=11),
        margin=dict(l=50,r=30,t=50,b=40), legend=dict(bgcolor='rgba(0,0,0,0)',borderwidth=0))
    fig.update_xaxes(gridcolor='#E2E8F0', gridwidth=1, zerolinecolor='#CBD5E1')
    fig.update_yaxes(gridcolor='#E2E8F0', gridwidth=1, zerolinecolor='#CBD5E1')
    return fig

# ─── APP ─────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'],
    title='Dashboard EDA — Ventas', suppress_callback_exceptions=True)

def filter_df(cities, tipos, include_zero):
    d = df_wide.copy()
    if cities: d = d[d['Ciudad'].isin(cities)]
    if tipos:  d = d[d['Tipo_cliente'].isin(tipos)]
    if not include_zero: d = d[d['Flag_activo'] == 1]
    return d

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.Div("📊", style={'fontSize':'28px'}),
        html.H5("MACROMAKERS", style={'margin':'0','fontWeight':'800','color':'white','letterSpacing':'2px'}),
        html.Small("Talento Tech IA — Nivel Explorador", style={'color':'#94A3B8','fontSize':'10px'}),
    ], style={'marginBottom':'24px'}),
    html.Label("Ciudad", style={'fontWeight':'600','fontSize':'11px','color':'#94A3B8','textTransform':'uppercase','letterSpacing':'.5px'}),
    dcc.Dropdown(id='filter-ciudad', options=[{'label':c,'value':c} for c in ALL_CITIES],
        multi=True, placeholder='Todas...', style={'marginBottom':'16px','fontSize':'12px'}),
    html.Label("Tipo Cliente", style={'fontWeight':'600','fontSize':'11px','color':'#94A3B8','textTransform':'uppercase','letterSpacing':'.5px'}),
    dcc.Dropdown(id='filter-tipo', options=[{'label':t,'value':t} for t in ALL_TYPES],
        multi=True, placeholder='Todos...', style={'marginBottom':'16px','fontSize':'12px'}),
    html.Label("Sin ventas", style={'fontWeight':'600','fontSize':'11px','color':'#94A3B8','textTransform':'uppercase','letterSpacing':'.5px'}),
    dbc.Switch(id='filter-zero', label='Incluir inactivas', value=False, style={'marginBottom':'24px','fontSize':'12px','color':'white'}),
    html.Hr(style={'borderColor':'#334155'}),
    html.Div(id='sidebar-stats'),
], className='sidebar', style={'width':'210px','minHeight':'100vh','background':'#1E293B','padding':'24px 14px',
          'position':'fixed','top':0,'left':0,'zIndex':1000,'overflowY':'auto'})

# ─── TABS ────────────────────────────────────────────────────────────────────
ts = {'padding':'8px 14px','fontWeight':'500','fontSize':'13px','color':COLORS['muted']}
ss = {**ts,'color':COLORS['primary'],'borderBottom':f'2px solid {COLORS["primary"]}'}

tabs = dcc.Tabs(id='main-tabs', value='tab-resumen', children=[
    dcc.Tab(label='📌 Resumen',            value='tab-resumen',    style=ts, selected_style=ss),
    dcc.Tab(label='🔍 Explorador X/Y',     value='tab-explorador', style=ts, selected_style=ss),
    dcc.Tab(label='📦 Boxplot',            value='tab-boxplot',    style=ts, selected_style=ss),
    dcc.Tab(label='🌡️ Heatmap & Pearson', value='tab-heatmap',    style=ts, selected_style=ss),
    dcc.Tab(label='📈 Regresión',          value='tab-regresion',  style=ts, selected_style=ss),
    dcc.Tab(label='🔵 K-Means',            value='tab-kmeans',     style=ts, selected_style=ss),
], style={'background':COLORS['card'],'borderBottom':f'1px solid {COLORS["border"]}','paddingLeft':'8px'})

app.layout = html.Div([
    html.Div([
        html.Span("MACROMAKERS"),
        html.Span("Talento Tech IA — Nivel Explorador", style={'fontSize':'11px','fontWeight':'400','color':'#94A3B8','marginLeft':'10px','letterSpacing':'0.5px'}),
    ], className='mobile-topbar'),
    sidebar,
    html.Div([
        tabs,
        html.Div(id='tab-content', className='tab-content-inner', style={'padding':'24px','minHeight':'80vh'}),
    ], className='main-content', style={'marginLeft':'210px','background':COLORS['bg'],'minHeight':'100vh'}),
], style={'fontFamily':'Inter, Arial, sans-serif'})

# ─── SIDEBAR STATS ───────────────────────────────────────────────────────────
@app.callback(Output('sidebar-stats','children'),
              Input('filter-ciudad','value'), Input('filter-tipo','value'), Input('filter-zero','value'))
def sidebar_stats(cities, tipos, inc_zero):
    d = filter_df(cities, tipos, inc_zero)
    return [
        html.Div(f"Empresas: {len(d)}",            style={'color':'#94A3B8','fontSize':'12px','marginBottom':'3px'}),
        html.Div(f"Activas: {d['Flag_activo'].sum()}",style={'color':'#94A3B8','fontSize':'12px','marginBottom':'3px'}),
        html.Div(f"Ciudades: {d['Ciudad'].nunique()}",style={'color':'#94A3B8','fontSize':'12px','marginBottom':'8px'}),
        html.Div(fmt_cop(d[d['Flag_activo']==1]['Total_anual'].sum()),
                 style={'color':'#38BDF8','fontWeight':'700','fontSize':'16px'}),
        html.Div("total ventas", style={'color':'#64748B','fontSize':'10px'}),
    ]

# ─── TAB ROUTER ──────────────────────────────────────────────────────────────
@app.callback(Output('tab-content','children'),
              Input('main-tabs','value'),
              Input('filter-ciudad','value'), Input('filter-tipo','value'), Input('filter-zero','value'))
def render_tab(tab, cities, tipos, inc_zero):
    d  = filter_df(cities, tipos, inc_zero)
    da = d[d['Flag_activo']==1]
    if tab == 'tab-resumen':    return tab_resumen(d, da)
    if tab == 'tab-explorador': return tab_explorador()
    if tab == 'tab-boxplot':    return tab_boxplot(da)
    if tab == 'tab-heatmap':    return tab_heatmap(da)
    if tab == 'tab-regresion':  return tab_regresion()
    if tab == 'tab-kmeans':     return tab_kmeans()
    return html.Div()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
def kpi(title, value, sub, color):
    return dbc.Col(html.Div([
        html.Div(title, style={'fontSize':'10px','fontWeight':'600','color':COLORS['muted'],'textTransform':'uppercase','letterSpacing':'.5px','marginBottom':'4px'}),
        html.Div(value, style={'fontSize':'24px','fontWeight':'700','color':color,'lineHeight':'1.1'}),
        html.Div(sub,   style={'fontSize':'10px','color':COLORS['muted'],'marginTop':'3px'}),
    ], style=card_style(color)), xs=6, md=3, style={'padding':'6px'})

def tab_resumen(d, da):
    total   = da['Total_anual'].sum()
    top_e   = da.nlargest(1,'Total_anual')
    top_c   = da.groupby('Ciudad')['Total_anual'].sum().idxmax() if not da.empty else '—'
    avg_m   = da['Promedio_mensual'].mean()

    kpis = dbc.Row([
        kpi('Ventas Totales', fmt_cop(total), f'{len(da)} empresas activas', COLORS['primary']),
        kpi('Empresa Top', (top_e['Nombre'].values[0][:22]+'…') if len(top_e) else '—',
            fmt_cop(top_e['Total_anual'].values[0]) if len(top_e) else '—', COLORS['secondary']),
        kpi('Ciudad Líder', top_c, 'mayor volumen', COLORS['success']),
        kpi('Promedio/empresa', fmt_cop(avg_m), 'promedio mensual', COLORS['warning']),
    ], style={'marginBottom':'16px'})

    top10 = da.nlargest(10,'Total_anual')
    f_bar = go.Figure(go.Bar(x=top10['Total_anual'], y=top10['Nombre'].str[:30],
        orientation='h', marker_color=COLORS['primary'],
        text=[fmt_cop(v) for v in top10['Total_anual']], textposition='outside'))
    fig_layout(f_bar,'🏆 Top 10 Empresas',320)
    f_bar.update_layout(yaxis=dict(autorange='reversed'), margin=dict(l=200))

    city_s = da.groupby('Ciudad')['Total_anual'].sum().sort_values(ascending=False)
    top8 = city_s.head(8); otros = city_s.iloc[8:].sum()
    labels = list(top8.index)+(['Otras'] if otros>0 else [])
    vals   = list(top8.values)+([otros] if otros>0 else [])
    f_pie = go.Figure(go.Pie(labels=labels, values=vals, hole=.4,
        marker_colors=PALETTE, textinfo='label+percent', textfont_size=10))
    fig_layout(f_pie,'🗺️ Por Ciudad',320)

    monthly = da[MONTHS].mean()
    f_line = go.Figure(go.Scatter(x=MONTHS, y=monthly, mode='lines+markers',
        line=dict(color=COLORS['primary'],width=2.5), marker=dict(size=6),
        fill='tozeroy', fillcolor='rgba(37,99,235,.1)',
        text=[fmt_cop(v) for v in monthly], hovertemplate='%{x}: %{text}<extra></extra>'))
    fig_layout(f_line,'📅 Promedio Mensual (empresas activas)',280)

    tipo_s = da.groupby('Tipo_cliente')['Total_anual'].sum()
    f_tipo = go.Figure(go.Pie(labels=tipo_s.index, values=tipo_s.values, hole=.4,
        marker_colors=[COLORS['primary'],COLORS['secondary']],
        textinfo='label+percent', textfont_size=11))
    fig_layout(f_tipo,'👥 Por Tipo de Cliente',280)

    header = html.Div([
        html.H2("Resúmen Principal", style={
            'fontWeight':'800','letterSpacing':'3px','color':COLORS['text'],
            'marginBottom':'2px','fontSize':'clamp(20px, 4vw, 32px)',
        }),
        html.Div("Análisis Exploratorio de Datos", style={'color':COLORS['muted'],'fontSize':'13px','marginBottom':'20px'}),
    ])

    return html.Div([header, kpis,
        dbc.Row([dbc.Col(dcc.Graph(figure=f_bar, config={'displayModeBar':False}),xs=12, md=7),
                 dbc.Col(dcc.Graph(figure=f_pie, config={'displayModeBar':False}),xs=12, md=5)],style={'marginBottom':'12px'}),
        dbc.Row([dbc.Col(dcc.Graph(figure=f_line,config={'displayModeBar':False}),xs=12, md=8),
                 dbc.Col(dcc.Graph(figure=f_tipo,config={'displayModeBar':False}),xs=12, md=4)])])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORADOR X/Y
# ══════════════════════════════════════════════════════════════════════════════
def tab_explorador():
    opts = [{'label':c,'value':c} for c in NUMERIC_COLS if c in df_wide.columns]
    return html.Div([
        html.H6("Explorador de Variables",style={'fontWeight':'700','marginBottom':'4px'}),
        html.P("Selecciona variables y tipo de gráfico.",style={'color':COLORS['muted'],'fontSize':'12px','marginBottom':'12px'}),
        dbc.Row([
            dbc.Col([html.Label("Eje X",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
                     dcc.Dropdown(id='exp-x',options=opts,value='Meses_activos' if 'Meses_activos' in df_wide.columns else opts[0]['value'],clearable=False,style={'fontSize':'12px'})],width=3),
            dbc.Col([html.Label("Eje Y",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
                     dcc.Dropdown(id='exp-y',options=opts,value='Total_anual',clearable=False,style={'fontSize':'12px'})],width=3),
            dbc.Col([html.Label("Tipo",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
                     dcc.RadioItems(id='exp-tipo',
                         options=[{'label':' Scatter','value':'scatter'},{'label':' Barras','value':'bar'},
                                  {'label':' Histograma+Normal','value':'hist'},{'label':' Línea','value':'line'}],
                         value='scatter', inline=True, labelStyle={'marginRight':'14px'},
                         style={'fontSize':'12px'})],width=6),
        ],style={'background':COLORS['card'],'padding':'16px','borderRadius':'12px','border':f'1px solid {COLORS["border"]}','marginBottom':'16px'}),
        dbc.Row([dbc.Col(dcc.Graph(id='exp-graph',config={'displayModeBar':False}),xs=12, md=8),
                 dbc.Col(dcc.Graph(id='exp-hist', config={'displayModeBar':False}),xs=12, md=4)]),
        html.Div(id='exp-interp',style={'marginTop':'12px'}),
    ])

@app.callback(Output('exp-graph','figure'),Output('exp-hist','figure'),Output('exp-interp','children'),
              Input('main-tabs','value'),Input('exp-x','value'),Input('exp-y','value'),Input('exp-tipo','value'),
              Input('filter-ciudad','value'),Input('filter-tipo','value'),Input('filter-zero','value'))
def cb_explorador(tab, xcol, ycol, tipo, cities, tipos, inc_zero):
    if tab != 'tab-explorador': return go.Figure(), go.Figure(), ''
    if not xcol or not ycol: return go.Figure(), go.Figure(), ''
    da = filter_df(cities, tipos, inc_zero)
    da = da[da['Flag_activo']==1].dropna(subset=[xcol,ycol])

    if tipo == 'scatter':
        fig = px.scatter(da,x=xcol,y=ycol,color='Ciudad',hover_name='Nombre',
                         color_discrete_sequence=PALETTE,opacity=.8)
        if len(da)>2:
            r,pv = stats.pearsonr(da[xcol],da[ycol])
            fig.add_annotation(x=.98,y=.04,xref='paper',yref='paper',
                text=f'r = {r:.3f}  p = {pv:.4f}',showarrow=False,
                bgcolor='white',bordercolor='#ccc',font=dict(size=11))
            interp = f"**Pearson r = {r:.3f}** — correlación {'positiva' if r>0 else 'negativa'} {'fuerte' if abs(r)>.6 else 'moderada' if abs(r)>.3 else 'débil'}. p = {pv:.4f} ({'significativo' if pv<.05 else 'no significativo'} al 5%)."
        else: interp = "Pocas observaciones para calcular correlación."
    elif tipo == 'bar':
        top = da.nlargest(20,ycol)
        fig = px.bar(top,x=xcol,y=ycol,color='Ciudad',hover_name='Nombre',color_discrete_sequence=PALETTE)
        interp = f"Top 20 empresas por **{ycol}**."
    elif tipo == 'hist':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=da[xcol],nbinsx=20,marker_color=COLORS['primary'],opacity=.75,name=xcol))
        mu,sig = da[xcol].mean(),da[xcol].std()
        if sig>0:
            xr = np.linspace(da[xcol].min(),da[xcol].max(),200)
            yn = stats.norm.pdf(xr,mu,sig)*len(da)*(da[xcol].max()-da[xcol].min())/20
            fig.add_trace(go.Scatter(x=xr,y=yn,mode='lines',name='Curva normal',line=dict(color=COLORS['danger'],width=2.5)))
        skew = da[xcol].skew(); kurt = da[xcol].kurtosis()
        interp = f"μ={fmt_cop(mu)}, σ={fmt_cop(sig)}. Asimetría={skew:.2f} ({'positiva →cola derecha' if skew>0 else 'negativa →cola izquierda'}), Curtosis={kurt:.2f}."
    else:
        monthly = da[MONTHS].mean()
        fig = go.Figure(go.Scatter(x=MONTHS,y=monthly,mode='lines+markers',
            line=dict(color=COLORS['primary'],width=2.5),marker=dict(size=7),
            fill='tozeroy',fillcolor='rgba(37,99,235,.1)'))
        interp = "Promedio mensual de ventas para el filtro actual."

    fig_layout(fig, f'{xcol}  vs  {ycol}', 400)

    # Mini hist Y
    fh = go.Figure()
    fh.add_trace(go.Histogram(y=da[ycol],nbinsy=20,marker_color=COLORS['secondary'],opacity=.8))
    mu2,sig2 = da[ycol].mean(),da[ycol].std()
    if sig2>0:
        yr = np.linspace(da[ycol].min(),da[ycol].max(),200)
        xn = stats.norm.pdf(yr,mu2,sig2)*len(da)*(da[ycol].max()-da[ycol].min())/20
        fh.add_trace(go.Scatter(x=xn,y=yr,mode='lines',name='Normal',line=dict(color=COLORS['danger'],width=2)))
    fig_layout(fh,f'Dist. {ycol}',400)
    fh.update_layout(margin=dict(l=10,r=10,t=50,b=40),showlegend=False)

    card = html.Div([
        html.Div("📖 Interpretación",style={'fontWeight':'700','fontSize':'13px','marginBottom':'6px'}),
        dcc.Markdown(interp,style={'fontSize':'12px','color':COLORS['text']}),
    ],style={**card_style(COLORS['primary']),'padding':'12px 16px'})

    return fig, fh, card

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BOXPLOT
# ══════════════════════════════════════════════════════════════════════════════
def tab_boxplot(da):
    fig = go.Figure()
    for i,m in enumerate(MONTHS):
        fig.add_trace(go.Box(y=da[m].values,name=m[:3],
            marker_color=PALETTE[i%len(PALETTE)],
            boxpoints='outliers',jitter=.3,pointpos=-1.8,line=dict(width=1.5),
            hovertemplate=f'<b>{m}</b><br>%{{y:,.0f}}<extra></extra>'))
    fig_layout(fig,'📦 Boxplot Mensual — Q1/Q2/Q3 · IQR · Outliers (regla 1.5·IQR)',460)
    fig.update_layout(showlegend=False)

    rows=[]
    for m in MONTHS:
        v=da[m]; q1,q2,q3=v.quantile(.25),v.quantile(.5),v.quantile(.75)
        iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
        rows.append({'Mes':m,'Q1':fmt_cop(q1),'Mediana':fmt_cop(q2),'Q3':fmt_cop(q3),
                     'IQR':fmt_cop(iqr),'Lím inf':fmt_cop(lo),'Lím sup':fmt_cop(hi),
                     'Outliers':int(((v<lo)|(v>hi)).sum())})

    table = dbc.Table.from_dataframe(pd.DataFrame(rows),striped=True,bordered=False,hover=True,size='sm',style={'fontSize':'11px'})

    interp = html.Div([
        html.Div("📖 Cómo leer el boxplot",style={'fontWeight':'700','fontSize':'13px','marginBottom':'8px'}),
        html.Ul([
            html.Li("La caja = 50% central de datos (Q1 a Q3). Línea interior = mediana (Q2)."),
            html.Li("Bigotes = Q1 − 1.5·IQR  y  Q3 + 1.5·IQR  (regla de Tukey)."),
            html.Li("Puntos fuera de los bigotes = outliers."),
            html.Li("IQR grande → alta variabilidad entre empresas en ese mes."),
            html.Li("Muchos outliers → pocas empresas dominan las ventas de ese mes."),
        ],style={'fontSize':'12px','color':COLORS['text'],'lineHeight':'1.8'}),
    ],style={**card_style(COLORS['warning']),'marginTop':'16px'})

    return html.Div([dcc.Graph(figure=fig,config={'displayModeBar':False}),
                     html.H6("Estadísticos por mes",style={'fontWeight':'700','margin':'16px 0 8px'}),
                     table, interp])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HEATMAP & PEARSON
# ══════════════════════════════════════════════════════════════════════════════
def tab_heatmap(da):
    pivot = da.groupby('Ciudad')[MONTHS].sum()
    fhm = go.Figure(go.Heatmap(z=pivot.values,x=[m[:3] for m in MONTHS],y=pivot.index.tolist(),
        colorscale='Blues',text=[[fmt_cop(v) for v in r] for r in pivot.values],
        texttemplate='%{text}',textfont=dict(size=8),
        colorbar=dict(title='Ventas',tickfont=dict(size=9))))
    fig_layout(fhm,'🌡️ Heatmap: Ventas por Ciudad × Mes',460)
    fhm.update_layout(margin=dict(l=160,r=30,t=50,b=40))

    pcols = MONTHS+['Total_anual','Promedio_mensual','Meses_activos']
    pcols = [c for c in pcols if c in da.columns]
    plabels = [c[:5] if c in MONTHS else c[:6] for c in pcols]
    corr = da[pcols].corr()
    mask = np.tril(np.ones(corr.shape),k=0).astype(bool)
    z_masked = np.where(mask, corr.values, np.nan)

    fpc = go.Figure(go.Heatmap(z=z_masked,x=plabels,y=plabels,
        colorscale='RdBu',zmin=-1,zmax=1,
        text=np.where(mask,np.round(corr.values,2).astype(str),''),
        texttemplate='%{text}',textfont=dict(size=8),
        colorbar=dict(title='r',tickfont=dict(size=9))))
    fig_layout(fpc,'🔗 Matriz Pearson (triángulo inferior)',460)
    fpc.update_layout(margin=dict(l=60,r=30,t=50,b=60))

    interp = html.Div([
        html.Div("📖 Lectura",style={'fontWeight':'700','fontSize':'13px','marginBottom':'8px'}),
        html.Ul([
            html.Li([html.B("Heatmap: "),"Colores más intensos = más ventas. Identifica mes pico por ciudad."]),
            html.Li([html.B("Pearson r: "),"1 = correlación perfecta positiva, -1 = negativa, 0 = sin relación."]),
            html.Li("r > 0.7 entre meses → empresas tienden a vender en los mismos meses."),
            html.Li("r cercano a 0 → comportamiento estacional distinto entre empresas."),
        ],style={'fontSize':'12px','color':COLORS['text'],'lineHeight':'1.8'}),
    ],style={**card_style(COLORS['secondary']),'marginTop':'16px'})

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fhm,config={'displayModeBar':False}),xs=12, md=6),
                 dbc.Col(dcc.Graph(figure=fpc,config={'displayModeBar':False}),xs=12, md=6)]),
        interp])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════
def tab_regresion():
    available = [c for c in ['Promedio_mensual','Meses_activos','Max_mensual','CV'] if c in df_wide.columns]
    opts = [{'label':c,'value':c} for c in available]
    return html.Div([
        html.H6("Regresión Lineal",style={'fontWeight':'700','marginBottom':'4px'}),
        html.P("Y = Total_anual. Elige una o más variables X.",style={'color':COLORS['muted'],'fontSize':'12px','marginBottom':'12px'}),
        dbc.Row([dbc.Col([
            html.Label("Variables X (una = simple, varias = múltiple)",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
            dcc.Dropdown(id='reg-x',options=opts,value=[available[0]] if available else [],multi=True,clearable=False,style={'fontSize':'12px'}),
        ],width=8)],style={'background':COLORS['card'],'padding':'16px','borderRadius':'12px','border':f'1px solid {COLORS["border"]}','marginBottom':'16px'}),
        html.Div(id='reg-output'),
    ])

@app.callback(Output('reg-output','children'),
              Input('main-tabs','value'),Input('reg-x','value'),
              Input('filter-ciudad','value'),Input('filter-tipo','value'),Input('filter-zero','value'))
def cb_regresion(tab, xcols, cities, tipos, inc_zero):
    if tab != 'tab-regresion': return ''
    if not xcols: return html.Div("Selecciona al menos una variable X.",style={'color':COLORS['muted']})

    da = filter_df(cities,tipos,inc_zero)
    da = da[da['Flag_activo']==1].dropna(subset=['Total_anual']+xcols)
    if len(da) < len(xcols)+2:
        return html.Div("Muy pocas observaciones para la regresión.",style={'color':COLORS['danger']})

    X = da[xcols].values; y = da['Total_anual'].values
    reg = LinearRegression().fit(X,y)
    yp  = reg.predict(X)
    r2  = reg.score(X,y); n,k = len(y),len(xcols)
    r2a = 1-(1-r2)*(n-1)/(n-k-1) if n>k+1 else float('nan')
    rmse = np.sqrt(np.mean((y-yp)**2))
    residuals = y-yp

    terms = ' '.join([f'{reg.coef_[i]:+.4g}·{xcols[i]}' for i in range(k)])
    eq = f"Total_anual = {reg.intercept_:,.0f} {terms}"

    if k==1:
        xv = X[:,0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv,y=y,mode='markers',marker=dict(color=COLORS['primary'],size=7,opacity=.7),name='Datos',hovertext=da['Nombre']))
        xl = np.linspace(xv.min(),xv.max(),100)
        fig.add_trace(go.Scatter(x=xl,y=reg.intercept_+reg.coef_[0]*xl,mode='lines',
            line=dict(color=COLORS['danger'],width=2.5,dash='dash'),name=f'Regresión (R²={r2:.3f})'))
        fig_layout(fig,f'Regresión simple: {xcols[0]} → Total_anual',420)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y,y=yp,mode='markers',marker=dict(color=COLORS['primary'],size=7,opacity=.7),name='Predicho vs Real',hovertext=da['Nombre']))
        mx=max(y.max(),yp.max())
        fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',line=dict(color=COLORS['danger'],dash='dash',width=2),name='Perfecta'))
        fig_layout(fig,'Regresión múltiple: Real vs Predicho',420)

    fres = go.Figure(go.Scatter(x=yp,y=residuals,mode='markers',marker=dict(color=COLORS['secondary'],size=6,opacity=.7)))
    fres.add_hline(y=0,line_dash='dash',line_color='gray')
    fig_layout(fres,'Residuos vs Valores ajustados',280)

    fcoef = go.Figure(go.Bar(x=xcols,y=reg.coef_,marker_color=[COLORS['success'] if c>0 else COLORS['danger'] for c in reg.coef_]))
    fig_layout(fcoef,'Coeficientes',280)

    eq_card = html.Div([
        html.Div("📐 Ecuación",style={'fontWeight':'700','fontSize':'13px','marginBottom':'8px'}),
        html.Code(eq,style={'fontSize':'12px','background':'#F1F5F9','padding':'8px 12px','borderRadius':'6px','display':'block','wordBreak':'break-all'}),
        dbc.Row([
            dbc.Col(html.Div([html.Div("R²",style={'fontSize':'11px','color':COLORS['muted']}),html.Div(f"{r2:.4f}",style={'fontSize':'22px','fontWeight':'700','color':COLORS['primary']})],style=card_style()),xs=6, md=3),
            dbc.Col(html.Div([html.Div("R² Ajust.",style={'fontSize':'11px','color':COLORS['muted']}),html.Div(f"{r2a:.4f}",style={'fontSize':'22px','fontWeight':'700','color':COLORS['secondary']})],style=card_style()),xs=6, md=3),
            dbc.Col(html.Div([html.Div("RMSE",style={'fontSize':'11px','color':COLORS['muted']}),html.Div(fmt_cop(rmse),style={'fontSize':'22px','fontWeight':'700','color':COLORS['warning']})],style=card_style()),xs=6, md=3),
            dbc.Col(html.Div([html.Div("n",style={'fontSize':'11px','color':COLORS['muted']}),html.Div(str(n),style={'fontSize':'22px','fontWeight':'700','color':COLORS['success']})],style=card_style()),xs=6, md=3),
        ],style={'marginTop':'12px'}),
    ],style={**card_style(COLORS['primary']),'marginBottom':'16px'})

    interp_text = (f"**R² = {r2:.4f}**: el modelo explica el **{r2*100:.1f}%** de la variación en ventas. "
        f"{'Buen ajuste.' if r2>.6 else 'Ajuste moderado.' if r2>.3 else 'Ajuste débil — considera más variables.'} "
        f"RMSE = {fmt_cop(rmse)} (error promedio de predicción).")
    interp = html.Div([html.Div("📖 Lectura",style={'fontWeight':'700','fontSize':'13px','marginBottom':'6px'}),
                       dcc.Markdown(interp_text,style={'fontSize':'12px','color':COLORS['text']})],
                      style={**card_style(COLORS['success']),'marginTop':'16px'})

    return html.Div([eq_card,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig,  config={'displayModeBar':False}),xs=12, md=7),
                 dbc.Col(dcc.Graph(figure=fcoef,config={'displayModeBar':False}),xs=12, md=5)]),
        dcc.Graph(figure=fres,config={'displayModeBar':False}), interp])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
def tab_kmeans():
    return html.Div([
        html.H6("Clustering K-Means",style={'fontWeight':'700','marginBottom':'4px'}),
        html.P("Segmenta empresas por perfil de ventas. El codo y la silueta ayudan a elegir K.",style={'color':COLORS['muted'],'fontSize':'12px','marginBottom':'12px'}),
        dbc.Row([
            dbc.Col([html.Label("K máximo para el codo",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
                     dcc.Slider(id='km-kmax',min=3,max=12,step=1,value=8,marks={i:str(i) for i in range(3,13)},tooltip={'placement':'bottom'})],xs=12, md=6),
            dbc.Col([html.Label("K elegido",style={'fontWeight':'600','fontSize':'12px','color':COLORS['muted']}),
                     dcc.Slider(id='km-k',  min=2,max=10,step=1,value=3,marks={i:str(i) for i in range(2,11)}, tooltip={'placement':'bottom'})],xs=12, md=6),
        ],style={'background':COLORS['card'],'padding':'16px 24px','borderRadius':'12px','border':f'1px solid {COLORS["border"]}','marginBottom':'16px'}),
        html.Div(id='km-output'),
    ])

@app.callback(Output('km-output','children'),
              Input('main-tabs','value'),Input('km-kmax','value'),Input('km-k','value'),
              Input('filter-ciudad','value'),Input('filter-tipo','value'),Input('filter-zero','value'))
def cb_kmeans(tab, kmax, k, cities, tipos, inc_zero):
    if tab != 'tab-kmeans': return ''
    feats = [c for c in ['Total_anual','Promedio_mensual','Meses_activos','CV'] if c in df_wide.columns]
    da = filter_df(cities,tipos,inc_zero)
    da = da[da['Flag_activo']==1].dropna(subset=feats)

    if len(da) < kmax+2:
        return html.Div(f"Muy pocas observaciones ({len(da)}). Amplía el filtro.",style={'color':COLORS['danger']})

    X = StandardScaler().fit_transform(da[feats].values)

    ks = range(2,kmax+1)
    inertias, sils = [], []
    for ki in ks:
        km = KMeans(n_clusters=ki,random_state=42,n_init=10)
        lb = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X,lb))

    felbow = make_subplots(specs=[[{'secondary_y':True}]])
    felbow.add_trace(go.Scatter(x=list(ks),y=inertias,mode='lines+markers',name='Inercia',
        marker=dict(color=COLORS['primary'],size=8),line=dict(width=2.5)),secondary_y=False)
    felbow.add_trace(go.Scatter(x=list(ks),y=sils,mode='lines+markers',name='Silueta',
        marker=dict(color=COLORS['success'],size=8),line=dict(width=2.5,dash='dot')),secondary_y=True)
    felbow.update_yaxes(title_text='Inercia',secondary_y=False,gridcolor='#E2E8F0',color=COLORS['primary'])
    felbow.update_yaxes(title_text='Silueta',secondary_y=True,color=COLORS['success'])
    felbow.update_layout(height=320,plot_bgcolor=COLORS['bg'],paper_bgcolor=COLORS['card'],
        font=dict(family='Inter,Arial',size=11),title='Método del Codo + Silueta',
        margin=dict(l=50,r=60,t=50,b=40),legend=dict(bgcolor='rgba(0,0,0,0)'))
    felbow.add_vline(x=k,line_dash='dash',line_color=COLORS['danger'],
        annotation_text=f'K={k}',annotation_position='top right')

    km_f = KMeans(n_clusters=k,random_state=42,n_init=10)
    da = da.copy()
    da['Cluster'] = km_f.fit_predict(X).astype(str)
    sil_f = silhouette_score(X,km_f.labels_)
    best_k = list(ks)[sils.index(max(sils))]

    fsc = px.scatter(da,x='Promedio_mensual',y='Total_anual',color='Cluster',
        hover_name='Nombre',size='Meses_activos',color_discrete_sequence=PALETTE,
        labels={'Promedio_mensual':'Promedio mensual','Total_anual':'Total anual'})
    fig_layout(fsc,f'Clusters K={k} — Promedio mensual vs Total anual',380)

    profile = da.groupby('Cluster')[feats].mean()
    profile.columns = [c.replace('_',' ') for c in feats]
    pfmt = profile.copy()
    for c in pfmt.columns:
        pfmt[c] = pfmt[c].apply(lambda v: fmt_cop(v) if abs(v)>100 else f'{v:.3f}')
    pfmt.insert(0,'N empresas',da.groupby('Cluster').size())
    table = dbc.Table.from_dataframe(pfmt.reset_index(),striped=True,bordered=False,hover=True,size='sm',style={'fontSize':'12px'})

    interp = html.Div([
        html.Div("📖 Justificación y Lectura",style={'fontWeight':'700','fontSize':'13px','marginBottom':'6px'}),
        dcc.Markdown(
            f"**K elegido: {k}** — silueta = {sil_f:.3f} "
            f"({'buena separación >0.5' if sil_f>.5 else 'aceptable 0.3–0.5' if sil_f>.3 else 'clusters solapados — prueba otro K'}). "
            f"K óptimo por silueta: **K={best_k}** ({max(sils):.3f}). "
            f"Features usadas y escaladas: {', '.join(feats)}.",
            style={'fontSize':'12px','color':COLORS['text']}),
        html.Ul([
            html.Li("Método del codo: elige el K donde la inercia deja de caer bruscamente."),
            html.Li("Silueta (0–1): qué tan bien separados están los clusters. >0.5 = bueno."),
            html.Li("Perfil de clusters: compara medias para etiquetar segmentos (ej. 'alto volumen', 'estacionales', 'inactivos')."),
        ],style={'fontSize':'12px','color':COLORS['text'],'lineHeight':'1.8','marginTop':'8px'}),
    ],style={**card_style(COLORS['secondary']),'marginTop':'16px'})

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=felbow,config={'displayModeBar':False}),xs=12, md=5),
                 dbc.Col(dcc.Graph(figure=fsc,   config={'displayModeBar':False}),xs=12, md=7)]),
        html.H6("Perfil por cluster",style={'fontWeight':'700','margin':'16px 0 6px'}),
        table, interp])

# ─── RUN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("✅  Dashboard listo → http://127.0.0.1:8050\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
