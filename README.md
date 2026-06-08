# AI Explorador — EDA de Ventas

Análisis exploratorio de datos (EDA) sobre un informe de ventas en Excel, con dos modos de uso: script de consola y dashboard web interactivo.

---

## Stack

| Capa | Librería |
|---|---|
| Manipulación de datos | `pandas`, `numpy` |
| Estadística / ML | `scipy`, `scikit-learn` |
| Visualización estática | `matplotlib`, `seaborn` |
| Dashboard interactivo | `dash`, `dash-bootstrap-components`, `plotly` |
| Lectura de Excel | `openpyxl` |
| Runtime | Python 3.13 |

---

## Estructura

```
ai-explorador/
├── dashboard.py          # Dashboard web interactivo (Dash)
├── informe.xlsx          # Dataset de ventas (fallback)
├── informe_limpio.xlsx   # Dataset procesado con hoja Wide_Maestra (preferido)
├── requirements.txt      # Dependencias
└── README.md
```

---

## Instalación

```bash
# Crear entorno conda (recomendado)
conda create -n tt-ai python=3.13
conda activate tt-ai

# Instalar dependencias
pip install -r requirements.txt

# Dependencias adicionales para el dashboard
pip install dash dash-bootstrap-components plotly scipy scikit-learn seaborn
```

---

## Uso

### Análisis en consola (`main.py`)

Ejecuta el EDA completo: tipos de datos, nulos, estadísticos descriptivos, cuartiles y gráficos de barras, líneas y mapa de calor.

```bash
python main.py
```

Genera automáticamente:
- Gráfico de barras: ventas totales por ciudad
- Gráfico de líneas: ventas mensuales totales
- Top 10 clientes por volumen de ventas
- Mapa de calor: ventas por ciudad × mes

### Dashboard interactivo (`dashboard.py`)

```bash
python dashboard.py
```

Abrir en el navegador: [http://127.0.0.1:8050](http://127.0.0.1:8050)

El dashboard carga automáticamente `informe_limpio.xlsx` (hoja `Wide_Maestra`) si existe, o `informe.xlsx` como fallback.

#### Pestañas disponibles

| Pestaña | Contenido |
|---|---|
| Resumen | KPIs, top 10 empresas, distribución por ciudad y tipo de cliente |
| Explorador X/Y | Scatter, barras, histograma con curva normal, línea de tiempo |
| Boxplot | Distribución mensual con Q1/Q2/Q3, IQR y outliers |
| Heatmap & Pearson | Mapa de calor ciudad × mes y matriz de correlación |
| Regresión | Regresión lineal simple o múltiple con R², RMSE y residuos |
| K-Means | Clustering por perfil de ventas con método del codo y silueta |

#### Filtros globales (sidebar)
- **Ciudad** — filtra por una o varias ciudades
- **Tipo Cliente** — Empresa / Persona Natural
- **Sin ventas** — incluir o excluir clientes inactivos

---

## Formato del Excel

El archivo debe tener al menos estas columnas:

| Columna | Descripción |
|---|---|
| `Nombre` | Nombre del cliente o empresa |
| `Ciudad` | Ciudad del cliente |
| `Enero` … `Diciembre` | Ventas del mes en COP |

Las columnas de mes toleran variaciones de mayúsculas y espacios (ej. `Febrero `, `ENERO`).

---

## Despliegue (producción)

Cambiar el host en `dashboard.py` antes de desplegar:

```python
app.run(debug=False, host='0.0.0.0', port=8050)
```

Opciones gratuitas recomendadas: **Render.com**, **Railway.app**.
