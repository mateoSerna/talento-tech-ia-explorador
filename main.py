"""
@author: Macromakers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
datos = pd.read_excel("informe.xlsx")

# Definición de las variables
print(datos.dtypes)

datos.head()
# CATEGORIZACIÓN DE LOS CAMPOS (CATEGORICO, NUMERICO)
print("\n--- CATEGORIZACIÓN DE LOS CAMPOS ---")
print("Campos numéricos:")
print(datos.select_dtypes(include="number").columns.tolist())
print("Campos categóricos / no numéricos:")
print(datos.select_dtypes(exclude="number").columns.tolist())

# CANTIDAD DE COLUMNAS BD
print("\n--- CANTIDAD DE COLUMNAS ---")
print(datos.shape[1])

# CANTIDAD DE FILAS
print("\n--- CANTIDAD DE FILAS ---")
print(datos.shape[0])

# CANTIDAD DE ELEMENTOS
print("\n--- CANTIDAD DE ELEMENTOS (filas x columnas) ---")
print(datos.size)

# CANTIDAD DE ELEMENTOS NULOS ISNULL
print("\n--- ELEMENTOS NULOS POR COLUMNA (isnull) ---")
print(datos.isnull().sum())

# CANTIDAD DE ELEMENTOS NULOS ISNA
print("\n--- ELEMENTOS NULOS POR COLUMNA (isna) ---")
print(datos.isna().sum())

# CANTIDAD TOTAL DE DATOS NULOS
print("\n--- TOTAL DE DATOS NULOS ---")
print(datos.isnull().sum().sum())

# NOMBRE DE LOS CAMPOS (KEYS / COLUMNS)
print("\n--- NOMBRE DE LOS CAMPOS ---")
print(datos.columns.tolist())

# ELIMINAR COLUMNAS INNECESARIAS
# Cambia la lista por las columnas que deseas eliminar
columnas_a_eliminar = []  # Ejemplo: ["columna1", "columna2"]
datos_limpios = datos.drop(columns=columnas_a_eliminar, errors="ignore")
print("\n--- COLUMNAS DESPUÉS DE ELIMINAR INNECESARIAS ---")
print(datos_limpios.columns.tolist())

# SELECCIÓN DE DATOS NUMÉRICOS
datos_numericos = datos.select_dtypes(include="number")
print("\n--- DATOS NUMÉRICOS ---")
print(datos_numericos.head())

# SELECCIÓN DE DATOS NO NUMÉRICOS
datos_no_numericos = datos.select_dtypes(exclude="number")
print("\n--- DATOS NO NUMÉRICOS ---")
print(datos_no_numericos.head())

# REEMPLAZAR DATOS NULOS (FILLNA) — se reemplazan con 0 en numéricos y "N/A" en texto
datos_sin_nulos = datos.copy()
datos_sin_nulos[datos_numericos.columns] = datos_numericos.fillna(0)
datos_sin_nulos[datos_no_numericos.columns] = datos_no_numericos.fillna("N/A")
print("\n--- DATOS DESPUÉS DE FILLNA ---")
print(datos_sin_nulos.head())

# MEDIA
print("\n--- MEDIA (por columna numérica) ---")
print(datos_numericos.mean())

# MEDIANA
print("\n--- MEDIANA (por columna numérica) ---")
print(datos_numericos.median())

# MODA
print("\n--- MODA (por columna) ---")
print(datos.mode().iloc[0])

# DESVIACIÓN ESTÁNDAR
print("\n--- DESVIACIÓN ESTÁNDAR ---")
print(datos_numericos.std())

# VARIANZA
print("\n--- VARIANZA ---")
print(datos_numericos.var())

# VALOR MÁXIMO Y MÍNIMO
print("\n--- VALOR MÁXIMO ---")
print(datos_numericos.max())
print("\n--- VALOR MÍNIMO ---")
print(datos_numericos.min())

# RESUMEN USANDO DESCRIBE()
print("\n--- RESUMEN ESTADÍSTICO COMPLETO (describe) ---")
print(datos.describe(include="all"))

# VALOR CUARTILES
print("\n--- CUARTIL 25% (Q1) ---")
print(datos_numericos.quantile(0.25))
print("\n--- CUARTIL 50% (Q2 - MEDIANA) ---")
print(datos_numericos.quantile(0.50))
print("\n--- CUARTIL 75% (Q3) ---")
print(datos_numericos.quantile(0.75))

# GRÁFICO DE BARRAS — MAYOR PRESENCIA DE VENTAS POR CIUDAD
meses = [
    "ENERO",
    "Febrero ",
    "Marzo",
    "Abril ",
    "Mayo",
    "Junio ",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]
# ventas_ciudad = (
#     datos.groupby("Ciudad")[meses].sum().sum(axis=1)
#          .sort_values(ascending=False)
#     / 1_000_000
# )

# plt.figure(figsize=(10, 6))
# plt.bar(
#     ventas_ciudad.index.astype(str),
#     ventas_ciudad.values,
#     color="steelblue",
#     edgecolor="black",
# )
# plt.title("Total de Ventas por Ciudad")
# plt.xlabel("Ciudad")
# plt.ylabel("Total Ventas (Millones COP)")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()
# print("\n--- TOTAL VENTAS POR CIUDAD ---")
# print(ventas_ciudad)

# GRÁFICO DE LÍNEAS — VENTAS MENSUALES TOTALES

# ventas_mensuales = datos[meses].sum() / 1_000_000

# plt.figure(figsize=(10, 6))
# plt.plot(meses_labels, ventas_mensuales.values, marker="o", color="steelblue", linewidth=2)
# plt.title("Ventas Mensuales Totales")
# plt.xlabel("Mes")
# plt.ylabel("Total Ventas (Millones COP)")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()
# print("\n--- VENTAS MENSUALES TOTALES ---")
# print(ventas_mensuales.rename(dict(zip(meses, meses_labels))))

# GRÁFICO DE BARRAS — TOP 10 CLIENTES CON MAYOR VENTAS
# top10 = (
#     datos.assign(total=datos[meses].sum(axis=1))
#     .nlargest(10, "total")
#     .set_index("Nombre")["total"]
#     / 1_000_000
# )

# plt.figure(figsize=(12, 6))
# plt.bar(range(len(top10)), top10.values, color="steelblue", edgecolor="black")
# plt.xticks(range(len(top10)), top10.index.astype(str), rotation=45, ha="right")
# plt.title("Top 10 Clientes con Mayor Ventas")
# plt.xlabel("Cliente")
# plt.ylabel("Total Ventas (Millones COP)")
# plt.tight_layout()
# plt.show()
# print("\n--- TOP 10 CLIENTES ---")
# print(top10)

# MAPA DE CALOR — VENTAS POR CIUDAD Y MES
meses_labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
calor = (
    datos.groupby("Ciudad")[meses].sum()
    / 1_000_000
).rename(columns=dict(zip(meses, meses_labels)))
calor = calor[calor.sum(axis=1) > 0]

plt.figure(figsize=(14, 8))
sns.heatmap(calor, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5)
plt.title("Mapa de Calor — Ventas por Ciudad y Mes (Millones COP)")
plt.xlabel("Mes")
plt.ylabel("Ciudad")
plt.tight_layout()
plt.show()
