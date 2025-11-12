"""
Funciones compartidas de la aplicación NYC-Taxi
(a partir del notebook 2023-01_4_2_F2_v6)

- v2: añade métricas para Gold
- v3: limpieza de v2

Utilización:

import sys
sys.path.append("/content/drive/MyDrive/taxi_project")
from nombre_archivo import nombre_funcion
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pyspark.sql import SparkSession, DataFrame

from pyspark.sql.functions import (
    sum as spark_sum,
    min as spark_min,
    max as spark_max,
    round as spark_round,
    pi as spark_pi,
    count, col, when, isnan, isnull, mean, stddev, desc, asc,
    year, month, dayofweek, dayofmonth, weekofyear, date_format, to_date,
    isnotnull, date_trunc, datediff, lit, percentile_approx, expr, broadcast,
    coalesce, avg, hour, unix_timestamp, sin, cos, udf,
    monotonically_increasing_id
)

from pyspark.sql.types import (
    NumericType, StringType, DateType, TimestampType, DoubleType, BooleanType,
    IntegerType
)

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer, StandardScaler, MinMaxScaler, VectorAssembler, StringIndexerModel,
    OneHotEncoder
)
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark import StorageLevel

from datetime import datetime, timezone

from google.colab import drive

from mpl_toolkits.mplot3d import Axes3D

from typing import Any

import os
import re
import json
import time
import sys
import gc
import psutil
import hashlib
import shutil


# *** ACCESO A LOS DATOS ***


def get_bronze_dataset_paths(bronze_root):
    """
    Devuelve todas las rutas individuales de datasets en la capa Bronze.

    Args:
        bronze_root (str): Ruta raíz de la capa Bronze.

    Returns:
        list: Lista de rutas individuales de datasets.
    """
    parquet_dirs = []
    for root, _, files in os.walk(bronze_root):
        if any(f.endswith(".parquet") for f in files):
            parquet_dirs.append(root)
    return sorted(parquet_dirs)
	

def extract_year_month_from_hive_partition(path):
    """
    Extrae año y mes de la estructura de particiones de Hive.

    Args:
        path (str): Ruta del dataset con estructura ingestion_year=YYYY/ingestion_month=MM

    Returns:
        tuple: (año, mes) o (None, None) si no se puede extraer
    """
    # busca patrones de partición de Hive en la ruta completa
    year_pattern = r'ingestion_year=(\d{4})'
    month_pattern = r'ingestion_month=(\d{1,2})'

    year_match = re.search(year_pattern, path)
    month_match = re.search(month_pattern, path)

    if year_match and month_match:
        year = int(year_match.group(1))
        month = int(month_match.group(1))

        # valida que el mes esté en rango válido
        if 1 <= month <= 12:
            return year, month

    return None, None
	
	
def format_date_info(year, month):
    """
    Formatea la información de fecha para mostrar.

    Args:
        year (int): Año
        month (int): Mes

    Returns:
        str: Fecha formateada
    """
    if year and month:
        try:
            # crea objeto datetime para obtener el nombre del mes
            date_obj = datetime(year, month, 1)
            month_names_es = {
                1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
            }
            return f"{year}-{month:02d} ({month_names_es[month]} {year})"
        except ValueError:
            return f"{year}-{month:02d}"
    return "Fecha no identificada"
	

# *** GESTIÓN DE MEMORIA ***


def show_memory_disk(msg: str = ""):
    """
    Muestra el uso de memoria RAM y de espacio de disco

    Args:
        msg: mensaje a mostrar
    """
    # memoria
    mem = psutil.virtual_memory()
    texto = f"Memoria usada: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB ({mem.percent}%)"
    if msg:
        print(f"\n[{msg}] {texto}")
    else:
        print(f"\n{texto}")

    # disco
    disk = psutil.disk_usage('/')
    texto = f"Disco usado: {disk.used / 1e9:.2f} GB / {disk.total / 1e9:.2f} GB ({disk.percent}%)"
    if msg:
        print(f"[{msg}] {texto}\n")
    else:
        print(f"{texto}\n")
        
        
def clear_all_caches(spark: SparkSession):
    """
    Limpieza completa de memoria de Spark, JVM y Python

    Args:
        spark: sesión de Spark
    """

    show_memory_disk("Inicio limpieza completa de memoria")

    # borra datos cacheados/persistidos de memoria y disco
    spark.catalog.clearCache()
    # recolector de basura de Spark/JVM
    spark._jvm.System.gc()
    # recolector de basura de Python
    gc.collect()

    show_memory_disk("Final limpieza completa de memoria")
    
    
# *** CARGA Y GUARDADO DE FICHEROS ***


def save_parquet(df, ruta, modo="overwrite", compresion="snappy"):
    """
    Guarda un DataFrame en formato Parquet.

    Args:
        df: DataFrame de PySpark
        ruta: str, ruta donde guardar el archivo
        modo: str, modo de escritura ("overwrite", "append", "ignore", "error")
        compresion: str, tipo de compresión ("snappy", "gzip", "lzo", "brotli", "lz4", "zstd")

    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    try:
        print(f"💾 Guardando DataFrame en {ruta}...")
        print(f"📊 Registros a guardar: {df.count()}")

        df.write \
          .mode(modo) \
          .option("compression", compresion) \
          .parquet(ruta)

        print(f"✅ Dataframe guardado correctamente en {ruta}")

        return True

    except Exception as e:
        print(f"❌ Error al guardar: {str(e)}")

        return False
        
        
def load_parquet(spark, ruta, mostrar_info=True):
    """
    Lee un DataFrame en formato Parquet.

    Args:
        spark: sesión de Spark
        ruta: str, ruta del archivo Parquet a leer
        mostrar_info: str, True para mostrar información del DataFrame leído

    Returns:
        DataFrame leído o None si hay error
    """
    try:
        print(f"📖 Leyendo DataFrame desde {ruta}...")

        df_leido = spark.read.parquet(ruta)

        if mostrar_info:
            filas = df_leido.count()
            columnas = len(df_leido.columns)
            print("✅ Dataframe cargado correctamente")
            print(f"📊 Registros leídos: {filas}")
            print(f"📋 Columnas: {columnas}")
            print(f"🏷️  Nombres de columnas: {df_leido.columns}")
        else:
            print("✅ Dataframe cargado correctamente")

        return df_leido

    except Exception as e:
        print(f"❌ Error al leer el Dataframe: {str(e)}")
        return None

        
# *** CORRELACIÓN ***


'''
v4
'''
def compute_correlation_matrix(ds_spark, variables_list, method="pearson"):
    """
    Calcula la matriz de correlación para un grupo de variables

    Args:
        ds_spark: DataFrame de PySpark
        variables_list: lista de nombres de columnas a correlacionar
        method: método de correlación ('pearson' o 'spearman')

    Returns:
        correlation_array: matriz de correlación como array numpy
        variables_list: lista de variables (para etiquetas)
    """
    print(f"🔄 Calculando correlaciones de {method} para {len(variables_list)} variables...")

    # comprueba que todas las variables existan en el DataFrame
    available_cols = set(ds_spark.columns)

    # lista de las variables que sí existen en el DataFrame
    variables_list = [var for var in variables_list if var in available_cols]

    if len(variables_list) < 2:
        raise ValueError("Se necesitan al menos 2 variables para calcular correlaciones")

    # crea vector de características
    assembler = VectorAssembler(
        inputCols=variables_list,
        outputCol="features",
        handleInvalid="skip"  # omite registros con valores nulos
    )

    # transforma dataframe en un vector, crea la columna 'features'
    ds_vector = assembler.transform(ds_spark).select("features")

    # prepara la caché en memoria y disco para acelerar las operaciones
    ds_vector = ds_vector.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

    # materializa la caché
    ds_vector.count()

    # calcula y extrae
    try:

        # calcula la matriz de correlación
        correlation_matrix = Correlation.corr(ds_vector, "features", method).collect()[0][0]
        # convierte a array NumPy
        correlation_array = correlation_matrix.toArray()

        return correlation_array, variables_list

    finally:
        # garantiza la limpieza incluso si hay error
        ds_vector.unpersist()
        del ds_vector, assembler, correlation_matrix
        gc.collect()

'''
v3 - añade métricas para Gold
'''
def analyze_correlation_group(
    ds_spark: DataFrame,
    variables_list: list,
    group_name: str = "Variables",
    method: str = "pearson"
):
    """
    Análisis completo de correlaciones para un grupo de variables

    Args:
        ds_spark: DataFrame de PySpark
        variables_list: lista de variables a analizar
        group_name: nombre del grupo (para títulos e informes)
        method: método de correlación ('pearson' o 'spearman')

    Returns:
        df_correlation: DataFrame de Pandas con la matriz
        summary_stats: resumen de estadísticas
        top_correlations: lista de pares más correlacionados
        corr_metadata: metadatos de correlación para la capa Gold
    """
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DE CORRELACIÓN - {group_name.upper()}")
    print(f"{'='*60}")

    # calcula matriz de correlación
    corr_matrix, final_variables = compute_correlation_matrix(ds_spark, variables_list, method)

    try:

      # convierte a DataFrame de Pandas para análisis
      df_correlation = pd.DataFrame(
          corr_matrix,
          index=final_variables,
          columns=final_variables
      )

      # limpia memoria
      del corr_matrix
      gc.collect()

      # resumen de estadísticas
      summary_stats = calculate_correlation_summary(df_correlation)

      ## top_correlations (lista plana ordenada desde la matriz)

      # inicializa lista de resultados
      flat = []
      # lista Python de columnas
      cols = df_correlation.columns.tolist()
      # recorre las columnas (i)
      for i in range(len(cols)):
          # recorre sólo las columnas (j) de índice mayor que el actual (i)
          # evita repetir pares (i-j, j-i) y la diagonal
          for j in range(i+1, len(cols)):
              # valor (i, j) de la matriz de correlación
              val = df_correlation.iloc[i, j]
              # añade dict a la lista de resultados
              flat.append(
                  {
                      "var1": cols[i],
                      "var2": cols[j],
                      # correlación: float, None si es NaN
                      "correlation": float(val) if not pd.isna(val) else None}
              )
      # ordena de mayor a menor sin considerar signo
      flat.sort(key=lambda x: abs(x["correlation"]) if x["correlation"] is not None else 0, reverse=True)
      # lista de diccionarios resultante
      top_correlations = flat

      # metadata de correlación
      corr_metadata = {
        "method": method,
        "n_rows_used": int(ds_spark.count()) if hasattr(ds_spark, "count") else None,
        "n_variables": len(final_variables),
        "variables": final_variables,
        
        #"timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat()
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
        
      }

      # muestra resumen
      print_correlation_summary(summary_stats, group_name)

      return df_correlation, summary_stats, top_correlations, corr_metadata

    finally:
      # garantiza la limpieza de memoria
      gc.collect()


def calculate_correlation_summary(df_correlation):
    """
    Calcula un resumen de estadísticas de la matriz de correlación

    Args:
        df_correlation: DataFrame de Pandas con la matriz de correlación

    Returns:
        summary: diccionario con estadísticas
    """
    # obtiene sólo la matriz triangular superior (sin diagonal)
    mask = np.triu(np.ones_like(df_correlation.values, dtype=bool), k=1)
    upper_triangle = df_correlation.values[mask]

    # estadísticas
    summary = {
        'total_pairs': len(upper_triangle),
        'mean_correlation': np.mean(upper_triangle),
        'max_correlation': np.max(upper_triangle),
        'min_correlation': np.min(upper_triangle),
        'std_correlation': np.std(upper_triangle),
        'high_correlations': np.sum(np.abs(upper_triangle) > 0.7),
        'moderate_correlations': np.sum((np.abs(upper_triangle) > 0.3) & (np.abs(upper_triangle) <= 0.7)),
        'low_correlations': np.sum(np.abs(upper_triangle) <= 0.3),
    }

    return summary
    

def print_correlation_summary(summary_stats, group_name):
    """
    Imprime un resumen de las estadísticas de correlación

    Args:
        summary_stats: diccionario con estadísticas
        group_name: nombre del grupo (para títulos e informes)
    """
    print(f"\n📈 RESUMEN ESTADÍSTICO - {group_name}")
    print("-" * 50)
    print(f"Total de pares de variables: {summary_stats['total_pairs']}")
    print(f"Correlación promedio: {summary_stats['mean_correlation']:.3f}")
    print(f"Correlación máxima: {summary_stats['max_correlation']:.3f}")
    print(f"Correlación mínima: {summary_stats['min_correlation']:.3f}")
    print(f"Desviación estándar: {summary_stats['std_correlation']:.3f}")

    print(f"\n🎯 DISTRIBUCIÓN DE CORRELACIONES:")
    print(f"Alto (|r| > 0.7): {summary_stats['high_correlations']} pares")
    print(f"Moderado (0.3 < |r| ≤ 0.7): {summary_stats['moderate_correlations']} pares")
    print(f"Bajo (|r| ≤ 0.3): {summary_stats['low_correlations']} pares") 
    
    
'''
v2 - mejor gestión de memoria
'''
def plot_correlation_heatmap(df_correlation, group_name="Variables", method="pearson", figsize=(10, 8)):
    """
    Crea un heatmap de la matriz de correlación

    Args:
        df_correlation: DataFrame de Pandas con la matriz de correlación
        group_name: nombre del grupo (para títulos e informes)
        method: método de correlación ('pearson' o 'spearman'), uso sólo para título del gráfico
        figsize: tamaño de la figura (por defecto: (10, 8))
    """
    # figura explícita para cerrar exactamente lo creado
    fig, ax = plt.subplots(figsize=figsize)

    # crea máscara para la matriz triangular inferior
    mask = np.tril(np.ones(df_correlation.shape, dtype=bool))

    # crea heatmap
    sns.heatmap(
        df_correlation,
        annot=True,
        cmap=sns.diverging_palette(20, 220, n=256),
        center=0,
        square=True,
        mask=mask,
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        fmt='.2f',
        vmin=-1,
        vmax=1,
        # genera el heatmap directamente sobre el eje
        ax=ax
    )

    ax.set_title(f'Correlación de {method} - {group_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Variables', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # evita bloquear la ejecución (la figura no se quedará en memoria)
    plt.show(block=False)

    # limpia la memoria
    plt.close('all')
    del mask, df_correlation, fig, ax
    gc.collect()

    
def identify_high_correlations(df_correlation, threshold=0.7):
    """
    Identifica pares de variables con alta correlación

    Args:
        df_correlation: DataFrame de Pandas con la matriz de correlación
        threshold: umbral para identificar correlaciones altas (por defecto: 0.7)

    Returns:
        high_corr_pairs: lista de tuplas (var1, var2, corr_value)
    """
    print(f"\n🔍 CORRELACIONES ALTAS (|r| > {threshold})")
    print("-" * 50)

    # inicializa resultados
    high_corr_pairs = []

    # recorre los pares (fila, columna)
    for i in range(len(df_correlation.columns)):
        for j in range(i+1, len(df_correlation.columns)):

            # correlación del par actual (fila, columna)
            var1 = df_correlation.columns[i]
            var2 = df_correlation.columns[j]
            corr_value = df_correlation.iloc[i, j]

            # compara correlación actual con el valor umbral
            if abs(corr_value) > threshold:
                # superior: guarda correlación actual
                high_corr_pairs.append((var1, var2, corr_value))
                print(f"{var1} ↔ {var2}: {corr_value:.3f}")

    if not high_corr_pairs:
        print("No se encontraron correlaciones altas.")

    return high_corr_pairs


def identify_high_correlations_from_toplist(
    top_correlations: list,
    threshold: float = 0.7,
    top_n: int = None
):
    """
    Identifica pares con |correlation| > threshold a partir de 'top_correlations'
    (lista plana ordenada de dicts {'var1','var2','correlation'}).

    Args:
      top_correlations: lista plana ordenada (desc por |corr|)
      threshold: umbral absoluto para considerar alta correlación
      top_n: opcional, limita la salida a los top_n pares que cumplan el umbral

    Returns:
      lista de tuplas (var1, var2, corr_value) ordenada por |corr| descendente
    """
    # inicializa resultados
    results = []
    # comprueba que se haya recibido top_correlations
    if not top_correlations:
        return results

    # recorre los pares de top_correlations
    for item in top_correlations:
        # correlación
        corr = item.get("correlation")
        # comprueba que tenga valor
        if corr is None:
            continue
        # comprueba si supera el umbral
        if abs(corr) > threshold:
            # añade el par al resultado
            results.append((item["var1"], item["var2"], corr))
            # comprueba si se ha llegado al límite de pares (si se ha solicitado)
            if top_n is not None and len(results) >= top_n:
                break

    return results

    
'''
v2 - añade métricas para Gold
'''
def analyze_correlations(
    ds_spark: DataFrame,
    variables_list: list,
    group_name: str = "Variables",
    method: str = "pearson",
    plot: bool = True,
    threshold: float = 0.7
):
    """
    Ejecuta un análisis completo de correlaciones para un grupo de variables

    Args:
        ds_spark: DataFrame de PySpark
        variables_list: lista de variables a analizar
        group_name: nombre descriptivo del grupo (para títulos e informes)
        method: método de correlación ('pearson' o 'spearman')
        plot: mostrar o no visualizaciones
        threshold: umbral para identificar correlaciones altas

    Returns:
        Diccionario con resultados del análisis
    """
    # realiza análisis
    df_correlation, summary_stats, top_correlations, corr_metadata = analyze_correlation_group(
        ds_spark, variables_list, group_name, method
    )

    # identifica correlaciones altas
    high_correlations = identify_high_correlations_from_toplist(top_correlations, threshold=threshold)

    # visualización
    if plot:
        plot_correlation_heatmap(df_correlation, group_name, method)

    # resultado
    results = {
        'df_correlation': df_correlation,
        'summary_stats': summary_stats,
        'top_correlations': top_correlations,
        'high_correlations': high_correlations,
        'variables_analyzed': df_correlation.columns.tolist(),
        'group_name': group_name,
        'corr_metadata': corr_metadata
    }

    return results
    

# *** VARIANCE INFLATION FACTOR (VIF) ***


'''
v3 - mejora la gestión de memoria
'''
def calculate_vif(ds: DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """
    Calcula el Variance Inflation Factor (VIF) para una lista de variables en un DataFrame de PySpark.

    El VIF mide la multicolinealidad de las variables dadas con todas las variables del dataset.
    Si sólo se aplica a dos variables, VIF mide la correlación lineal de esas dos variables con todas
    las variables del dataset.

    - VIF = 1: No hay correlación con otras variables
    - VIF = 1-5: Correlación moderada
    - VIF = 5-10: Correlación alta
    - VIF > 10: Correlación muy alta (problema de multicolinealidad)

    Args:
        ds: DataFrame de PySpark con los datos
        feature_columns: lista con los nombres de las columnas

    Returns:
        Diccionario con el VIF de cada variable {variable: vif_value}
    """
    # confirma que se hayan recibido al menos 2 variables
    if len(feature_columns) < 2:
        raise ValueError("=== Se necesitan al menos 2 variables para calcular VIF ===")

    # comprueba que todas las columnas existan en el DataFrame
    missing_columns = [col for col in feature_columns if col not in ds.columns]
    if missing_columns:
        raise ValueError(f"=== Columnas no encontradas: {missing_columns} ===")

    # inicializa resultados
    vif_results = {}

    # recorre las variables
    for target_var in feature_columns:

        # variable actual: variable dependiente, las demás serán variables independientes

        # variables independientes
        independent_vars = [col for col in feature_columns if col != target_var]

        # crea el vector de características independientes
        assembler = VectorAssembler(
            inputCols=independent_vars,
            outputCol="features",
            handleInvalid="skip"  # omite filas con valores nulos
        )

        # ensambla las características
        ds_assembled = assembler.transform(ds).select("features", target_var)

        # prepara la caché en memoria y disco
        ds_assembled = ds_assembled.persist(StorageLevel.MEMORY_AND_DISK)

        # materializa la caché
        ds_assembled.count()

        try:

            # crea el modelo de regresión lineal
            lr = LinearRegression(
                featuresCol="features",
                labelCol=target_var,
                regParam=0.0  # sin regularización para obtener R² puro
            )

            # entrena el modelo
            model = lr.fit(ds_assembled)

            # calcula R²
            r_squared = model.summary.r2

            ## calcula VIF = 1 / (1 - R²)

            # si R² es muy cercano a 1, VIF será muy alto
            vif_value = float('inf') if r_squared >= 0.999999 else 1 / (1 - r_squared)

            vif_results[target_var] = round(vif_value, 4)

        except Exception as e:
            print(f"=== Error calculando VIF para {target_var}: {str(e)} ===")
            vif_results[target_var] = float('nan')

        finally:
            # limpia la memoria
            ds_assembled.unpersist()
            del ds_assembled, assembler, lr, model
            gc.collect()

    return vif_results
    
    
'''
v2 - mejor gestión de memoria
'''
def vif_to_pandas(vif_dict: dict[str, float]) -> pd.DataFrame:
    """
    Convierte el resultado del VIF en un DataFrame de Pandas para mejor visualización.

    Args:
        vif_dict: diccionario con los resultados del VIF

    Returns:
        DataFrame con columnas 'Variable' y 'VIF'
    """
    # convierte a DataFrame de Pandas
    df_result = pd.DataFrame.from_dict(vif_dict, orient='index', columns=['VIF'])
    df_result.index.name = 'Variable'
    df_result.reset_index(inplace=True)

    # ordena por VIF (in-place evita copia completa)
    df_result.sort_values('VIF', ascending=False, inplace=True, ignore_index=True)

    # agrega interpretación del VIF
    def interpret_vif(vif_value):
        if pd.isna(vif_value) or vif_value == float('inf'):
            return "=== Problema en el cálculo ==="
        elif vif_value < 1:
            return "=== Valor inválido ==="
        elif vif_value <= 5:
            return "Baja correlación"
        elif vif_value <= 10:
            return "Correlación moderada"
        else:
            return "Alta multicolinealidad"

    df_result['Interpretacion'] = [interpret_vif(v) for v in df_result['VIF']]

    # limpieza de memoria
    gc.collect()

    return df_result
    
    
def filter_low_vif_variables(vif_dict: dict[str, float], threshold: float = 10.0) -> list[str]:
    """
    Filtra variables con VIF por debajo del umbral especificado.

    Args:
        vif_dict: diccionario con los resultados del VIF
        threshold: float, default=10.0, umbral máximo de VIF aceptable

    Returns:
        Lista de variables con VIF por debajo del umbral
    """
    return [var for var, vif in vif_dict.items()
            if not pd.isna(vif) and vif != float('inf') and vif <= threshold
            ]
            
            
def analyze_vif(ds: DataFrame, feature_columns: list[str]):
    """
    Analiza el Variance Inflation Factor (VIF) para una lista de variables en un DataFrame de PySpark.

    El VIF mide la multicolinealidad entre variables independientes:
    - VIF = 1: No hay correlación con otras variables
    - VIF = 1-5: Correlación moderada
    - VIF = 5-10: Correlación alta
    - VIF > 10: Correlación muy alta (problema de multicolinealidad)

    Args:
        df: DataFrame de PySpark con los datos
        feature_columns: lista con los nombres de las columnas

    Returns:
        None
    """
    # calcula VIF
    vif_results = calculate_vif(ds, feature_columns)

    # muestra resultados ordenados
    vif_df = vif_to_pandas(vif_results)
    print(vif_df)

    # muestra variables con VIF aceptable
    good_vars = filter_low_vif_variables(vif_results, threshold=10.0)
    print("\n=== Variables con VIF aceptable ===\n")
    display(good_vars)

    # muestra variables con VIF no aceptable (>= 10)
    bad_vars = [var for var in feature_columns if var not in good_vars]
    print("\n=== Variables con VIF no aceptable (>= 10) ===\n")
    display(bad_vars)

    # limpia la memoria
    del vif_df, good_vars, bad_vars, vif_results
    gc.collect()
    
    
# *** K-MEANS ***


'''
v4 - mejora la gestión de memoria
'''
def silhouette_score_spark(ds: DataFrame, cols: list[str], range_n_clusters: list[int]):
    """
    Análisis de siluetas para estudiar la distancia de separación entre los grupos resultantes
    de aplicar K-Means en PySpark.

    Args:
        ds: DataFrame de PySpark
        cols: lista de nombres de columnas numéricas para clustering
        range_n_clusters: lista con el número de clusters a probar

    Returns:
        lista de tuplas (n_clusters, silhouette_score)
    """
    # vectoriza las columnas de entrada en una sola columna 'features' y guarda dicha columna en ds_features
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    ds_features = assembler.transform(ds).select("features")

    # prepara la caché en memoria y disco
    ds_features = ds_features.persist(StorageLevel.MEMORY_AND_DISK)
    # materializa la caché
    ds_features.count()

    # evaluador de clustering
    evaluator = ClusteringEvaluator(
        featuresCol = "features",
        predictionCol = "prediction",
        metricName = "silhouette",
        distanceMeasure = "squaredEuclidean"
    )

    # inicializa resultados
    results = []

    try:

      # recorre la lista con el número de clusters a probar
      for k in range_n_clusters:

          # entrena el modelo K-Means
          kmeans = KMeans(k=k, seed=10, featuresCol="features", predictionCol="prediction")
          model = kmeans.fit(ds_features)

          # genera predicciones
          predictions = model.transform(ds_features).select("prediction", "features")

          # comprueba el número real de clusters
          num_clusters_real = predictions.select("prediction").distinct().count()

          # comprueba que haya más de un cluster
          if num_clusters_real > 1:

              # evalúa con silhouette score
              silhouette = evaluator.evaluate(predictions)
              print(f"Para k = {k}, la silueta media es: {silhouette:.4f}")

              # guarda el resultado
              results.append((k, silhouette))

          else:
              print(f"Para k = {k}, sólo se formó 1 cluster. Silhouette no aplicable.")

              # guarda el resultado
              results.append((k, float('nan')))

          # limpieza de memoria
          predictions.unpersist()
          del predictions, model, kmeans
          gc.collect()

    finally:
        # garantiza la limpieza
        ds_features.unpersist()
        del ds_features, assembler, evaluator
        gc.collect()

    return results
    
    
def plot_silhouette_scores(scores):
    """
    Grafica los silhouette scores obtenidos con distintos k.

    Args:
        scores: lista de tuplas (k, silhouette_score)
    """
    ks = [k for k, _ in scores]
    sils = [s for _, s in scores]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Scores for different k (PySpark KMeans)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette score")
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

    # limpia la memoria
    plt.close('all')
    
    
'''
v3 - añade métricas para Gold
'''
def apply_kmeans_spark(ds: DataFrame,
                       variables: list[str],
                       k: int,
                       standardize: bool = True,
                       seed: int = 42,
                       max_iter: int = 100,
                       tol: float = 1e-4
                       ) -> tuple[DataFrame, KMeans, dict]:
    """
    Aplica K-Means clustering a un DataFrame de PySpark.

    Args:
        ds: DataFrame de PySpark con los datos
        variables: lista de nombres de columnas a usar para el clustering
        k: int, número de clusters (k óptimo)
        standardize: bool, default=True; si True, estandariza las variables antes del clustering
        seed: int, optional, default=42, semilla para reproducibilidad
        max_iter: int, default=100, número máximo de iteraciones
        tol: float, default=1e-4, tolerancia para convergencia

    Returns:
        Tuple[DataFrame, KMeans, cluster_stats]
            DataFrame con columna 'cluster' añadida
            Modelo KMeans entrenado
            Diccionario con estadísticas del clustering
    """
    # validaciones
    if not isinstance(ds, DataFrame):
        raise TypeError("El DataFrame debe ser PySpark")

    if not isinstance(variables, list) or not variables:
        raise ValueError("La lista de variables debe ser una lista no vacía de nombres de columnas")

    if k <= 0:
        raise ValueError("k debe ser un entero positivo")

    # verifica que las columnas existan en el DataFrame
    ds_columns = ds.columns
    missing_cols = [col for col in variables if col not in ds_columns]
    if missing_cols:
        raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing_cols}")

    # verifica que no haya valores nulos en las variables seleccionadas
    null_counts = ds.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in variables]
        ).collect()[0]
    null_vars = [var for var in variables if null_counts[var] > 0]
    if null_vars:
        print(f"Advertencia: las siguientes variables contienen valores nulos: {null_vars}")
        print("Se filtrarán las filas con valores nulos...")
        ds = ds.dropna(subset=variables)

    # captura tiempo de inicio
    start_time = time.time()

    # ensambla características en un vector
    assembler = VectorAssembler(
        inputCols = variables,
        outputCol = "features_raw"
    )

    # crea pipeline stages
    stages = [assembler]

    # estandarización (opcional)
    if standardize:
        scaler = StandardScaler(
            inputCol = "features_raw",
            outputCol = "features_scaled",
            withStd = True,
            withMean = True
        )
        stages.append(scaler)

    features_col = "features_scaled" if standardize else "features_raw"

    # configura K-Means
    kmeans = KMeans(
        featuresCol = features_col,
        predictionCol = "cluster",
        k = k,
        seed = seed,
        maxIter = max_iter,
        tol = tol
    )
    stages.append(kmeans)

    # crea y ajusta el pipeline
    pipeline = Pipeline(stages=stages)

    try:
        # entrenamiento
        model = pipeline.fit(ds)
        ds_clustered = model.transform(ds)

        # prepara la caché en memoria y disco
        ds_clustered = ds_clustered.persist(StorageLevel.MEMORY_AND_DISK)
        # materializa la caché
        ds_clustered.count()

        # obtiene el modelo K-Means del pipeline
        kmeans_model = model.stages[-1]

        # limpia columnas intermedias (sólo mantiene la columna 'cluster')
        columns_to_keep = ds.columns + ["cluster"]
        ds_result = ds_clustered.select(*columns_to_keep)

        # limpieza de memoria
        ds_clustered.unpersist(blocking=True)
        del model, pipeline, assembler
        if standardize:
          del scaler
        gc.collect()

        # muestra información del clustering
        print(f"K-Means aplicado con éxito:")
        print(f"- Número de clusters: {k}")
        print(f"- Variables utilizadas: {variables}")
        print(f"- Estandarización: {'Sí' if standardize else 'No'}")
        print(f"- WSSSE (Within Set Sum of Squared Errors): {kmeans_model.summary.trainingCost:.4f}")

        # muestra distribución de clusters y calcula estadísticas
        print("\nDistribución de clusters:")
        cluster_dist = ds_result.groupBy("cluster").count().orderBy("cluster")
        cluster_dist.show(truncate=False)

        # captura tiempo de finalización
        end_time = time.time()
        execution_time = end_time - start_time

        # calcula estadísticas de distribución
        cluster_counts = [row['count'] for row in cluster_dist.collect()]
        total_points = sum(cluster_counts)
        cluster_percentages = {i: (count/total_points)*100 for i, count in enumerate(cluster_counts)}

        # estadísticas de balance
        cluster_stats = {
            'sizes': {i: count for i, count in enumerate(cluster_counts)},
            'percentages': cluster_percentages,
            'min_size': min(cluster_counts),
            'max_size': max(cluster_counts),
            'mean_size': np.mean(cluster_counts),
            'std_size': np.std(cluster_counts),
            'execution_time': execution_time
        }

        print(f"\n⏱️ Tiempo de ejecución: {execution_time:.2f} segundos")
        print(f"📊 Balance de clusters - Min: {cluster_stats['min_size']}, ",
              f"Max: {cluster_stats['max_size']}, Std: {cluster_stats['std_size']:.2f}")

        # limpia la memoria
        del null_counts, cluster_dist, cluster_counts
        gc.collect()

        return ds_result, kmeans_model, cluster_stats

    except Exception as e:
        raise RuntimeError(f"Error durante el proceso de clustering: {str(e)}")


# *** REDUCCIÓN DE DIMENSIONALIDAD ***


'''
v3 - añade métricas para Gold
'''
def calculate_dimensionality_reduction(ds_clustered: DataFrame,
                                        spark: SparkSession,
                                        variables: list[str],
                                        method: str = 'pca',
                                        max_sample_size: int = 10_000,
                                        tsne_max_points: int = 1_000
                                        ) -> tuple:
    """
    Calcula reducción de dimensionalidad (PCA y/o t-SNE) para visualización de clusters

    Args:
        ds_clustered: DataFrame PySpark con columna 'cluster'
        spark: sesión Spark
        variables: lista de variables para la reducción
        method: 'pca', 'tsne', o 'both'
        max_sample_size: tamaño máximo de muestra para visualización
        tsne_max_points: tamaño máximo de muestra para t-SNE

    Returns:
        tuple: (ds_result, method_info, sample_size)
            ds_result: DataFrame PySpark con columnas de componentes y cluster
            method_info: dict con información del método (ej: varianza explicada)
            sample_size: número de puntos en la muestra
    """
    # comprueba si hay que aplicar muestreo
    row_count = ds_clustered.count()
    if row_count > max_sample_size:
        # sí hay muestreo
        fraction = max_sample_size / row_count
        ds_sample = ds_clustered.sample(fraction=fraction, seed=42)
        print(f"⚠️ Muestra de {max_sample_size} puntos de {row_count} totales")
        sample_size = max_sample_size
    else:
        # no hay muestreo
        ds_sample = ds_clustered
        sample_size = row_count

    # inicializa diccionario de tiempos
    execution_times = {}

    # convierte a Pandas para visualización
    df = ds_sample.select(*variables + ['cluster']).toPandas()

    # libera memoria
    del ds_sample
    gc.collect()

    # prepara datos
    X = df[variables].to_numpy()
    clusters = df['cluster'].to_numpy()

    # inicializa resultados
    method_info = {}

    ## calcula PCA si es necesario

    if method in ['pca', 'both']:

        # captura tiempo de inicio PCA
        pca_start = time.time()

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # captura tiempo de fin PCA
        pca_end = time.time()
        execution_times['pca'] = pca_end - pca_start

        # crea DataFrame de Pandas
        df_pca = pd.DataFrame({
            "pca_component_1": X_pca[:, 0],
            "pca_component_2": X_pca[:, 1],
            "cluster": clusters
        })

        # guarda información de PCA
        method_info["pca"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_cumsum": np.cumsum(pca.explained_variance_ratio_),
            "total_variance_explained": sum(pca.explained_variance_ratio_),
            "n_components": 2,
            "execution_time": execution_times['pca'],
            "dataframe": df_pca
        }

        # limpia memoria
        del pca, X_pca
        gc.collect()

    ## calcula t-SNE si es necesario

    if method in ['tsne', 'both']:
        # muestreo adicional para t-SNE si es necesario
        if len(X) > tsne_max_points:
            sample_idx = np.random.choice(len(X), tsne_max_points, replace=False)
            X_tsne_sample = X[sample_idx]
            clusters_tsne = clusters[sample_idx]
        else:
            X_tsne_sample = X
            clusters_tsne = clusters

        # captura tiempo de inicio t-SNE
        tsne_start = time.time()

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne_sample)-1))
        X_tsne = tsne.fit_transform(X_tsne_sample)

        # captura tiempo de fin t-SNE
        tsne_end = time.time()
        execution_times['tsne'] = tsne_end - tsne_start

        # crea DataFrame de Pandas
        df_tsne = pd.DataFrame({
            "tsne_component_1": X_tsne[:, 0],
            "tsne_component_2": X_tsne[:, 1],
            "cluster": clusters_tsne
        })

        # guarda información de tsne
        method_info["tsne"] = {
            "dataframe": df_tsne,
            "sample_size": len(X_tsne_sample),
            "n_components": 2,
            "execution_time": execution_times['tsne']
        }

        # limpia memoria
        del tsne, X_tsne_sample, X_tsne, clusters_tsne
        gc.collect()

    # convierte a DataFrame de PySpark según el método
    if method == 'both':
        # combina ambos DataFrames (mismo número de filas para PCA, subset para t-SNE)
        ds_pca = spark.createDataFrame(method_info['pca']['dataframe'])
        ds_tsne = spark.createDataFrame(method_info['tsne']['dataframe'])
        ds_result = {'pca': ds_pca, 'tsne': ds_tsne}
    elif method == 'pca':
        ds_result = spark.createDataFrame(method_info['pca']['dataframe'])
    elif method == 'tsne':
        ds_result = spark.createDataFrame(method_info['tsne']['dataframe'])
    else:
        raise ValueError(f"Método desconocido: {method}")

    # añade información de muestreo
    method_info['sampling'] = {
        'used_sampling': row_count > max_sample_size,
        'original_size': row_count,
        'sample_size': sample_size,
        'sampling_fraction': sample_size / row_count if row_count > 0 else 1.0
    }

    # limpia memoria
    del df, X, clusters
    gc.collect()

    print("✅ Reducción de dimensionalidad completada")

    if execution_times:
        total_time = sum(execution_times.values())
        #print(f"⏱️ Tiempo total: {total_time:.2f}s")

    return ds_result, method_info, sample_size

        
# *** VISUALIZACIÓN DE CLUSTERING ***


'''
v5 - deja de llamar a calculate_dimensionality_reduction y recibe sus resultados
como parámetros
'''
def plot_clusters_2d(ds_clustered: DataFrame,
                     variables: list[str],
                     method: str = 'pca',
                     figsize: tuple[int, int] = (12, 8),
                     save_path: str = None,
                     spark: SparkSession = None,
                     ds_result: Any = None,
                     method_info: dict | None = None,
                     sample_size: int | None = None
                     ) -> None:
    """
    Crea visualizaciones 2D de los clusters usando reducción de dimensionalidad

    Args:
        ds_clustered: DataFrame PySpark con columna 'cluster'
        variables: lista de variables utilizadas en el clustering
        method: str, método de reducción: 'pca' (default), 'tsne', o 'both'
        figsize: Tuple[int, int], tamaño de la figura
        save_path: str, optional, ruta para guardar la figura
        spark: sesión Spark; no necesario si se recibe ds_result
        
        Resultados de calculate_dimensionality_reduction:
        ds_result: DataFrame PySpark con columnas de componentes y cluster
        method_info: dict con información del método (ej: varianza explicada)
        sample_size: número de puntos en la muestra
    """
    # comprueba si se reciben los resultados de calculate_dimensionality_reduction
    if ds_result is None or method_info is None or sample_size is None:
        if spark is None:
            raise ValueError(
                "Se necesita 'spark' si no se proporcionan ds_result/method_info/sample_size"
            )
        
        # calcula reducción de dimensionalidad
        ds_result, method_info, sample_size = calculate_dimensionality_reduction(
            ds_clustered, variables, method=method
        )

    # convierte a Pandas para graficar
    if method == 'both':
        df_pca = ds_result['pca'].toPandas()
        df_tsne = ds_result['tsne'].toPandas()
    elif method == 'pca':
        df_pca = ds_result.toPandas()
    elif method == 'tsne':
        df_tsne = ds_result.toPandas()

    def plot_pca(ax, df, colors, unique_clusters, pca_info):
        """
        Grafica PCA
        """
        for i, cluster in enumerate(unique_clusters):
            mask = df["cluster"] == cluster
            df_cluster = df[mask]
            ax.scatter(
                df_cluster["pca_component_1"], df_cluster["pca_component_2"],
                c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50
            )

        explained_var = pca_info['explained_variance_ratio']

        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} varianza)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} varianza)')
        ax.set_title('Clusters - Análisis de Componentes Principales (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_tsne(ax, df, colors, unique_clusters, tsne_info):
        """
        Grafica t-SNE
        """
        for i, cluster in enumerate(unique_clusters):
            mask = df["cluster"] == cluster
            df_cluster = df[mask]
            if len(df_cluster) > 0:
                ax.scatter(
                    df_cluster["tsne_component_1"], df_cluster["tsne_component_2"],
                    c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50
                )

        ax.set_xlabel('t-SNE Dimensión 1')
        ax.set_ylabel('t-SNE Dimensión 2')
        ax.set_title('Clusters - t-SNE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # configura colores
    if method == 'both':
        unique_clusters = sorted(df_pca['cluster'].unique())
    elif method == 'pca':
        unique_clusters = sorted(df_pca['cluster'].unique())
    else:
        unique_clusters = sorted(df_tsne['cluster'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

    ## PCA y t-SNE

    if method == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        plot_pca(ax1, df_pca, colors, unique_clusters, method_info["pca"])
        plot_tsne(ax2, df_tsne, colors, unique_clusters, method_info["tsne"])

    else:
        fig, ax = plt.subplots(figsize=figsize)

        ## PCA

        if method == 'pca':
            plot_pca(ax, df_pca, colors, unique_clusters, method_info["pca"])

        ## t-SNE

        elif method == 'tsne':
            plot_tsne(ax, df_tsne, colors, unique_clusters, method_info["tsne"])

        else:
            raise ValueError(f"Método desconocido: {method}")

    plt.tight_layout()

    # guarda el resultado si se solicita
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # muestra la gráfica
    plt.show()

    # limpia la memoria
    if method == 'both':
        del df_pca, df_tsne
    elif method == 'pca':
        del df_pca
    else:
        del df_tsne
    del ds_result, colors, unique_clusters, method_info
    gc.collect()
    # cierra las figuras
    plt.close('all')


'''
v3 - mejora la gestión de memoria
'''
def plot_cluster_profiles(ds_clustered: DataFrame,
                         variables: list[str],
                         figsize: tuple[int, int] = (15, 8),
                         save_path: str = None) -> None:
    """
    Crea gráficos de perfil de clusters (radar chart y barras).

    Args:
        ds_clustered: dataFrame con columna 'cluster'
        variables: lista de variables para el perfil
        figsize: Tuple[int, int], tamaño de la figura
        save_path: str, optional, ruta para guardar la figura
    """
    # calcula estadísticas por cluster
    cluster_stats = ds_clustered.groupBy('cluster').agg(
        *[avg(var).alias(f'{var}_mean') for var in variables],
        count('*').alias('count')
    ).orderBy('cluster')

    # persiste mientras se trae a Pandas
    cluster_stats_spark = cluster_stats.cache()
    cluster_stats = cluster_stats_spark.toPandas()

    # limpia la memoria
    cluster_stats_spark.unpersist()
    del cluster_stats_spark
    gc.collect()

    # prepara datos para el gráfico
    mean_cols = [f'{var}_mean' for var in variables]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ## gráfico de barras agrupadas

    x = np.arange(len(variables))
    width = 0.8 / len(cluster_stats)

    for i, row in cluster_stats.iterrows():
        cluster_id = row['cluster']
        values = [row[col] for col in mean_cols]
        ax1.bar(x + i * width, values, width,
                label=f'Cluster {cluster_id} (n={row["count"]})',
                alpha=0.8)

    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Valor Promedio')
    ax1.set_title('Perfil de Clusters - Valores Promedio')
    ax1.set_xticks(x + width * (len(cluster_stats) - 1) / 2)
    ax1.set_xticklabels(variables, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ## mapa de calor de los perfiles

    heatmap_data = cluster_stats[mean_cols].to_numpy()

    cmap = sns.diverging_palette(20, 220, n=256, as_cmap=True)
    im = ax2.imshow(heatmap_data, cmap=cmap, aspect='auto')
    ax2.set_xticks(range(len(variables)))
    ax2.set_xticklabels(variables, rotation=45)
    ax2.set_yticks(range(len(cluster_stats)))
    ax2.set_yticklabels([f'Cluster {c}' for c in cluster_stats['cluster']])
    ax2.set_title('Mapa de Calor - Perfiles de Clusters')

    # añade valores en el mapa de calor
    for i in range(len(cluster_stats)):
        for j in range(len(variables)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="white", fontsize=10)

    # barra de color
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    # guarda la imagen si se solicita
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # muestra la gráfica
    plt.show()

    # limpia la memoria
    del cluster_stats, mean_cols, heatmap_data, cmap, im, x, width
    gc.collect()
    # cierra las figuras
    plt.close('all')
    
    
'''
v2 - mejora la gestión de memoria
'''
def plot_cluster_distribution(ds_clustered: DataFrame,
                             figsize: tuple[int, int] = (10, 6),
                             save_path: str = None) -> None:
    """
    Visualiza la distribución de observaciones por cluster.

    Args:
        ds_clustered: DataFrame con columna 'cluster'
        figsize: Tuple[int, int], tamaño de la figura
        save_path: str, optional, ruta para guardar la figura
    """
    # obtiene conteos por cluster
    cluster_counts_spark = ds_clustered.groupBy('cluster').count().orderBy('cluster')

    # persiste mientras se trae a Pandas
    cluster_counts = cluster_counts_spark.toPandas()

    # limpia la memoria
    cluster_counts_spark.unpersist()
    del cluster_counts_spark
    gc.collect()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

    ## gráfico de barras

    bars = ax1.bar(cluster_counts['cluster'].astype(str), cluster_counts['count'], color=colors)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Número de Observaciones')
    ax1.set_title('Distribución de Observaciones por Cluster')
    ax1.grid(True, alpha=0.3)

    # añade valores en las barras
    max_count = max(cluster_counts['count'])
    for bar, count in zip(bars, cluster_counts['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max_count,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    ## gráfico de pastel

    ax2.pie(cluster_counts['count'],
            labels=[f'Cluster {c}' for c in cluster_counts['cluster']],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
    )
    ax2.set_title('Distribución Porcentual por Cluster')

    plt.tight_layout()

    # guarda la imagen si se solicita
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # muestra la gráfica
    plt.show()

    # limpia la memoria
    del cluster_counts, bars, colors, max_count
    gc.collect()
    plt.close('all')
    
    
'''
v2 - añade t-SNE
'''
def visualize_clusters_complete(ds_clustered: DataFrame,
                                variables: list[str],
                                save_plots: bool = False,
                                output_dir: str = "./plots/",
                                method: str = "pca",
                                spark: SparkSession = None,
                                ds_result: Any = None,
                                method_info: dict | None = None,
                                sample_size: int | None = None
) -> None:
    """
    Crea un conjunto completo de visualizaciones para los clusters.

    Args:
        ds_clustered: DataFrame con columna 'cluster'
        variables: lista de variables utilizadas en clustering
        save_plots: bool, default=False, guardar o no los gráficos
        output_dir: str, nombre del directorio donde guardar los gráficos
        method: str, default='pca', método de reducción: 'pca', 'tsne', o 'both'
        spark: sesión Spark; no necesario si se recibe ds_result
        
        Resultados de calculate_dimensionality_reduction:
        ds_result: DataFrame PySpark con columnas de componentes y cluster
        method_info: dict con información del método (ej: varianza explicada)
        sample_size: número de puntos en la muestra
    """
    # comprueba si se reciben los resultados de calculate_dimensionality_reduction
    if ds_result is None or method_info is None or sample_size is None:
        if spark is None:
            raise ValueError(
                "Se necesita 'spark' si no se proporcionan ds_result/method_info/sample_size"
            )
        
        # calcula reducción de dimensionalidad
        ds_result, method_info, sample_size = calculate_dimensionality_reduction(
            ds_clustered, variables, method=method
        )

    # prepara directorio si se solicita guardar los gráficos
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    print("🎨 Generando visualizaciones de clusters...\n")

    # 1. Visualización 2D
    print("1. Gráfico 2D...")
    plot_clusters_2d(
        ds_clustered, 
        variables, 
        method=method,
        save_path=f"{output_dir}/clusters_pca.png" if save_plots else None,
        spark=spark,
        ds_result=ds_result,
        method_info=method_info,
        sample_size=sample_size
    )

    # 2. Perfiles de clusters
    print("2. Perfiles de clusters...")
    plot_cluster_profiles(ds_clustered, variables,
                         save_path=f"{output_dir}/clusters_profiles.png" if save_plots else None)

    # 3. Distribución de clusters
    print("3. Distribución de clusters...")
    plot_cluster_distribution(ds_clustered,
                             save_path=f"{output_dir}/clusters_distribution.png" if save_plots else None)

    print("✅ Visualizaciones completadas!")

    
'''
v2 - mejora la gestión de memoria
'''
def plot_clusters_3d(
    ds, cluster_col='cluster', x_col='passengers', y_col='distance', z_col='duration', name_fig='Clusters en 3D'
    ):
    """
    Grafica un scatter 3D de clusters desde un DataFrame de PySpark.

    Args:
        ds: DataFrame de PySpark
        cluster_col: str, columna con la asignación de cluster
        x_col, y_col, z_col: str, columnas a graficar en los ejes X, Y y Z
        name_fig: str, nombre del gráfico

    Returns:
        None
    """
    # convierte a Pandas para visualización con muestreo de datos para DataFrames grandes
    row_count = ds.count()

    max_points = 5_000
    if row_count > max_points:
        fraction = max_points / row_count
        df = ds.select(
            cluster_col, x_col, y_col, z_col
            ).sample(fraction=fraction, seed=42).toPandas()
        print(f"⚠️ Mostrando muestra de {max_points} puntos de {row_count} totales")
    else:
        df = ds.select(cluster_col, x_col, y_col, z_col).toPandas()

    # prepara la figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # obtiene los clusters únicos
    clusters = df[cluster_col].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))

    # recorre los clusters
    for c, color in zip(clusters, colors):
        subset = df[df[cluster_col] == c]
        # grafica el cluster actual
        ax.scatter(
            subset[x_col], subset[y_col], subset[z_col], label=f'Cluster {c}', alpha=0.6,
            color=color
                   )

    # etiquetas
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.legend()
    plt.title(name_fig)

    # muestra el gráfico
    plt.show()

    # limpia la memoria
    del df, clusters, colors, fig, ax
    gc.collect()
    plt.close('all')


# *** PREPARACIÓN DATOS PARA GOLD ***


def generate_metadata(
    group_name: str, 
    model_type: str, 
    row_count: int, 
    parameters: dict,
    spark: SparkSession = None
    ) -> dict:
    """
    Genera metadatos completos para reproducibilidad del análisis.

    Args:
      group_name: str, nombre del grupo de variables
      model_type: str, tipo de modelo ("kmeans", "pca", "tsne", "correlation")
      row_count: int, número de filas del dataset
      parameters: dict, hiperparámetros del modelo/análisis
      spark: sesión Spark

    Returns:
      dict con timestamp, versión hash y metadatos completos
    """
    # fecha y hora actual
    timestamp = datetime.now().isoformat()

    # genera hash único basado en parámetros + timestamp
    metadata_str = json.dumps({
        "group": group_name,
        "model": model_type,
        "params": parameters,
        "timestamp": timestamp
    }, sort_keys=True)

    # hash corto: 12 primeros caracteres del MD5 del hash único
    version_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]

    # diccionario de respuesta
    metadata = {
        "group_name": group_name,
        "model_type": model_type,
        "row_count": row_count,
        "parameters": parameters,
        "processing_date": timestamp,
        "model_version": version_hash,
        "featureset_version": f"{group_name}_{version_hash}"
    }

    # añade información adicional de configuración
    metadata["python_version"] = sys.version.split()[0]
    if spark is None:
        metadata["spark_version"] = "unknown"
    else:
        metadata["spark_version"] = spark.version

    return metadata
    
    
def save_features_table(
    ds: DataFrame,
    feature_columns: list,
    metadata: dict,
    trip_ids: str = None,
    group_name: str = 'Variable',
    output_path: str = "./silver_data",
    mode: str = 'overwrite'
    ) -> str:
    """
    Guarda las features del grupo con metadatos en formato Parquet (PySpark)

    Args:
      ds: DataFrame PySpark con los datos
      feature_columns: list, columnas a guardar
      metadata: dict, metadatos generados por generate_metadata
      trip_ids: str (nombre de columna con id de registro) o None (genera IDs automáticos)
      group_name: str, nombre del grupo
      output_path: str, ruta base donde guardar
      mode: str, modo de guardado ('overwrite' o 'append')

    Returns:
      str con la ruta del archivo guardado
    """
    # si no hay trip_ids, genera IDs únicos
    if trip_ids is None:
        ds_features = ds.withColumn('trip_id', monotonically_increasing_id())
    else:
        ds_features = ds.withColumn('trip_id', col(trip_ids))

    # columnas a conservar: trip_id y las features del grupo
    columns_to_keep = ['trip_id'] + feature_columns
    # descarta el resto de columnas
    ds_features = ds_features.select(columns_to_keep)

    # añade columnas de metadatos
    ds_features = ds_features \
        .withColumn('group_name', lit(group_name)) \
        .withColumn('featureset_version', lit(metadata['featureset_version'])) \
        .withColumn('fecha_ingesta', lit(metadata['processing_date'])) \
        .withColumn('metadata_json', lit(json.dumps(metadata)))

    # nombre de archivo, incluye fecha/hora actuales
    processing_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_group = group_name[:20].replace(" ", "_").lower()
    filename = f"{short_group}__silver_features_{processing_ts}.parquet"
    filepath = f"{output_path}/{filename}"

    # guarda en Parquet
    ds_features.write.mode(mode).parquet(filepath)

    # número de registros guardados
    record_count = ds_features.count()
    # número de columnas guardadas
    column_count = len(ds_features.columns)

    # muestra información del resultado
    print(f"✓ Features guardadas:\n{filepath}")
    print(f"  Registros: {record_count:,}")
    print(f"  Columnas: {column_count}")

    return filepath
    
    
'''
v4 - corrige errores
'''
def compute_quality_metrics(
    ds: DataFrame, feature_columns: list, group_name: str, approx_error: float=0.01
) -> dict:
    """
    Calcula métricas de calidad de datos para las features del grupo.

    Args:
      ds: DataFrame de PySpark con los datos
      feature_columns: list, nombres de las columnas a analizar
      group_name: str, nombre del grupo
      approx_error: float, error aproximado para percentiles

    Returns:
      dict con métricas de calidad por variable
    """
    # registros del DataFrame
    total_rows = ds.count()
    # inicializa el diccionario de resultaods
    quality_metrics = {
        "group_name": group_name,
        "total_rows": total_rows,
        "variables": {}
    }

    ## agregación única para n_nulls, n_valid, min, max, mean, std de todas las columnas

    # resultados de agregación
    agg_exprs = []

    # recorre las columnas a analizar
    for col_name in feature_columns:
        # agregaciones básicas en una sola consulta
        agg_exprs += [
            spark_sum(when(col(col_name).isNull(), 1).otherwise(0)).alias(f"{col_name}__n_nulls"),
            spark_sum(when(col(col_name).isNotNull(), 1).otherwise(0)).alias(f"{col_name}__n_valid"),
            spark_min(col(col_name)).alias(f"{col_name}__min"),
            spark_max(col(col_name)).alias(f"{col_name}__max"),
            avg(col(col_name)).alias(f"{col_name}__mean"),
            stddev(col(col_name)).alias(f"{col_name}__std"),
        ]

    # comprueba que se haya obtenido la agregación
    if agg_exprs:
        # agregación única sobe el DataFrame
        agg_row_all = ds.agg(*agg_exprs).collect()[0]
        # convierte en diccionario
        agg_row_all_dict = agg_row_all.asDict()
    else:
        agg_row_all_dict = {}

    ## approxQuantile en lote para todas las columnas (si falla, lo deja vacío)

    # inicializa resultados
    quantiles_res = {}

    # recorre las columnas
    for col_name in feature_columns:

        # intenta obtener cuartiles/mediana con approxQuantile (numérico)
        # si alguno no es numérico, approxQuantile puede fallar -> captura el error
        try:
            q = ds.approxQuantile(col_name, [0.25, 0.5, 0.75], approx_error)

            if q is not None and len(q) == 3:
                # guarda los resultados por columna
                quantiles_res[col_name] = (q[0], q[1], q[2])

        except Exception:
            # columna no numérica u otro error
            quantiles_res[col_name] = (None, None, None)

    ## calcula outliers sólo para las columnas que tienen cuantiles válidos

    # inicializa las expresiones de agregación para contar outliers
    outliers_exprs = []

    # recorre todas las columnas
    for col_name in feature_columns:

        # cuantiles previamente calculados
        q25, _, q75 = quantiles_res.get(col_name, (None, None, None))

        if q25 is not None and q75 is not None:
            # sí existen los cuantiles: calcula límites inferior y superior para outliers
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q25 + 1.5 * iqr
            # condición True si es outlier
            cond = (col(col_name).cast("double").isNotNull()) & (
                (col(col_name).cast("double") < lit(lower)) | (col(col_name).cast("double") > lit(upper))
            )
            # añade expresión de agregación con núm de filas que cumplen cond (outliers)
            outliers_exprs.append(
                spark_sum(when(cond, 1).otherwise(0)).alias(f"{col_name}__n_outliers_iqr")
            )

    # inicializa dict con los resultados
    outliers_dict = {}

    if outliers_exprs:
        # ejecuta la agregación para todas las columnas con outliers
        outliers_row = ds.agg(*outliers_exprs).collect()[0]
        # convierte en diccionario
        outliers_dict = outliers_row.asDict()

    ## ensambla métricas por columna usando los resultados agregados

    def to_float(x):
        # convierte a float
        try:
            return float(x) if x is not None else None
        except Exception:
            # None si no es convertible
            return None

    # recorre las columnas
    for col_name in feature_columns:
        # valores nulos
        n_nulls = int(agg_row_all_dict.get(f"{col_name}__n_nulls", 0))
        # valores válidos
        n_valid = int(agg_row_all_dict.get(f"{col_name}__n_valid", 0))

        # obtiene los cuantines
        q25, q50, q75 = quantiles_res.get(col_name, (None, None, None))

        # inicializa el número de outliers
        n_outliers_iqr = None
        # alias del contador de outliers
        out_alias = f"{col_name}__n_outliers_iqr"
        # comprueba que la columna tenga outliers
        if out_alias in outliers_dict and outliers_dict[out_alias] is not None:

            try:
                # convierte el número de outliers a entero
                n_outliers_iqr = int(outliers_dict[out_alias])
            except Exception:
                # el número de outliers no es un entero válido
                n_outliers_iqr = None

        # métricas para el diccionario
        min_val = to_float(agg_row_all_dict.get(f"{col_name}__min"))
        max_val = to_float(agg_row_all_dict.get(f"{col_name}__max"))
        mean_val = to_float(agg_row_all_dict.get(f"{col_name}__mean"))
        std_val = to_float(agg_row_all_dict.get(f"{col_name}__std"))

        # diccionario de métricas de la columna
        metrics = {
            "n_nulls": n_nulls,
            "pct_nulls": (n_nulls / total_rows) * 100 if total_rows > 0 else None,
            "n_valid": n_valid,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": to_float(q50),
            "std": std_val,
            "q25": to_float(q25),
            "q75": to_float(q75),
            "n_outliers_iqr": n_outliers_iqr
        }

        # añade la columna al diccionario de resultados
        quality_metrics["variables"][col_name] = metrics

    # añade contador de variables
    quality_metrics['n_variables'] = len(feature_columns)

    return quality_metrics
    

def attach_correlations_to_quality_metrics(
    quality_metrics: dict,
    analyze_results: dict,
    method: str = 'pearson',
    top_n: int = 5
):
    """
    Inserta summary_stats y top_correlations en quality_metrics['correlations'][method].

    Args:
      quality_metrics: dict devuelto por compute_quality_metrics
      analyze_results: dict devuelto por analyze_correlations
      method: 'pearson' o 'spearman'
      top_n: número de pares a mantener en top_correlations embebido

    Returns:
      quality_metrics: dict modificado con summary_stats y top_correlations
    """
    # resumen de estadísticas
    summary = analyze_results.get("summary_stats")

    # lista de pares más correlacionados
    top_list_full = analyze_results.get("top_correlations", [])
    # recorta la lista de pares según lo solicitado
    top_list = top_list_full[:top_n] if top_list_full else []

    # metadatos de correlación
    meta = analyze_results.get("corr_metadata", {})

    # crea 'correlations' si no existe
    quality_metrics.setdefault("correlations", {})
    # crea la entrada del método si no existe
    quality_metrics["correlations"].setdefault(method, {})
    # asigna valores de resumen, lista de pares y metadatos
    quality_metrics["correlations"][method]["summary_stats"] = summary
    quality_metrics["correlations"][method]["top_correlations"] = top_list
    quality_metrics["correlations"][method]["metadata"] = meta

    # añade referencias al tamaño de la matriz de correlación
    if analyze_results.get("df_correlation") is not None:
        # número de variables
        n_vars = len(analyze_results["df_correlation"].columns)
        # dimensiones de la matriz
        quality_metrics["correlations"][method]["matrix_shape"] = {
            "n_variables": n_vars,
            "n_pairs": n_vars * (n_vars - 1) // 2
        }

    return quality_metrics
    
    
def save_quality_metrics(
    quality_metrics: dict,
    output_path: str,
    spark: SparkSession,
    format: str = "json",
    mode: str = "overwrite",
    group_name: str = "Variables"
    ):
    """
    Guarda quality_metrics en formato JSON (por defecto) o Parquet

    Args:
      quality_metrics: dict a persistir
      output_path: ruta destino
      spark: sesión Spark
      format: 'json' o 'parquet' (si parquet, crea un one-row dataframe)
      mode: 'overwrite' o 'append'
      group_name: nombre descriptivo del grupo (para títulos e informes)

    Returns:
      dict {'path','format','checksum'}
    """
    # copia referencia al diccionario de métricas
    payload = quality_metrics

    # nombre de archivo, incluye fecha/hora actuales
    processing_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_group = group_name[:20].replace(" ", "_").lower()
    filename = f"{short_group}__silver_quality_metrics_{processing_ts}"

    ## salida JSON

    if format == "json":

        filepath = f"{output_path}/{filename}.json"

        # serializa payload a JSON, str cualquier objeto no serializable
        json_str = json.dumps(payload, default=str)
        # crea un RDD con una única entrada y lo guarda como texto
        if mode == "overwrite" and os.path.exists(filepath):
            # borra el directorio
            shutil.rmtree(filepath)

        rdd = spark.sparkContext.parallelize([json_str], 1)

        if mode == 'append' and os.path.exists(filepath):
            # carga el fichero existente
            existing = spark.read.text(filepath).rdd.map(lambda r: r[0]).collect()
            rdd = spark.sparkContext.parallelize(existing + [json_str], 1)

        rdd.saveAsTextFile(filepath)

        # usará el JSON serializado para calcular el hash
        content_for_hash = json_str

    ## salida Parquet

    else:
        # parquet: transforma dict a single-row DataFrame
        filepath = f"{output_path}/{filename}.parquet"

        # convierte dict a DataFrame plano de Pandas (1 registro, 1 columna por clave del dict)
        df = pd.json_normalize(payload)
        # convierte a DataFrame de PySpark y lo escribe como Parquet
        spark.createDataFrame(df).write.mode(mode).parquet(filepath)

        # usará los primeros 1000 elementos del array plano para el hash
        content_for_hash = str(df.values.flatten()[:1000])

    # calcula el MD5 y obtiene su hex digest como checksum
    checksum = hashlib.md5(content_for_hash.encode("utf-8")).hexdigest()

    return {"path": output_path, "format": format, "checksum": checksum}
    

'''
v2 - añade métricas para Gold
'''
def compute_model_metrics(
    model,
    ds_clustered: DataFrame,
    silhouette_scores: list,
    k_optimo: int,
    cluster_stats: dict = None,
    metric_type: str = "clustering"
    ) -> dict:
    """
    Calcula métricas del modelo de clustering

    Args:
      model: modelo KMeans entrenado (modelo_kmeans)
      ds_clustered: DataFrame con columna 'prediction' (ds_con_clusters)
      silhouette_scores: lista de tuplas [(k, score), ...] de silhouette_score_spark
      k_optimo: int, número de clusters del modelo final
      cluster_stats: dict, estadísticas del clustering
      metric_type: str, tipo de métricas ('clustering', 'dimensionality_reduction')
          Nota: sólo preparado para 'clustering'

    Returns:
      dict con métricas del modelo
    """
    ## clustering

    if metric_type == "clustering":

        # busca silhouette del k óptimo, None si no lo encuentra
        silhouette_optimo = next((score for k, score in silhouette_scores if k == k_optimo), None)

        # calcula la inercia (WSSSE)
        wssse = model.summary.trainingCost

        # diccionario con las métricas relevantes
        metrics = {
            "n_clusters": k_optimo,
            "silhouette_score": silhouette_optimo,
            "wssse": wssse,
            "cluster_sizes": ds_clustered.groupBy("cluster").count().collect(),
            "silhouette_all_k": {k: score for k, score in silhouette_scores},
            "centers": [center.tolist() for center in model.clusterCenters()]
        }

        # añade estadísticas de distribución si están disponibles
        if cluster_stats:
            metrics.update({
                "cluster_percentages": cluster_stats['percentages'],
                "min_cluster_size": cluster_stats['min_size'],
                "max_cluster_size": cluster_stats['max_size'],
                "mean_cluster_size": cluster_stats['mean_size'],
                "std_cluster_size": cluster_stats['std_size'],
                "execution_time_clustering": cluster_stats['execution_time']
            })

        return metrics

    ## reducción de dimensionalidad

    elif metric_type == "dimensionality_reduction":
        # no implementado
        return {}

    else:
        # tipo de métrica no válido: devuelve diccionario vacío
        return {}
        
        
def save_clusters_table(
    ds_clustered: DataFrame,
    model_name: str,
    metadata: dict,
    model_metrics: dict,
    trip_ids: str = None,
    group_name: str = "Variables",
    output_path: str = "./silver_data",
    mode: str = "overwrite"
    ) -> str:
    """
    Guarda los resultados de clustering con metadatos en formato Parquet (PySpark)

    Args:
      ds_clustered: DataFrame de PySpark con columna 'prediction'
      model_name: str, identificador del modelo (ej. "kmeans_k5")
      metadata: dict, metadatos generados por generate_metadata
      model_metrics: dict, métricas del modelo (de compute_model_metrics)
      trip_ids: str (nombre de columna con id del registro) o None (genera IDs automáticos)
      group_name: str, nombre del grupo
      output_path: str, ruta base donde guardar
      mode: str, modo de guardado ('overwrite' o 'append')

    Returns:
      str con la ruta del archivo guardado
    """
    # si no hay trip_ids, genera IDs únicos
    if trip_ids is None:
        ds_clusters = ds_clustered.withColumn('trip_id', monotonically_increasing_id())
    else:
        ds_clusters = ds_clustered.withColumn('trip_id', col(trip_ids))

    # selecciona trip_id y prediction (cluster_id)
    ds_clusters = ds_clusters.select('trip_id', 'cluster').withColumnRenamed('cluster', 'cluster_id')

    # añade columnas de metadatos
    ds_clusters = ds_clusters \
        .withColumn('group_name', lit(group_name)) \
        .withColumn('model_name', lit(model_name)) \
        .withColumn('model_version', lit(metadata['model_version'])) \
        .withColumn('fecha_modelo', lit(metadata['processing_date'])) \
        .withColumn('silhouette_score', lit(float(model_metrics['silhouette_score']))) \
        .withColumn('n_clusters', lit(model_metrics['n_clusters']))

    # nombre del archivo, incluye fecha/hora actuales
    processing_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_group = group_name[:20].replace(" ", "_").lower()
    filename = f"{short_group}__silver_clusters_{processing_ts}.parquet"
    filepath = f"{output_path}/{filename}"

    # guarda en formato Parquet
    ds_clusters.write.mode(mode).parquet(filepath)

    # total de registros
    record_count = ds_clusters.count()

    # muestra estadísticas
    print(f"✓ Clusters guardados:\n{filepath}")
    print(f"  Registros: {record_count:,}")
    print(f"  Clusters únicos: {model_metrics['n_clusters']}")

    return filepath
    
    
def prepare_dimensionality_metrics(method_info: dict, method: str) -> dict:
    """
    Prepara métricas de reducción de dimensionalidad

    Args:
        method_info: dict retornado por calculate_dimensionality_reduction
        method: str, 'pca', 'tsne', o 'both'

    Returns:
        dict con métricas formateadas para log
    """
    # inicializa resultados
    metrics = {}

    # PCA
    if method in ['pca', 'both'] and 'pca' in method_info:
        pca_info = method_info['pca']
        metrics.update({
            'variance_explained': pca_info['explained_variance_ratio'].tolist(),
            'total_variance_explained': float(pca_info['total_variance_explained']),
            'n_components': pca_info['n_components'],
            'execution_time_pca': pca_info['execution_time']
        })

    # t-SNE
    if method in ['tsne', 'both'] and 'tsne' in method_info:
        tsne_info = method_info['tsne']
        metrics.update({
            'tsne_sample_size': tsne_info['sample_size'],
            'n_components_tsne': tsne_info['n_components'],
            'execution_time_tsne': tsne_info['execution_time']
        })

    # muestreo
    if 'sampling' in method_info:
        metrics['sampling_info'] = method_info['sampling']

    return metrics
    
    
'''
v2 - preparada para recibir DataFrame de PySpark con columnas pca_components_1 y _2
'''
def save_transformations_table(
    ds_transformed: DataFrame,
    metadata: dict,
    n_components: int,
    trip_ids: str = None,
    group_name: str = 'Variables',
    transformation_type: str = 'pca',
    variance_explained: list = None,
    output_path: str = "./silver_data",
    mode: str = 'overwrite'
    ) -> str:
    """
    Guarda las transformaciones dimensionales (PCA/t-SNE) en formato Parquet (PySpark)

    Args:
      ds_transformed: DataFrame PySpark con columnas de componentes ya calculadas
      metadata: dict, metadatos generados
      n_components: int, número de componentes/dimensiones
      trip_ids: str (nombre de columna de id de registro) o None (genera IDs automáticos)
      group_name: str, nombre del grupo
      transformation_type: str, "pca" o "tsne"
      variance_explained: list opcional, varianza explicada por cada componente (solo PCA)
      output_path: str, ruta base donde guardar
      mode: str, modo de guardado ('overwrite' o 'append')

    Returns:
      str con la ruta del archivo guardado
    """
    # si no hay trip_ids, genera IDs únicos
    if trip_ids is None:
        ds_result = ds_transformed.withColumn('trip_id', monotonically_increasing_id())
    else:
        ds_result = ds_transformed.withColumn('trip_id', col(trip_ids))

    # nombres de las columnas de componentes según el tipo de transformación
    component_cols_original = [f"{transformation_type}_component_{i+1}" for i in range(n_components)]
    component_cols_renamed = [f"component_{i+1}" for i in range(n_components)]

    # renombra componentes para formato estándar
    for orig, renamed in zip(component_cols_original, component_cols_renamed):
        ds_result = ds_result.withColumnRenamed(orig, renamed)

    # selecciona columnas relevantes (incluye cluster si existe)
    select_cols = ['trip_id'] + component_cols_renamed
    if 'cluster' in ds_result.columns:
        select_cols.append('cluster')
    ds_result = ds_result.select(select_cols)

    # añade metadatos
    ds_result = ds_result \
        .withColumn('group_name', lit(group_name)) \
        .withColumn('transformation_type', lit(transformation_type)) \
        .withColumn('model_version', lit(metadata['model_version'])) \
        .withColumn('fecha_modelo', lit(metadata['processing_date'])) \
        .withColumn('n_components', lit(n_components))

    if variance_explained and transformation_type == "pca":
        ds_result = ds_result.withColumn('variance_explained', lit(str(variance_explained)))

    # nombre del archivo, incluye fecha/hora actuales
    processing_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_group = group_name[:20].replace(" ", "_").lower()
    filename = f"{short_group}__silver_transformations_{transformation_type}_{processing_ts}.parquet"
    filepath = f"{output_path}/{filename}"

    # guardado
    ds_result.write.mode(mode).parquet(filepath)

    # total de registros
    record_count = ds_result.count()

    # muestra resultados
    print(f"✓ Transformaciones guardadas:\n{filepath}")
    print(f"  Registros: {record_count:,}")
    print(f"  Componentes: {n_components}")
    if variance_explained:
        print(f"  Varianza explicada: {[f'{v:.3f}' for v in variance_explained]}")

    return filepath
    
    
# *** LOGGING Y LIMPIEZA ***


def save_log_metrics(
    model_type: str,
    quality_metrics: dict,
    model_metrics: dict,
    metadata: dict,
    artifact_paths: dict,
    group_name: str = 'Variables',
    output_path: str = './silver_data',
    mode: str = 'overwrite'
    ) -> None:
    """
    Guarda todas las métricas del análisis en un log centralizado

    Args:
      model_type: str, tipo de análisis/modelo
      quality_metrics: dict, métricas de calidad de datos
      model_metrics: dict, métricas del modelo (clustering, PCA, etc.)
      metadata: dict, metadatos del proceso
      artifact_paths: dict, rutas de los archivos generados
      group_name: str, nombre del grupo
      output_path: str, ruta base donde guardar el log
      mode: str, modo de guardado ('overwrite' o 'append')
    """
    # construye un registro plano
    log_entry = {
        # identificación
        "group_name": group_name,
        "model_type": model_type,
        "model_version": metadata["model_version"],
        "featureset_version": metadata["featureset_version"],
        "processing_date": metadata["processing_date"],

        # información de entorno
        "python_version": metadata.get("python_version"),
        "spark_version": metadata.get("spark_version"),
        "row_count": metadata["row_count"],

        # resumen de métricas de calidad
        "total_rows": quality_metrics["total_rows"],
        "n_variables": len(quality_metrics["variables"]),
    }

    # añade métricas de calidad por variable
    total_nulls = int(sum((v.get("n_nulls") or 0) for v in quality_metrics["variables"].values()))
    total_outliers = int(
        sum((v.get("n_outliers_iqr") or 0) for v in quality_metrics["variables"].values())
        )

    log_entry.update({
        "total_nulls": total_nulls,
        "total_outliers_iqr": total_outliers,
    })

    # métricas de calidad adicionales
    if quality_metrics.get("top_correlations"):
        # guarda las top 3 correlaciones como string
        top_corr = quality_metrics["top_correlations"][:3]
        corr_str = "; ".join([f"{c['var1']}-{c['var2']}: {c['correlation']:.3f}" for c in top_corr])
        log_entry["top_correlations"] = corr_str

    # estadísticas agregadas por variable
    all_means = [
        v.get("mean") for v in quality_metrics["variables"].values() if v.get("mean") is not None
        ]
    all_stds = [
        v.get("std") for v in quality_metrics["variables"].values() if v.get("std") is not None
        ]

    if all_means:
        log_entry["mean_of_means"] = float(np.mean(all_means))
    if all_stds:
        log_entry["mean_of_stds"] = float(np.mean(all_stds))

    # añade métricas del modelo (si existen)
    if model_metrics:
        # métricas de clustering
        if "n_clusters" in model_metrics:
            log_entry.update({
                "n_clusters": model_metrics["n_clusters"],
                "silhouette_score": model_metrics["silhouette_score"],
                "wssse": model_metrics["wssse"],
                "cluster_sizes": str(model_metrics.get("cluster_sizes", {})),
            })

            # métricas de balance y distribución
            if "cluster_percentages" in model_metrics:
                log_entry.update({
                    "cluster_percentages": str(model_metrics["cluster_percentages"]),
                    "min_cluster_size": model_metrics.get("min_cluster_size"),
                    "max_cluster_size": model_metrics.get("max_cluster_size"),
                    "mean_cluster_size": model_metrics.get("mean_cluster_size"),
                    "std_cluster_size": model_metrics.get("std_cluster_size"),
                    "execution_time_clustering": model_metrics.get("execution_time_clustering")
                })

        # métricas de PCA
        if "variance_explained" in model_metrics:
            log_entry.update({
                "variance_explained_total": model_metrics.get(
                    "total_variance_explained", sum(model_metrics["variance_explained"])
                    ),
                "variance_explained_components": str(model_metrics["variance_explained"]),
                "n_components_pca": model_metrics.get("n_components", 2),
                "execution_time_pca": model_metrics.get("execution_time_pca")
            })

        # métricas de t-SNE
        if "tsne_sample_size" in model_metrics:
            log_entry.update({
                "tsne_sample_size": model_metrics["tsne_sample_size"],
                "n_components_tsne": model_metrics.get("n_components_tsne", 2),
                "execution_time_tsne": model_metrics.get("execution_time_tsne")
            })

        # información de muestreo
        if "sampling_info" in model_metrics:
            sampling = model_metrics["sampling_info"]
            log_entry.update({
                "used_sampling": sampling.get("used_sampling", False),
                "original_size": sampling.get("original_size"),
                "sample_size": sampling.get("sample_size"),
                "sampling_fraction": sampling.get("sampling_fraction")
            })

    # añade rutas de artefactos
    log_entry.update({
        "artifact_features": artifact_paths.get("features", ""),
        "artifact_clusters": artifact_paths.get("clusters", ""),
        "artifact_pca": artifact_paths.get("pca", ""),
        "artifact_tsne": artifact_paths.get("tsne", ""),
    })

    # añade parámetros como JSON
    log_entry["parameters"] = json.dumps(metadata["parameters"])

    # convierte a DataFrame de Pandas
    log_df = pd.DataFrame([log_entry])

    # nombre de archivo, incluye fecha/hora actuales
    processing_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_group = group_name[:20].replace(" ", "_").lower()
    filename = f"{short_group}__silver_logs_{processing_ts}.parquet"

    # ruta del log
    filepath = f"{output_path}/{filename}"

    # añade al log existente, si se solicita
    if mode == 'append' and os.path.exists(filepath):
        # el log existe: lo carga
        existing_log = pd.read_parquet(filepath)
        # añade el log existente
        log_df = pd.concat([existing_log, log_df], ignore_index=True)

    # guardado
    log_df.to_parquet(filepath, index=False)

    print(f"✓ Métricas registradas en:\n{filepath}")
    print(f"  Total de registros en log: {len(log_df)}")
    
    
def clear_memory():
    """
    Limpia variables de memoria
    """
    show_memory_disk("Inicio")

    names_to_unpersist = ['ds_con_clusters', 'ds_pca', 'ds_tsne']
    names_to_delete = [
        'metadata', 'filepath_features', 'quality_metrics', 'results_grupo_pearson',
        'results_grupo_spearman', 'persist_info', 'scores', 'modelo_kmeans', 'cluster_stats',
        'clusters', 'model_metrics_kmeans', 'filepath_clusters', 'filepath_tsne', 'filepath_pca',
        'method_info', 'sample_size', 'dim_metrics', 'combined_metrics', 'artifact_paths'
    ]

    # aplica unpersist
    for name in names_to_unpersist:
        if name is globals():
            # la variable existe
            obj = globals()[name]
            if hasattr(obj, "unpersist"):
                # la variable tiene el método unpersist
                try:
                    obj.unpersist()
                except Exception:
                    # ignora el error
                    pass

    # borra variables
    for name in names_to_delete:
        if name in globals():
            # la variable existe
            try:
                del globals()[name]
            except Exception:
                # ignora el error
                pass

    gc.collect()

    show_memory_disk("Final")
