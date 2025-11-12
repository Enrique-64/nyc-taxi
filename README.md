# 🚕 NYC Taxi Data Analysis - Medallion Architecture

Estudio exhaustivo del dataset NYC Taxi Trip Records mediante arquitectura Medallion (Bronze → Silver → preparación para Gold), enfocado en análisis de correlaciones y preparación de datos para modelado.

## 📊 Descripción del Proyecto

Proyecto de análisis de datos profesional que procesa más de 9 millones de registros mensuales de viajes en taxi de Nueva York. Implementa pipeline completo de ingesta, validación, transformación y análisis exploratorio siguiendo principios de Data Lakehouse.

**Alcance actual**: Bronze Layer → Silver Layer → preparación para Gold Layer (sin implementación de objetivos de negocio)

## 🎯 Objetivos

- Automatizar ingesta y validación de datos desde fuente oficial NYC TLC
- Implementar arquitectura Medallion con capas Bronze y Silver
- Realizar análisis exhaustivo de correlaciones entre variables
- Identificar y documentar grupos de variables relacionadas
- Preparar datasets limpios para futura capa Gold

## 🏗️ Arquitectura de Datos

```
├── raw/                          # Datos originales descargados
│   ├── yellow_tripdata_2023-01.parquet
│   ├── yellow_tripdata_2023-02.parquet
│   └── yellow_tripdata_2023-03.parquet
│
├── bronze/                       # Capa Bronze - Datos validados
│   └── taxi_data/
│       └── ingestion_year=2023/
│           ├── ingestion_month=01/
│           ├── ingestion_month=02/
│           └── ingestion_month=03/
│
├── metadata/                     # Logs y metadatos de ingesta
│   ├── ingestion_log.jsonl
│   └── bronze_layer_metadata.json
│
└── silver/                       # Capa Silver - Datos transformados
    ├── dataset_202301_filtered_outliers/
    ├── dataset_202301_for_correlation/
    ├── models_202301_for_correlation/
    ├── dataset_202301_for_correlation_clean/
    └── for_gold/                 # Preparación para Gold Layer
```

## 📚 Notebooks (Orden de Ejecución)

### 1. Configuración y Exploración
**`1_v1_Colab_Setup_Exploracion.ipynb`**
- Configuración inicial del entorno Google Colab
- Exploración preliminar del dataset

### 2. Capa Bronze - Ingesta
**`2_v2_Ingesta_Bronze.ipynb`**
- Automatización de descarga desde fuente oficial NYC TLC
- Estandarización del proceso de ingesta
- Validación inicial robusta
- Generación de metadatos y logs

### 3. Bronze → Silver - Exploración
**`2023-01_3_v4_Bronze_to_Silver.ipynb`**
- Estudio detallado de variables (dataset Enero 2023)
- Visualizaciones exploratorias
- Análisis de distribuciones

### 4. Capa Silver (Paso 1, Fase 1) - Limpieza
**`2023-01_4_1_F1_v5_Silver_Preparar_Datos.ipynb`**
- Depuración de nulos, duplicados y valores inconsistentes
- Homogeneización de tipos, unidades y formatos
- Tratamiento de valores atípicos

### 5. Capa Silver (Paso 1, Fase 2) - Feature Engineering
**`2023-01_4_1_F2_v5_Silver_Preparar_Datos.ipynb`**
- Creación de variables derivadas
- Preparación para análisis de correlación

### 6. Capa Silver (Paso 2, Fase 1) - Análisis Inicial
**`2023-01_4_2_F1_v6_Silver_to_Gold.ipynb`**
- Análisis de correlaciones base

### 7. Capa Silver (Paso 2, Fase 2) - Grupos A y B
**`2023-01_4_2_F2_v9_Silver_to_Gold.ipynb`**
- **Grupo A**: Variables con relación matemática directa
- **Grupo B**: Redundancia temporal
- Visualizaciones de clusters
- Generación de archivos JSON y Parquet por subgrupo

### 8. Capa Silver (Paso 2, Fase 3) - Grupos C (1-5)
**`2023-01_4_2_F3_v9_Silver_to_Gold.ipynb`**
- **Grupo C-1**: Ubicación y tarifas
- **Grupo C-2**: Propinas
- **Grupo C-3**: Variables de servicio
- **Grupo C-4**: Variables técnicas/operativas
- **Grupo C-5**: Correlaciones cruzadas contextuales (ubicación-tarifas)
- Visualizaciones de clusters
- Generación de archivos JSON y Parquet por subgrupo

### 9. Capa Silver (Paso 2, Fase 4) - Grupos C (6-7)
**`2023-01_4_2_F4_v9_Silver_to_Gold.ipynb`**
- **Grupo C-6**: Correlaciones temporales-servicio
- **Grupo C-7**: Correlaciones costo-distancia/duración
- Visualizaciones de clusters
- Generación de archivos JSON y Parquet por subgrupo

### 10. Módulo de Funciones
**`silver_functions_v3.py`**
- Funciones compartidas para procesamiento Silver
- Utilidades de análisis y visualización

## 🔍 Grupos de Variables Analizadas

### Grupo A: Relación Matemática Directa
#### A-1: Variables de Movimiento
- `trip_distance_encoded`, `trip_duration_minutes_encoded`, `average_speed_mph_encoded`

#### A-2: Variables Tarifarias
- `total_amount_encoded`, `fare_per_mile_encoded`, `tip_amount_encoded`, `extra_encoded`, `mta_tax_encoded`, `improvement_surcharge_encoded`

### Grupo B: Redundancia Temporal
#### B-1: Día de Semana (Inicio)
- `tpep_pickup_datetime_dayofweek`, componentes sin/cos, `is_weekend`

#### B-2: Hora (Inicio)
- `pickup_hour`, componentes sin/cos

#### B-3: Día de Semana (Final)
- `tpep_dropoff_datetime_dayofweek`, componentes sin/cos

#### B-4: Correlación Cruzada
- Variables pickup vs dropoff equivalentes

### Grupo C: Correlaciones Contextuales
#### C-1: Ubicación y Tarifas
- `PULocationID_encoded`, `DOLocationID_encoded`, `RatecodeID_encoded`

#### C-2: Propinas
- `tip_amount_encoded`, `tip_score_encoded`, `payment_type`

#### C-3: Servicio
- `passenger_count_encoded`, `trip_distance_encoded`, `trip_duration_minutes_encoded`

#### C-4: Técnicas/Operativas
- `store_and_fwd_flag_encoded`, `VendorID_encoded`

#### C-5: Cruzadas Ubicación-Tarifas
- Correlaciones entre LocationID, RatecodeID, fare_per_mile y extra

#### C-6: Cruzadas Temporales-Servicio
- Correlaciones entre pickup_hour, is_weekend, dayofweek y passenger_count, trip_distance, average_speed, toll_indicator

#### C-7: Cruzadas Costo-Distancia/Duración
- Correlaciones entre total_amount, tip_amount, fare_per_mile, trip_extra_cost_ratio y trip_distance, trip_duration, average_speed

## 🛠️ Tecnologías

- **Python**: 3.12.12 (Google Colab)
- **PySpark**: 3.5.1 (procesamiento distribuido)
- **Pandas**: 2.2.2
- **NumPy**: 2.0.2
- **Scikit-learn**: 1.6.1 (análisis y clustering)
- **Visualización**: Matplotlib 3.10.0, Seaborn 0.13.2, Plotly 5.24.1
- **Análisis estadístico**: SciPy 1.16.3
- **Utilidades**: requests 2.32.4, psutil 5.9.5

## 📦 Instalación

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
pyspark==3.5.1
pandas==2.2.2
numpy==2.0.2
seaborn==0.13.2
matplotlib==3.10.0
plotly==5.24.1
scipy==1.16.3
scikit-learn==1.6.1
requests==2.32.4
psutil==5.9.5
```

## 🚀 Uso

1. **Clonar repositorio**
```bash
git clone [URL_REPOSITORIO]
cd nyc-taxi-analysis
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecutar notebooks en orden secuencial** (1 → 2 → 3 → ... → 10)
   - Todos los notebooks son obligatorios
   - Deben ejecutarse en el orden indicado
   - Recomendado: Google Colab

## 📁 Fuente de Datos

**NYC Taxi & Limousine Commission (TLC)**
- URL: `https://d37ci6vzurychx.cloudfront.net/trip-data/`
- Formato: Parquet
- Volumen: ~9 millones de registros/mes
- Período analizado: Enero-Marzo 2023

## 📈 Resultados

- Pipeline automatizado de ingesta y validación
- Datasets limpios y normalizados en capa Silver
- Análisis exhaustivo de correlaciones entre 40+ variables
- Identificación de 7 grupos principales de variables relacionadas
- Visualizaciones de clusters por grupo
- Datasets preparados para modelado (capa Gold)

## 🔄 Próximos Pasos

- [ ] Implementar capa Gold con objetivos de negocio
- [ ] Desarrollar modelos predictivos
- [ ] Ampliar análisis a más meses
- [ ] Dashboard interactivo de visualizaciones

## 👤 Autor

Enrique
- GitHub: [@Enrique-64](https://github.com/Enrique-64)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

*Desarrollado con arquitectura Medallion siguiendo mejores prácticas de Data Lakehouse*