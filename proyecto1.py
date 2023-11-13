import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency

st.title('Proyecto 1')
st.text("Herramienta de exploración y análisis de datos.")
st.text("Grupo 7: Marvin Chávez 08105031, Maycol Córdova 22007865 y David Rivera 22000785.")

#Tipo de columna
def classify_columns(df):
    discrete, continuous, categorical, date = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < 30:
                discrete.append(col)
            else:
                continuous.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
        
            date.append(col)
        else:
          
            categorical.append(col)
    return discrete, continuous, categorical, date

# Botón para cambiar el archivo cargado
if 'df' in st.session_state and st.sidebar.button('Cambiar archivo'):
    del st.session_state['df']
    st.session_state['columns_classified'] = False
    st.experimental_rerun()

# Cargar archivo y almacenar 
if 'df' not in st.session_state or 'columns_classified' not in st.session_state:
    st.session_state['columns_classified'] = False

if not st.session_state['columns_classified']:
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV o XLSX", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            st.session_state['df'] = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state['df'] = pd.read_excel(uploaded_file)
        st.session_state['columns_classified'] = True
        st.sidebar.success('Archivo cargado y clasificado exitosamente!')

if 'df' in st.session_state and st.session_state['columns_classified']:
    dataset = st.session_state['df']  
    discrete, continuous, categorical, date = classify_columns(dataset)

# Graficas Individuales (una variable) : 

# Gráfica de densidad variable continua
def mostrar_grafica_continua(variable, df):
    fig, ax = plt.subplots()
    sns.histplot(df[variable], kde=True, ax=ax)
    mean_val = df[variable].mean()
    median_val = df[variable].median()
    std_val = df[variable].std()
    var_val = df[variable].var()

    ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='-', label=f'Mediana: {median_val:.2f}')

    # std y var
    #ax.text(0.95, 0.95, f'Desviación Estándar: {std_val:.2f}\nVarianza: {var_val:.2f}', 
    #        transform=ax.transAxes, horizontalalignment='right', 
    #        verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    st.pyplot(fig)
    st.write(f"Desviación Estándar de {variable}: {std_val:.2f}")
    st.write(f"Varianza de {variable}: {var_val:.2f}")

# Gráfica de histograma variable discreta
def mostrar_grafica_discreta(variable, df):
    fig, ax = plt.subplots()
    sns.histplot(df[variable], kde=False, discrete=True, ax=ax)
    mean_val = df[variable].mean()
    median_val = df[variable].median()
    std_val = df[variable].std()
    var_val = df[variable].var()
    mode_val = df[variable].mode().values[0] 

    ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='-', label=f'Mediana: {median_val:.2f}')
    ax.axvline(mode_val, color='blue', linestyle='-.', label=f'Moda: {mode_val}')

    # Añadir anotaciones para std y var
    #ax.text(0.95, 0.85, f'Desviación Estándar: {std_val:.2f}\nVarianza: {var_val:.2f}\nModa: {mode_val}', 
    #        transform=ax.transAxes, horizontalalignment='right', 
    #        verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    st.pyplot(fig)
    st.write(f"Estadísticas de {variable}:")
    st.write(f"Media: {mean_val:.2f},  Desviación Estándar: {std_val:.2f}, Varianza: {var_val:.2f}")

# Gráficas de barras variable categórica
def mostrar_grafica_categorica(variable, df):
    fig, ax = plt.subplots()
    sns.countplot(y=variable, data=df, ax=ax)
    
    # Añadir anotaciones para conteo total por categoría
    for p in ax.patches:
        width = p.get_width()
        plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),
                 f'{int(width)}',
                 ha='center', va='center')

    st.pyplot(fig)

#  Graficas Combinadas (dos variables)

def mostrar_scatter_plot(variable_x, variable_y, df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=variable_x, y=variable_y, ax=ax)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    st.pyplot(fig)
    
    # Cálculo y muestra de la correlación
    correlacion = df[variable_x].corr(df[variable_y])
    st.write(f"Correlación entre {variable_x} y {variable_y}: {correlacion:.2f}")

def mostrar_serie_tiempo(variable_numerica, variable_temporal, df):
    fig, ax = plt.subplots()
    df = df.sort_values(by=variable_temporal) 
    sns.lineplot(data=df, x=variable_temporal, y=variable_numerica, ax=ax)
    plt.xlabel(variable_temporal)
    plt.ylabel(variable_numerica)
    st.pyplot(fig)

def mostrar_boxplot(variable_continua, variable_categorica, df):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=variable_categorica, y=variable_continua, ax=ax)
    plt.xlabel(variable_categorica)
    plt.ylabel(variable_continua)
    st.pyplot(fig)

# Gráfico de mosaico
def mostrar_mosaico_y_cramer(variable_categorica1, variable_categorica2, df):
    if variable_categorica1 != variable_categorica2: # Varibles diferentes
        # Temporal para esta grafica
        contingency_table = pd.crosstab(df[variable_categorica1], df[variable_categorica2])
        
        
        plt.figure(figsize=(10, 5))
        mosaic(df, [variable_categorica1, variable_categorica2])
        plt.show()
        st.pyplot(plt)

        # Cálculo del coeficiente de contingencia de Cramer
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = sum(contingency_table.sum())
        cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        st.write(f"Coeficiente de contingencia de Cramer para {variable_categorica1} y {variable_categorica2}: {cramer_v:.2f}")
    else:
        st.error("Selecciona dos categorías diferentes para el gráfico de mosaico.")


# Menú de selección de operaciones
menu = st.sidebar.radio("Selecciona una operación", ('1. Despligue Datos', '2. Clasificación de Columnas', '3. Graficas Individuales','4. Graficas Combinadas'))

# 1. Despligue Datos
if menu == '1. Despligue Datos' and st.session_state['columns_classified']:
    st.subheader("Menú 1: Visualizar Datos")
    st.dataframe(st.session_state['df'])


# 2. Clasificación de Columnas
elif menu == '2. Clasificación de Columnas' and st.session_state['columns_classified']:
    st.subheader("Menú 2: Clasificación de Columnas")
    if 'df' in st.session_state:
        discrete, continuous, categorical, date = classify_columns(st.session_state['df'])
        st.write("Columnas Discretas:", discrete)
        st.write("Columnas Continuas:", continuous)
        st.write("Columnas Categóricas:", categorical)
        st.write("Columnas de Tipo Fecha:", date)
# 3. Graficas Individuales
if menu == '3. Graficas Individuales':
    st.subheader("Menú 3: Gráficas Individuales")

    tipo_variable = st.radio("Selecciona el tipo de variable para graficar", ('Continua', 'Discreta', 'Categórica'))
    
    if tipo_variable == 'Continua':
        variable_continua = st.selectbox("Selecciona una Variable Continua", continuous)
        mostrar_grafica_continua(variable_continua, dataset)

    elif tipo_variable == 'Discreta':
        variable_discreta = st.selectbox("Selecciona una Variable Discreta", discrete)
        mostrar_grafica_discreta(variable_discreta, dataset)

    elif tipo_variable == 'Categórica':
        variable_categorica = st.selectbox("Selecciona una Variable Categórica", categorical)
        mostrar_grafica_categorica(variable_categorica, dataset)
# 4. Graficas Combinadas
if menu == '4. Graficas Combinadas':
    st.subheader("Menú 4: Gráficas Combinadas (dos variables)")

    tipo_grafica = st.radio("Selecciona el tipo de gráfica a realizar", 
                            ('Numérica vrs Numérica', 'Numérica vrs Temporal', 'Continua vrs Categórica', 'Categórica vrs Categórica'))

    if tipo_grafica == 'Numérica vrs Numérica':
        variable_x = st.selectbox("Selecciona la Variable para eje X (Numérica)", continuous + discrete)
        variable_y = st.selectbox("Selecciona la Variable para eje Y (Numérica)", continuous + discrete)
        if variable_x and variable_y:
            mostrar_scatter_plot(variable_x, variable_y, st.session_state['df'])

    elif tipo_grafica == 'Numérica vrs Temporal':
        variable_numerica = st.selectbox("Selecciona la Variable Numérica", continuous + discrete)
        variable_temporal = st.selectbox("Selecciona la Variable Temporal", date)
        if variable_numerica and variable_temporal:
            mostrar_serie_tiempo(variable_numerica, variable_temporal, st.session_state['df'])

    elif tipo_grafica == 'Continua vrs Categórica':
        variable_continua = st.selectbox("Selecciona la Variable Continua", continuous)
        variable_categorica = st.selectbox("Selecciona la Variable Categórica", categorical)
        if variable_continua and variable_categorica:
            mostrar_boxplot(variable_continua, variable_categorica, st.session_state['df'])

    elif tipo_grafica == 'Categórica vrs Categórica':
        variable_categorica1 = st.selectbox("Selecciona la primera Variable Categórica", categorical)
        variable_categorica2 = st.selectbox("Selecciona la segunda Variable Categórica", categorical)
        if variable_categorica1 and variable_categorica2:
            mostrar_mosaico_y_cramer(variable_categorica1, variable_categorica2, st.session_state['df'])