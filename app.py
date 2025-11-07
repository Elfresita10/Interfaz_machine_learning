import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

# --- Configuración de la página ---
st.set_page_config(
    page_title="Pronóstico de Ventas | Grupo Mimesa",
    page_icon="logo_techday.png",  # Icono que aparece en la pestaña del navegador
    layout="centered"
)

# --- Función para convertir imagen local a base64 ---
def image_to_base64(img):
    # Convertir la imagen a RGB (para eliminar el canal alfa)
    img_rgb = img.convert('RGB')
    
    buffered = BytesIO()
    img_rgb.save(buffered, format="PNG")  # Guardar como PNG
    return base64.b64encode(buffered.getvalue()).decode()

# --- Cargar imagen local (reemplaza la ruta con la tuya) ---
image_path = "fondo.png"  # Cambia por la ruta de tu imagen local
image = Image.open(image_path)

# Convertir la imagen a base64
img_base64 = image_to_base64(image)

# --- CSS para agregar imagen de fondo y el efecto glassmorphism ---
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;  /* Asegura que la imagen cubra todo el fondo */
            background-position: center center;  /* Centra la imagen */
            background-attachment: fixed;  /* La imagen no se moverá al hacer scroll */
            height: 100%;
        }}
        .stApp {{
            color: #FFFFFF;  /* Asegura que el texto sea blanco para contraste */
        }}

        /* Efecto glassmorphism */
        .stButton>button {{
            background: rgba(255, 255, 255, 0.1);  /* Fondo translúcido */
            backdrop-filter: blur(10px);  /* Desenfoque de fondo */
            border-radius: 15px;
            padding: 10px 20px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);  /* Sombra difusa */
        }}
        .stButton>button:hover {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.5);
            color: #FFB81C;
        }}

        /* Efecto glassmorphism para campos de entrada */
        .stSelectbox, .stNumberInput, .stDateInput, .stCheckbox, .stTextArea {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 10px;
            color: white;
        }}
        .stSelectbox:hover, .stNumberInput:hover, .stDateInput:hover, .stCheckbox:hover, .stTextArea:hover {{
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
    </style>
""", unsafe_allow_html=True)

# --- Ignorar Warnings ---
warnings.filterwarnings('ignore', category=UserWarning, message='.*X does not have valid feature names.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in subtract.*')

# --- Mostrar el logo y presentación del proyecto ---
st.image("logo_techday.png", width=250)  # Asegúrate de tener el logo en la misma ruta
st.title("Aplicación de Pronóstico de Ventas")
st.markdown("""
    Esta herramienta utiliza un modelo de **Machine Learning** (XGBoost) para predecir las unidades vendidas de un producto.
    \n\n
    **Grupo Mimesa** se dedica a mejorar la eficiencia de ventas y el pronóstico de demanda en empresas, ayudando a optimizar sus decisiones comerciales mediante la inteligencia artificial.
""")

# --- Función de Preprocesamiento (sin cambios) ---
def preparar_conjunto(df):
    df_copia = df.copy()
    df_copia['Date'] = pd.to_datetime(df_copia['Date'])
    df_copia['year'] = df_copia['Date'].dt.year
    df_copia['month'] = df_copia['Date'].dt.month
    df_copia['day_of_week'] = df_copia['Date'].dt.dayofweek
    df_copia['week_of_year'] = df_copia['Date'].dt.isocalendar().week.astype(int)
    columnas_categoricas = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    columnas_a_eliminar = ['Date', 'Store ID', 'Product ID']
    df_procesado = pd.get_dummies(df_copia, columns=columnas_categoricas, drop_first=True)
    df_procesado = df_procesado.drop(columns=columnas_a_eliminar)
    return df_procesado

# --- Cargar el modelo entrenado ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("modelo_final_ventas.json")
    return model

xgb_model = load_model()

# --- Definir las columnas esperadas por el modelo (sin cambios) ---
expected_columns = [
    'month', 'Demand Forecast', 'Discount', 'Price', 'Seasonality_Summer', 'day_of_week', 'Category_Toys', 'Category_Furniture', 'Category_Electronics', 'Weather Condition_Rainy', 'Seasonality_Spring', 'Weather Condition_Snowy', 'Inventory Level', 'Category_Groceries', 'Weather Condition_Sunny', 'year', 'Units Ordered', 'Region_North', 'Holiday/Promotion', 'Competitor Pricing', 'week_of_year', 'Seasonality_Winter', 'Region_South', 'Region_West'
]

# --- Interfaz de Usuario ---
st.header("Parámetros de Entrada")

# Controles de entrada para el usuario
col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Categoría del Producto", ["Groceries", "Toys", "Electronics", "Furniture", "Clothing"])
    region = st.selectbox("Región de la Tienda", ["North", "South", "West", "East"])
    weather_condition = st.selectbox("Condición Climática", ["Rainy", "Sunny", "Cloudy", "Snowy"])
    seasonality = st.selectbox("Estacionalidad", ["Autumn", "Summer", "Winter", "Spring"])
    prediction_date = st.date_input("Fecha de Pronóstico", pd.to_datetime("2025-01-01"))

with col2:
    inventory_level = st.number_input("Nivel de Inventario", min_value=0, value=250)
    units_ordered = st.number_input("Unidades Pedidas", min_value=0, value=100)
    demand_forecast = st.number_input("Pronóstico de Demanda (previo)", min_value=0.0, value=100.0, format="%.2f")
    price = st.number_input("Precio del Producto (€)", min_value=0.0, value=50.0, format="%.2f")
    competitor_pricing = st.number_input("Precio de la Competencia (€)", min_value=0.0, value=52.0, format="%.2f")
    discount = st.slider("Descuento Aplicado (%)", min_value=0, max_value=20, value=10)
    holiday_promotion = st.checkbox("¿Es día festivo o promoción?")

# Botón para realizar la predicción
if st.button("Calcular Pronóstico de Ventas"):
    input_data = pd.DataFrame([{
        'Date': prediction_date,
        'Store ID': 'S001', 'Product ID': 'P0001',  # IDs ficticios
        'Category': category, 'Region': region, 'Inventory Level': inventory_level,
        'Units Ordered': units_ordered, 'Demand Forecast': demand_forecast, 'Price': price,
        'Discount': discount, 'Weather Condition': weather_condition,
        'Holiday/Promotion': 1 if holiday_promotion else 0,
        'Competitor Pricing': competitor_pricing, 'Seasonality': seasonality
    }])

    processed_input_data = preparar_conjunto(input_data)
    final_input_data = processed_input_data.reindex(columns=expected_columns, fill_value=0)

    try:
        prediction = xgb_model.predict(final_input_data)[0]
        st.success(f"**Pronóstico de unidades a vender: {int(round(prediction))}**")
        st.balloons()
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.warning("Verifica que el archivo 'modelo_final_ventas.json' esté en la misma carpeta.")

# --- Pie de página ---
st.markdown("---")
st.markdown("Desarrollado por INTEZIA para el IA TECHDAY 2025 | www.intezia.com | www.iatechday.com")
