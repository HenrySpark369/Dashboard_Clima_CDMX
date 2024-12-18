from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
import gzip
import json
import threading
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta
import os


# URL y cabeceras para la solicitud
url = "https://smn.conagua.gob.mx/tools/GUI/webservices/"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate"
}
params = {"method": 3}

# Variable global para compartir los datos
global_df = pd.DataFrame()

def descargar_y_guardar_comprimido(url, headers, params, file_path="HourlyForecast_MX.gz"):
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            with open(file_path, "wb") as gz_file:
                gz_file.write(response.content)
            print("Datos descargados y guardados en", file_path)
        else:
            print(f"Error en la solicitud. Código: {response.status_code}")
    except Exception as e:
        print(f"Error en la descarga: {e}")

def leer_comprimido_a_dataframe(file_path="HourlyForecast_MX.gz"):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
            data = json.load(gz_file)
        df = pd.DataFrame(data)
        categorical_cols = ['desciel', 'nes', 'nmun', 'dirvienc', 'dsem']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        df['hloc'] = pd.to_datetime(df['hloc'], format='%Y%m%dT%H') + pd.Timedelta(hours=0)
        df.set_index('hloc', inplace=True)
        cols_to_float = ['temp', 'prec', 'velvien', 'dirvieng', 'dpt', 'hr', 'raf', 'lat', 'lon']
        for col in cols_to_float:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df['nes'] == 'Ciudad de México'].copy()
        return df
    except Exception as e:
        print(f"Error al leer y procesar el archivo: {e}")
        return pd.DataFrame()

def scrapeo_periodico():
    global global_df
    while True:
        try:
            descargar_y_guardar_comprimido(url, headers, params)
            global_df = leer_comprimido_a_dataframe()
            print("Datos actualizados.")
        except Exception as e:
            print(f"Error en el scrapeo periódico: {e}")
        time.sleep(3600)

# Inicializar el hilo de scrapeo
scrapeo_thread = threading.Thread(target=scrapeo_periodico, daemon=True)
scrapeo_thread.start()

# Lista de contaminantes, sus unidades y colores
contaminantes = {
    'CO': ('ppm', '#05b1f7', 9),
    'NO': ('ppb', '#018a3d', None),
    'NO2': ('ppb', '#466e02', 106),
    'NOX': ('ppb', '#0262b3', None),
    'O3': ('ppb', '#fa4902', 60),
    'PM10': ('µg/m³', '#de7e02', 70),
    'PM25': ('µg/m³', '#873901', 41),
    'SO2': ('ppb', '#FFD700', 40)
}

# Cargar datos del archivo CSV
csv = os.path.join(os.getcwd(), 'rama_2023_05.csv')
Rama = pd.read_csv(csv)
Rama["fecha"] = pd.to_datetime(Rama["fecha"])
Rama.set_index('fecha', inplace=True)

# Estilos globales para texto
ESTILO_TEXTO = {
    "color": "#465973",  # Color estandarizado para el texto
    "font-size": "1.5rem",
    "font-weight": "bold",
    "font-family": "Arial, sans-serif"
}

ESTILO_TEXTO_TITULO = {
    "color": "#898d8d",
    "font-weight": "bold",
    "font-size": "2rem",
    "textAlign": "center"
}

# Inicializar la aplicación Dash con el tema Sketchy
app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
server = app.server  # Exponer el servidor Flask


# Layout de la aplicación
app.layout = dbc.Container([
    # Título principal
    html.H1(
        "Contaminantes a lo largo del tiempo",
        className="text-center mb-4",
        style=ESTILO_TEXTO_TITULO
    ),

    # Dropdown para seleccionar columnas
    html.Div(
        [
            html.Label(
                'Selecciona las columnas para graficar:',
                className="mb-2",
                style=ESTILO_TEXTO
            ),
            dcc.Dropdown(
                id='dropdown-columnas',
                options=[{'label': col, 'value': col} for col in Rama.columns],
                multi=True,
                placeholder="Selecciona parámetros",
                style=ESTILO_TEXTO
            ),
        ],
        className="mb-4"
    ),

    # Sliders de selección
    html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.Label("Año:", style=ESTILO_TEXTO),
                    dcc.RangeSlider(
                        id='range-slider-años',
                        min=2015, max=2023, step=1,
                        marks={i: str(i) for i in range(2015, 2024)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=[2015, 2023]
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Meses (Rango):", style=ESTILO_TEXTO),
                    dcc.RangeSlider(
                        id='range-slider-meses',
                        min=1, max=12, step=1,
                        marks={
                            i: mes for i, mes in enumerate(
                                ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], start=1
                            )
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=[1, 12]
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Días (Rango):", style=ESTILO_TEXTO),
                    dcc.RangeSlider(
                        id='range-slider-dias',
                        min=1, max=31, step=1,
                        marks={i: str(i) for i in range(1, 32)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=[1, 31]
                    )
                ], width=4),
            ])
        ],
        className="mb-4"
    ),

    # Gráficos
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(id='graficos-dinamicos')
                ),
                style={"border": "2px dashed #235B4E"}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(id='graficos-distribuciones')
                ),
                style={"border": "2px dashed #235B4E"}
            ),
            width=6
        ),
    ]),
    
    html.Br(),
    html.H1("Predicciones del Clima en México", className="text-center mb-4",
    style=ESTILO_TEXTO_TITULO),
    dcc.Interval(id="intervalo-actualizacion", interval=60*1000),  # Intervalo de 1 minuto
    html.Div([
        html.Div(dcc.Graph(id="mapa-clima"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(dcc.Graph(id="predicciones-reales"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(id="metricas-modelo", style={"width": "33%", "display": "inline-block", "padding": "0 10px"})
    ]),
    html.Div([
        html.Div(dcc.Graph(id="grafico-temp"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(dcc.Graph(id="grafico-hr"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(dcc.Graph(id="grafico-prec"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-between"})
    
], fluid=True, style={"backgroundColor": "#f8f9fa"})



# Callback para actualizar el gráfico
@app.callback(
    [Output('graficos-dinamicos', 'children'),
     Output('graficos-distribuciones', 'children')],
    [Input('dropdown-columnas', 'value'),
     Input('range-slider-años', 'value'),
     Input('range-slider-meses', 'value'),
     Input('range-slider-dias', 'value')]
)
def actualizar_graficos(columnas_seleccionadas, rango_años, rango_meses, rango_dias):
    # Si no hay columnas seleccionadas, muestra un mensaje
    if not columnas_seleccionadas:
        mensaje = dbc.Card(
            [
                dbc.CardHeader("Atención", className="text-white bg-secondary"),
                dbc.CardBody([
                    html.H4("Sin variables seleccionadas", className="card-title"),
                    html.P(
                        "Selecciona columnas para visualizar los gráficos.",
                        className="card-text"
                    )
                ])
            ],
            style={"max-width": "20rem"},
            className="mb-3 text-white bg-secondary"
        )
        return mensaje, mensaje

    # Filtrar los datos por año, mes y rango de días seleccionados
    datos_filtrados = Rama.copy()
    datos_filtrados['formatted_date'] = datos_filtrados.index.strftime('%Y-%m-%d')

    if rango_años:
        año_inicio, año_fin = rango_años
        datos_filtrados = datos_filtrados[(datos_filtrados.index.year >= año_inicio) &
                                          (datos_filtrados.index.year <= año_fin)]
    if rango_meses:
        mes_inicio, mes_fin = rango_meses
        datos_filtrados = datos_filtrados[(datos_filtrados.index.month >= mes_inicio) &
                                          (datos_filtrados.index.month <= mes_fin)]
    if rango_dias:
        dia_inicio, dia_fin = rango_dias
        datos_filtrados = datos_filtrados[(datos_filtrados.index.day >= dia_inicio) &
                                          (datos_filtrados.index.day <= dia_fin)]

    # === Gráfico Principal ===
    # === Gráfico Principal Simplificado con limite_diario ===
   # === Gráfico Principal con Suavización (Media Móvil) ===
    fig_principal = go.Figure()
    
    # Ventana para la media móvil
    window_size = 1  # Número de días para suavizar
    
    for columna in columnas_seleccionadas:
        if columna not in contaminantes:
            continue
    
        unidad, color, limite_diario = contaminantes[columna]
        
        # Aplicar media móvil
        datos_suavizados = datos_filtrados[columna].rolling(window=window_size, min_periods=1).max()
    
        # Agregar la traza de los datos suavizados
        fig_principal.add_trace(go.Scatter(
            x=datos_filtrados['formatted_date'],
            y=datos_suavizados,
            mode='lines',
            name=f"{columna} Suavizado ({unidad})",
            line=dict(color=color, width=2)
        ))
    
        # Línea horizontal para el límite diario
        if limite_diario is not None:
            fig_principal.add_hline(
                y=limite_diario,
                line_dash="dot",
                line_color=color,
                annotation_text=f"Límite: {limite_diario} {unidad}",
                annotation_position="top right",
                annotation_font_size=12,
                annotation_font_color=color
            )
    
    # Configuración básica del layout
    fig_principal.update_layout(
        title="Gráfico de Contaminantes Seleccionados (Suavizado)",
        xaxis_title="Fecha",
        yaxis_title="Concentración",
        height=500,
        legend_title="Parámetros Seleccionados",
        paper_bgcolor="#898d8d",
        font=dict(color="white")
    )
        
    # === Gráfico de Histogramas ===
    fig_histogramas = go.Figure()
    for columna in columnas_seleccionadas:
        if columna not in contaminantes:
            continue
        
        _, color, _ = contaminantes[columna]
        fig_histogramas.add_trace(go.Histogram(
            x=datos_filtrados[columna],
            name=f"{columna} ({unidad})",
            nbinsx=100,
            marker=dict(color=color),
            opacity=0.7
        ))
    fig_histogramas.update_layout(
        xaxis_title="Concentración",
        yaxis_title="Frecuencia",
        height=500,
        barmode='overlay',
        legend_title="Parámetros",
        paper_bgcolor="#898d8d",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Devolver ambos gráficos en tarjetas estilo Sketchy
    return (
        dbc.Card(
            [
                dbc.CardHeader("Gráfico Superpuesto de Contaminantes", className="text-white bg-secondary"),
                dbc.CardBody(
                    dcc.Graph(figure=fig_principal)
                )
            ],
            style={"max-width": "100%"},
            className="mb-4 text-white bg-secondary"
        ),
        dbc.Card(
            [
                dbc.CardHeader("Histogramas de Contaminantes Seleccionados", className="text-white bg-secondary"),
                dbc.CardBody(
                    dcc.Graph(figure=fig_histogramas)
                )
            ],
            style={"max-width": "100%"},
            className="mb-4 text-white bg-secondary"
        )
    )

@app.callback(
    [Output("mapa-clima", "figure"),
     Output("grafico-temp", "figure"),
     Output("grafico-hr", "figure"),
     Output("grafico-prec", "figure"),
     Output("metricas-modelo", "children"),
     Output("predicciones-reales", "figure")],
    [Input("intervalo-actualizacion", "n_intervals")]
)

def actualizar_dashboard(n_intervals):
    global global_df
    if not global_df.empty:
        df_dashboard = global_df[['dpt', 'hr', 'temp', 'raf', 'prec', 'lat', 'lon', 'desciel']].reset_index()
        
        # Filtrar las próximas 24 horas
        now = datetime.now()
        next_24_hours = now + timedelta(hours=24)
        df_temp_24h = df_dashboard[(df_dashboard['hloc'] >= now) & (df_dashboard['hloc'] <= next_24_hours)]
        #df_temp_24h = df_temp_24h.groupby("hloc", as_index=False).mean(numeric_only=True)  # Eliminar duplicados
        df_temp_24h = df_temp_24h.set_index("hloc").resample("1h").mean(numeric_only=True).reset_index()  # Asegurar intervalos regulares
        
        # Mapa interactivo
        mapa_clima = go.Figure()
        mapa_clima.add_trace(go.Scattermapbox(
            lat=df_dashboard["lat"],
            lon=df_dashboard["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=12,
                color=df_dashboard["temp"],
                colorscale="thermal",
                showscale=True,
                sizemode="area",
            ),
            text=df_dashboard["desciel"],
            hoverinfo="lon+lat+text",  # Información a mostrar: latitud, longitud y texto
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Latitud: %{lat}<br>" +
                "Longitud: %{lon}<br>" +
                "Temperatura: %{marker.color} °C<br>" +
                "Precipitación: %{customdata[0]} mm<br>" +
                "Hora: %{customdata[1]}"  # Mostrar la hora desde customdata
            ),
            customdata=list(zip(df_dashboard["prec"], df_dashboard["hloc"].astype(str))),  # Crear pares de valores directamente
            name="Estado <br> Temperatura (°C)<br> y Precipitación (mm)"
        ))
        
        mapa_clima.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=10,
                center={"lat": 19.4326, "lon": -99.1332}
            ),
            title="Mapa Interactivo de Condiciones Climáticas"
        )        
        # Gráfico de tendencias temporales mejorado
        # Gráfico de temperatura con suavizado
        # Crear gráfico para las próximas 24 horas
        grafico_temp_24h = go.Figure()
        grafico_temp_24h.add_trace(go.Scatter(
            x=df_temp_24h["hloc"],
            y=df_temp_24h["temp"],
            mode="lines+markers",  # Agregar puntos y líneas para mayor claridad
            name="Temperatura (Próximas 24h)",
            line=dict(color='blue', width=2),
            marker=dict(size=6),  # Tamaño de los puntos
            hovertemplate="Hora: %{x}<br>Temperatura: %{y} °C"  # Información detallada en hover
        ))
        
        grafico_temp_24h.update_layout(
            title="Tendencia de Temperatura (Próximas 24 horas)",
            xaxis=dict(title="Hora Local", showgrid=True),
            yaxis=dict(title="Temperatura (°C)", showgrid=True),
            template="plotly_white"
        )
        
        # Gráfico de humedad relativa
        grafico_hr = go.Figure()
        grafico_hr.add_trace(go.Scatter(
            x=df_temp_24h["hloc"],
            y=df_temp_24h["hr"],
            mode="lines+markers",
            name="Humedad Relativa (Próximas 24h)",
            line=dict(color="red", width=2),
        ))
        grafico_hr.update_layout(
            title="Tendencia de Humedad Relativa (Próximas 24 horas)",
            xaxis=dict(title="Hora Local", showgrid=True),
            yaxis=dict(title="Humedad Relativa (%)", showgrid=True)
        )
        
        # Gráfico de precipitación con barras
        grafico_prec = go.Figure()
        grafico_prec.add_trace(go.Bar(
            x=df_temp_24h["hloc"],
            y=df_temp_24h["prec"],
            name="Precipitación (Próximas 24h)",
            marker=dict(color="green"),
        ))
        grafico_prec.update_layout(
            title="Tendencia de Precipitación (Próximas 24 horas)",
            xaxis=dict(title="Hora Local", showgrid=True),
            yaxis=dict(title="Precipitación (mm)", showgrid=True)
        )

        # Modelo predictivo
        scaler = StandardScaler()
        df_scaled = df_dashboard[['dpt', 'hr', 'prec', 'temp']].dropna()
        df_scaled[['dpt', 'hr', 'prec', 'temp']] = scaler.fit_transform(df_scaled)

        X = df_scaled[['dpt', 'hr', 'prec']]
        y = df_scaled['temp']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        # Métricas del modelo
        metricas_modelo = {
            "R²": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        metricas_texto = [
            html.P(f"R²: {metricas_modelo['R²']:.2f}"),
            html.P(f"RMSE: {metricas_modelo['RMSE']:.2f}"),
            html.P(f"MAE: {metricas_modelo['MAE']:.2f}"),
        ]
        
                # Invertir normalización para y_test
        y_test_inverse = (y_test * scaler.scale_[-1]) + scaler.mean_[-1]
        
        # Invertir normalización para y_pred
        y_pred_inverse = (y_pred * scaler.scale_[-1]) + scaler.mean_[-1]


        # Gráfico de predicciones vs reales
        # Gráfico de predicciones vs reales con valores originales
        pred_vs_real = go.Figure()
        
        # Graficar los puntos reales vs predichos
        pred_vs_real.add_trace(go.Scatter(
            x=y_test_inverse,  # Valores reales desnormalizados
            y=y_pred_inverse,  # Predicciones desnormalizadas
            mode='markers',
            name='Predicciones',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Línea de igualdad
        pred_vs_real.add_trace(go.Scatter(
            x=[y_test_inverse.min(), y_test_inverse.max()],
            y=[y_test_inverse.min(), y_test_inverse.max()],
            mode='lines',
            name='Igualdad',
            line=dict(color='red', dash='dash')
        ))
        
        # Layout del gráfico
        pred_vs_real.update_layout(
            title="Comparación de Predicciones vs Valores Reales",
            xaxis_title="Valores Reales (Temperatura)",
            yaxis_title="Valores Predichos (Temperatura)",
            showlegend=True
        )

        return mapa_clima, grafico_temp_24h, grafico_hr, grafico_prec, metricas_texto, pred_vs_real
    else:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div("Cargando datos..."), go.Figure()


# Ejecutar el servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Usar el puerto asignado por Render
    app.run_server(host="0.0.0.0", port=port, debug=True)
