#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 01:54:39 2024

@author: sparkmachine
"""

# Importar librerías necesarias
import requests
import gzip
import json
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import threading
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta


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

# Configurar la aplicación Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard del Clima en México"),
    html.Div("Tendencias de Variables Clave y Modelo Predictivo"),
    dcc.Interval(id="intervalo-actualizacion", interval=60*1000),  # Intervalo de 1 minuto
    html.Div([
        dcc.Graph(id="mapa-clima"),
    ]),
    html.Div([
        html.Div(dcc.Graph(id="grafico-temp"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(dcc.Graph(id="grafico-hr"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
        html.Div(dcc.Graph(id="grafico-prec"), style={"width": "33%", "display": "inline-block", "padding": "0 10px"}),
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-between"}),
    html.Div(id="metricas-modelo"),
    html.Div([
        dcc.Graph(id="predicciones-reales"),
    ]),
])


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

        # Métricas del modelo
        metricas_modelo = {
            "R²": r2_score(y_test, model.predict(X_test)),
            "RMSE": np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
            "MAE": mean_absolute_error(y_test, model.predict(X_test))
        }
        metricas_texto = [
            html.P(f"R²: {metricas_modelo['R²']:.2f}"),
            html.P(f"RMSE: {metricas_modelo['RMSE']:.2f}"),
            html.P(f"MAE: {metricas_modelo['MAE']:.2f}"),
        ]

        # Gráfico de predicciones vs reales
        pred_vs_real = go.Figure()
        pred_vs_real.add_trace(go.Scatter(
            x=y_test,
            y=model.predict(X_test),
            mode='markers',
            name='Predicciones'
        ))
        pred_vs_real.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Igualdad',
            line=dict(color='red', dash='dash')
        ))
        pred_vs_real.update_layout(
            title="Comparación de Predicciones vs Valores Reales",
            xaxis_title="Valores Reales (Temperatura)",
            yaxis_title="Valores Predichos (Temperatura)",
            showlegend=True
        )

        return mapa_clima, grafico_temp_24h, grafico_hr, grafico_prec, metricas_texto, pred_vs_real
    else:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div("Cargando datos..."), go.Figure()

if __name__ == "__main__":
    app.run_server(debug=True)