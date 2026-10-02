import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONSTANTES Y CONFIGURACIÓN GENERAL
# ============================================================

COLOR_FONDO = "#F0F2F5"
COLOR_TARJETA = "#FFFFFF"
COLOR_BORDE = "#E2E8F0"
COLOR_PRIMARIO = "#2563EB"
COLOR_TEXTO = "#1E293B"
COLOR_TEXTO_SUAVE = "#64748B"
COLOR_NARANJA = "#F59E0B"
COLOR_ROJO = "#EF4444"
COLOR_VERDE = "#10B981"

# Variables enfocadas en Consumo Eléctrico
COLUMNAS_X = ["num_personas", "area_m2", "temp_ambiente_c", "num_electrodomesticos", "horas_aire_ac"]
COLUMNA_Y = "consumo_kwh"


class ConsumoElectricoMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Consumo Eléctrico")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLOR_FONDO)

        self.df = pd.DataFrame()
        self.modelo = None
        self.mae = None
        self.rmse = None
        self.r2 = None
        self.predicciones = None
        self.valores_reales = None

        self.contenido = tk.Frame(self.root, bg=COLOR_FONDO)
        self.contenido.pack(fill="both", expand=True)

        self.invalidar_modelo()

        # Cargar datos predeterminados de consumo eléctrico y entrenar automáticamente
        self.cargar_datos_predeterminados()
        self.entrenar_modelo(mostrar_alerta=False)

        # Muestra la interfaz de entrenamiento/modelo por defecto para visualizar el gráfico de entrada
        self.mostrar_modelo()

    # ========================================================
    # DATOS PREDETERMINADOS (CONSUMO ELÉCTRICO SINTÉTICO)
    # ========================================================

    def cargar_datos_predeterminados(self):
        np.random.seed(42)
        n_muestras = 120

        personas = np.random.randint(1, 7, n_muestras)
        area = np.random.uniform(40.0, 250.0, n_muestras)
        temp = np.random.uniform(22.0, 36.0, n_muestras)
        electrodomesticos = np.random.randint(3, 15, n_muestras)
        horas_aire = np.random.uniform(0.0, 16.0, n_muestras)

        # Simulación del consumo diario en kWh
        consumo = (
            15.0
            + (personas * 4.5)
            + (area * 0.12)
            + (temp * 1.8)
            + (electrodomesticos * 2.1)
            + (horas_aire * 5.2)
            + np.random.normal(0, 8, n_muestras)
        )

        self.df = pd.DataFrame({
            "num_personas": personas,
            "area_m2": area,
            "temp_ambiente_c": temp,
            "num_electrodomesticos": electrodomesticos,
            "horas_aire_ac": horas_aire,
            "consumo_kwh": consumo
        })

    # ========================================================
    # INTERFAZ BASE Y COMPONENTES REUTILIZABLES
    # ========================================================

    def limpiar_contenido(self):
        for widget in self.contenido.winfo_children():
            widget.destroy()

    def crear_cabecera(self, titulo, subtitulo):
        header = tk.Frame(self.contenido, bg=COLOR_FONDO)
        header.pack(fill="x", padx=36, pady=(20, 15))

        tk.Label(
            header,
            text=titulo,
            font=("Segoe UI", 18, "bold"),
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO,
        ).pack(anchor="w")

        tk.Label(
            header,
            text=subtitulo,
            font=("Segoe UI", 10),
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO_SUAVE,
        ).pack(anchor="w")

    def boton(self, parent, texto, comando, color_bg):
        return tk.Button(
            parent,
            text=texto,
            command=comando,
            bg=color_bg,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
        )

    def crear_tarjeta(self, parent, titulo, valor, color_borde):
        card = tk.Frame(
            parent,
            bg=COLOR_TARJETA,
            highlightbackground=color_borde,
            highlightthickness=2,
            padx=15,
            pady=12,
        )
        card.pack(side="left", fill="both", expand=True, padx=6)

        tk.Label(
            card,
            text=titulo,
            font=("Segoe UI", 8, "bold"),
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
        ).pack(anchor="w")

        tk.Label(
            card,
            text=valor,
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO,
        ).pack(anchor="w", pady=(4, 0))

    # ========================================================
    # GESTIÓN DEL MODELO
    # ========================================================

    def invalidar_modelo(self):
        self.modelo = None
        self.mae = None
        self.rmse = None
        self.r2 = None
        self.predicciones = None
        self.valores_reales = None

    def entrenar_modelo(self, mostrar_alerta=True):
        if len(self.df) < 5:
            if mostrar_alerta:
                messagebox.showwarning(
                    "Datos insuficientes",
                    "Necesitas al menos 5 registros para entrenar el modelo.",
                )
            return

        X = self.df[COLUMNAS_X]
        y = self.df[COLUMNA_Y]

        if len(self.df) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        self.modelo = LinearRegression()
        self.modelo.fit(X_train, y_train)

        y_pred = self.modelo.predict(X_test)

        self.mae = mean_absolute_error(y_test, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        self.r2 = r2_score(y_test, y_pred)

        self.predicciones = y_pred
        self.valores_reales = y_test.values

        if mostrar_alerta:
            messagebox.showinfo(
                "Entrenamiento completado",
                "El modelo de Regresión Lineal para consumo eléctrico se ha entrenado con éxito.",
            )
            self.mostrar_modelo()

    # ========================================================
    # MOSTRAR MODELO Y DIBUJAR GRÁFICO
    # ========================================================

    def mostrar_modelo(self):
        self.limpiar_contenido()

        self.crear_cabecera(
            "Entrenamiento y Evaluación del Consumo",
            "Métricas del modelo de regresión entrenado y gráfica de comparación real vs. predicho.",
        )

        panel_superior = tk.Frame(self.contenido, bg=COLOR_FONDO)
        panel_superior.pack(fill="x", padx=36, pady=(0, 15))

        self.boton(
            panel_superior,
            "⚙️ Reentrenar modelo",
            lambda: self.entrenar_modelo(mostrar_alerta=True),
            COLOR_PRIMARIO,
        ).pack(side="left", padx=(0, 10))

        self.boton(
            panel_superior,
            "🔮 ir a Predecir",
            self.mostrar_predecir,
            COLOR_VERDE,
        ).pack(side="left")

        tarjetas = tk.Frame(self.contenido, bg=COLOR_FONDO)
        tarjetas.pack(fill="x", padx=30, pady=(0, 15))

        self.crear_tarjeta(
            tarjetas,
            "ERROR ABSOLUTO MEDIO (MAE)",
            self.formatear_metrica(self.mae),
            COLOR_NARANJA,
        )

        self.crear_tarjeta(
            tarjetas,
            "ERROR CUADRÁTICO (RMSE)",
            self.formatear_metrica(self.rmse),
            COLOR_ROJO,
        )

        self.crear_tarjeta(
            tarjetas,
            "COEFICIENTE R²",
            self.formatear_r2(),
            COLOR_VERDE,
        )

        grafico_frame = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )
        grafico_frame.pack(fill="both", expand=True, padx=36, pady=(0, 20))

        if self.modelo is None:
            tk.Label(
                grafico_frame,
                text="Haz clic en 'Reentrenar modelo' para evaluar los datos.",
                bg=COLOR_TARJETA,
                fg=COLOR_TEXTO_SUAVE,
                font=("Segoe UI", 11),
            ).pack(expand=True)
        else:
            self.generar_grafico(grafico_frame)

    # ========================================================
    # GENERAR GRÁFICO CON MATPLOTLIB
    # ========================================================

    def generar_grafico(self, parent):
        # Crear la figura de Matplotlib
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
        fig.patch.set_facecolor(COLOR_TARJETA)
        ax.set_facecolor("#F8FAFC")

        # Dispersión de Puntos (Real vs Predicho)
        ax.scatter(
            self.valores_reales,
            self.predicciones,
            color=COLOR_PRIMARIO,
            edgecolors="w",
            s=70,
            alpha=0.8,
            label="Consumo predicho",
        )

        # Línea de referencia de la predicción ideal
        val_min = min(min(self.valores_reales), min(self.predicciones))
        val_max = max(max(self.valores_reales), max(self.predicciones))

        ax.plot(
            [val_min, val_max],
            [val_min, val_max],
            color=COLOR_ROJO,
            linestyle="--",
            linewidth=1.8,
            label="Predicción ideal (y = x)",
        )

        ax.set_title("Consumo Real vs. Consumo Predicho (kWh)", fontsize=12, fontweight="bold", pad=12, color=COLOR_TEXTO)
        ax.set_xlabel("Consumo real (kWh)", fontsize=10, color=COLOR_TEXTO)
        ax.set_ylabel("Consumo predicho (kWh)", fontsize=10, color=COLOR_TEXTO)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDE)

        plt.tight_layout()

        # Renderizar en el Canvas de Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ========================================================
    # PREDECIR
    # ========================================================

    def mostrar_predecir(self):
        self.limpiar_contenido()

        self.crear_cabecera(
            "Realizar Predicción",
            "Introduce las características de la vivienda/local para estimar el consumo eléctrico diario.",
        )

        contenedor = tk.Frame(self.contenido, bg=COLOR_FONDO)
        contenedor.pack(fill="both", expand=True, padx=36, pady=(0, 20))

        formulario = tk.Frame(
            contenedor,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )
        formulario.pack(side="left", fill="both", expand=True, padx=(0, 15))

        campos_pred = [
            ("Número de personas habitando", "num_personas"),
            ("Área de la vivienda (m²)", "area_m2"),
            ("Temperatura ambiente (°C)", "temp_ambiente_c"),
            ("Cantidad de electrodomésticos", "num_electrodomesticos"),
            ("Horas de uso de Aire Acondicionado/día", "horas_aire_ac"),
        ]

        self.entradas_prediccion = {}

        for texto, nombre in campos_pred:
            bloque = tk.Frame(formulario, bg=COLOR_TARJETA)
            bloque.pack(fill="x", padx=20, pady=8)

            tk.Label(
                bloque,
                text=texto,
                bg=COLOR_TARJETA,
                fg=COLOR_TEXTO_SUAVE,
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w")

            entrada = ttk.Entry(bloque)
            entrada.pack(fill="x", pady=(4, 0))

            self.entradas_prediccion[nombre] = entrada

        self.boton(
            formulario,
            "⚡ Calcular consumo estimado",
            self.calcular_prediccion,
            COLOR_PRIMARIO,
        ).pack(anchor="w", padx=20, pady=(15, 20))

        self.panel_resultado = tk.Frame(
            contenedor,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
            width=350,
        )
        self.panel_resultado.pack(side="right", fill="both", expand=True)

        tk.Label(
            self.panel_resultado,
            text="CONSUMO ESTIMADO DIARIO",
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="center", pady=(50, 10))

        self.lbl_resultado = tk.Label(
            self.panel_resultado,
            text="-- kWh",
            bg=COLOR_TARJETA,
            fg=COLOR_PRIMARIO,
            font=("Segoe UI", 32, "bold"),
        )
        self.lbl_resultado.pack(anchor="center", pady=(0, 20))

        self.boton(
            self.panel_resultado,
            "📊 Ver Evaluación / Gráfico",
            self.mostrar_modelo,
            COLOR_TEXTO_SUAVE,
        ).pack(anchor="center")

    def calcular_prediccion(self):
        try:
            personas = int(self.entradas_prediccion["num_personas"].get())
            area = float(self.entradas_prediccion["area_m2"].get())
            temp = float(self.entradas_prediccion["temp_ambiente_c"].get())
            electro = int(self.entradas_prediccion["num_electrodomesticos"].get())
            horas_ac = float(self.entradas_prediccion["horas_aire_ac"].get())
        except ValueError:
            messagebox.showerror(
                "Datos incorrectos",
                "Por favor, completa todos los campos con valores numéricos válidos.",
            )
            return

        if personas < 1 or area <= 0 or electro < 0 or not (0 <= horas_ac <= 24):
            messagebox.showerror(
                "Rango inválido",
                "Asegúrate de introducir valores coherentes en el formulario.",
            )
            return

        entrada_df = pd.DataFrame([[personas, area, temp, electro, horas_ac]], columns=COLUMNAS_X)

        consumo_estimado = self.modelo.predict(entrada_df)[0]
        consumo_estimado = max(0.0, consumo_estimado)

        self.lbl_resultado.config(text=f"{consumo_estimado:.2f} kWh")

    # ========================================================
    # MÉTODOS AUXILIARES
    # ========================================================

    def formatear_metrica(self, valor):
        if valor is None:
            return "N/A"
        return f"{valor:.2f} kWh"

    def formatear_r2(self):
        if self.r2 is None:
            return "N/A"
        return f"{self.r2:.3f}"


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ConsumoElectricoMLApp(root)
    root.mainloop()