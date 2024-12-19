import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import customtkinter as ctk
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
import numpy as np
import os

class AdvancedAnomalyDetectorApp:
    def __init__(self):
        # Setup window
        self.root = ctk.CTk()
        self.root.geometry("1400x800")
        self.root.title("Advanced Anomaly Detector")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.data = None
        self.anomalies = None
        self.model_choice = tk.StringVar(value="Isolation Forest")  # Default model choice
        self.threshold = tk.DoubleVar(value=3.0)  # Default Z-score threshold
        self.setup_gui()

    def setup_gui(self):
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Sidebar for controls
        self.sidebar = self.create_sidebar()
        self.sidebar.pack(side="left", fill="y", padx=(0, 20))

        # Content area
        self.content_area = self.create_content_area()
        self.content_area.pack(side="right", fill="both", expand=True)

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self.main_container, width=200)

        # Title
        ctk.CTkLabel(
            sidebar, text="Advanced\nAnomaly Detector",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#1f538d"
        ).pack(pady=20)

        # Load data button
        self.load_button = ctk.CTkButton(
            sidebar, text="Load Data",
            command=self.load_file,
            fg_color="#1f538d", hover_color="#2980b9"
        )
        self.load_button.pack(pady=10, padx=20, fill="x")

        # Model selection dropdown
        ctk.CTkLabel(sidebar, text="Select Method:", font=ctk.CTkFont(size=14)).pack(pady=(20, 5))
        self.model_dropdown = ttk.Combobox(
            sidebar, textvariable=self.model_choice, state="readonly",
            values=["Isolation Forest", "DBSCAN + Isolation Forest", "Autoencoder", "Prophet"]
        )
        self.model_dropdown.pack(pady=5, padx=20, fill="x")

        # Threshold input for Z-score
        ctk.CTkLabel(sidebar, text="Set Threshold (Z-Score):", font=ctk.CTkFont(size=14)).pack(pady=(20, 5))
        ctk.CTkEntry(sidebar, textvariable=self.threshold).pack(pady=5, padx=20, fill="x")

        # Detect anomalies button
        self.detect_button = ctk.CTkButton(
            sidebar, text="Detect Anomalies",
            command=self.run_detection, state="disabled",
            fg_color="#1f538d", hover_color="#2980b9"
        )
        self.detect_button.pack(pady=10, padx=20, fill="x")

        # Save report button
        self.save_button = ctk.CTkButton(
            sidebar, text="Save Report",
            command=self.save_report, state="disabled",
            fg_color="#1f538d", hover_color="#2980b9"
        )
        self.save_button.pack(pady=10, padx=20, fill="x")

        return sidebar

    def create_content_area(self):
        content = ctk.CTkFrame(self.main_container)

        # Status bar
        self.status_bar = ctk.CTkLabel(
            content, text="Ready to process data...", font=ctk.CTkFont(size=14), anchor="w"
        )
        self.status_bar.pack(fill="x", padx=20, pady=10)

        # Table area
        self.table_frame = ctk.CTkFrame(content)
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        return content

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.status_bar.configure(
                    text=f"Loaded: {os.path.basename(file_path)}"
                )
                self.detect_button.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def preprocess_data(self):
        # Create required features
        self.data["balance_change"] = self.data["account_balance"].diff().fillna(0)
        self.data["percent_change"] = (
            self.data["balance_change"] / self.data["account_balance"].shift(1)
        ).fillna(0)
        self.data["scaled_transaction_amount"] = StandardScaler().fit_transform(
            self.data[["transaction_amount"]]
        )

    def run_detection(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded!")
            self.preprocess_data()

            method = self.model_choice.get()
            if method == "Isolation Forest":
                self.isolation_forest_detection()
            elif method == "DBSCAN + Isolation Forest":
                self.hybrid_detection()
            elif method == "Autoencoder":
                self.autoencoder_detection()
            elif method == "Prophet":
                self.prophet_detection()

            self.save_button.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection: {e}")

    def isolation_forest_detection(self):
        model = IsolationForest(contamination=0.01, random_state=42)
        features = ["scaled_transaction_amount", "percent_change"]
        self.data["anomaly"] = model.fit_predict(self.data[features])
        self.anomalies = self.data[self.data["anomaly"] == -1]
        self.status_bar.configure(
            text=f"Anomalies detected with Isolation Forest: {len(self.anomalies)}"
        )

    def hybrid_detection(self):
        features = ["scaled_transaction_amount", "percent_change"]
        model = IsolationForest(contamination=0.01, random_state=42)
        self.data["anomaly_if"] = model.fit_predict(self.data[features])
        
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.data["cluster"] = dbscan.fit_predict(self.data[features])
        self.anomalies = self.data[(self.data["anomaly_if"] == -1) & (self.data["cluster"] == -1)]
        self.status_bar.configure(
            text=f"Anomalies detected with Hybrid Method: {len(self.anomalies)}"
        )

    def autoencoder_detection(self):
        # Build autoencoder model
        input_dim = self.data.shape[1]
        model = Sequential([
            Dense(16, activation="relu", input_dim=input_dim),
            Dense(8, activation="relu"),
            Dense(16, activation="relu"),
            Dense(input_dim, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(), loss="mse")
        # Train the model
        X = self.data.values
        model.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        # Reconstruct data
        reconstructed = model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        self.data["mse"] = mse
        threshold = np.percentile(mse, 95)
        self.anomalies = self.data[self.data["mse"] > threshold]
        self.status_bar.configure(
            text=f"Anomalies detected with Autoencoder: {len(self.anomalies)}"
        )

    def prophet_detection(self):
        # Prepare data for Prophet
        df = self.data[["date", "transaction_amount"]].rename(
            columns={"date": "ds", "transaction_amount": "y"}
        )
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        self.data["anomaly"] = (forecast["yhat_lower"] > self.data["transaction_amount"]) | (
            forecast["yhat_upper"] < self.data["transaction_amount"]
        )
        self.anomalies = self.data[self.data["anomaly"]]
        self.status_bar.configure(
            text=f"Anomalies detected with Prophet: {len(self.anomalies)}"
        )

    def save_report(self):
        if self.anomalies is None:
            messagebox.showwarning("Warning", "No anomalies to save!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            self.anomalies.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Report saved to {file_path}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AdvancedAnomalyDetectorApp()
    app.run()




















