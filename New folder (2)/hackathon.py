import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import customtkinter as ctk
import os


class AdvancedAnomalyDetectorApp:
    def __init__(self):
        # Setup window
        self.root = ctk.CTk()
        self.root.geometry("1400x800")
        self.root.title("Enhanced Advanced Anomaly Detector")

        # Set theme - changing to dark mode for better contrast
        ctk.set_appearance_mode("dark")
        self.root.configure(bg="#1a1a1a")  # Dark background

        self.data = None
        self.anomalies = None
        self.model_choice = tk.StringVar(value="Isolation Forest")  # Default model choice
        self.threshold = tk.DoubleVar(value=3.0)  # Default Z-score threshold

        self.setup_gui()

    def setup_gui(self):
        # Main container
        self.main_container = ctk.CTkFrame(self.root, fg_color="#F7F7F7")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Sidebar for controls
        self.sidebar = self.create_sidebar()
        self.sidebar.pack(side="left", fill="y", padx=(0, 20))

        # Content area
        self.content_area = self.create_content_area()
        self.content_area.pack(side="right", fill="both", expand=True)

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self.main_container, width=250, fg_color="#2b2b2b")  # Wider, darker sidebar

        # Title with gradient-like effect
        title_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        title_frame.pack(pady=20, fill="x")
        
        ctk.CTkLabel(
            title_frame,
            text="Advanced",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#4da6ff",  # Light blue
        ).pack()
        
        ctk.CTkLabel(
            title_frame,
            text="Anomaly Detector",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#80ccff",  # Lighter blue
        ).pack()

        # Buttons with new color scheme
        button_config = {
            "fg_color": "#0066cc",  # Rich blue
            "hover_color": "#0052a3",  # Darker blue for hover
            "text_color": "white",
            "corner_radius": 10,
            "height": 40,
        }

        self.load_button = ctk.CTkButton(
            sidebar,
            text="Load Data",
            command=self.load_file,
            **button_config
        )
        self.load_button.pack(pady=15, padx=20, fill="x")

        # Model selection with styled label
        ctk.CTkLabel(
            sidebar, 
            text="Detection Method:", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#4da6ff"
        ).pack(pady=(20, 5))
        
        # Styled combobox frame
        combo_frame = ctk.CTkFrame(sidebar, fg_color="#363636")
        combo_frame.pack(pady=5, padx=20, fill="x")
        
        self.model_dropdown = ttk.Combobox(
            combo_frame,
            textvariable=self.model_choice,
            state="readonly",
            values=["Isolation Forest"],
        )
        self.model_dropdown.pack(pady=5, padx=5, fill="x")

        # Threshold input with styled label
        ctk.CTkLabel(
            sidebar,
            text="Threshold Value:",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#4da6ff"
        ).pack(pady=(20, 5))
        
        self.threshold_entry = ctk.CTkEntry(
            sidebar,
            textvariable=self.threshold,
            border_color="#0066cc",
            fg_color="#363636",
            text_color="white",
            height=35,
        )
        self.threshold_entry.pack(pady=5, padx=20, fill="x")

        # Action buttons
        self.detect_button = ctk.CTkButton(
            sidebar,
            text="Detect Anomalies",
            command=self.run_detection,
            state="disabled",
            **button_config
        )
        self.detect_button.pack(pady=15, padx=20, fill="x")

        self.save_button = ctk.CTkButton(
            sidebar,
            text="Save Report",
            command=self.save_report,
            state="disabled",
            **button_config
        )
        self.save_button.pack(pady=15, padx=20, fill="x")

        return sidebar

    def create_content_area(self):
        content = ctk.CTkFrame(self.main_container, fg_color="#1a1a1a")

        # Enhanced status bar
        status_frame = ctk.CTkFrame(content, fg_color="#2b2b2b", height=50)
        status_frame.pack(fill="x", padx=20, pady=10)
        status_frame.pack_propagate(False)
        
        self.status_bar = ctk.CTkLabel(
            status_frame,
            text="Ready to process data...",
            font=ctk.CTkFont(size=16),
            text_color="#4da6ff",
            anchor="w",
        )
        self.status_bar.pack(fill="x", padx=15, pady=10)

        # Table area with border
        self.table_frame = ctk.CTkFrame(content, fg_color="#2b2b2b", border_width=1, border_color="#404040")
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Configure Treeview style
        style = ttk.Style()
        style.configure("Treeview", 
                       background="#2b2b2b",
                       foreground="white",
                       fieldbackground="#2b2b2b")
        style.configure("Treeview.Heading",
                       background="#363636",
                       foreground="white")

        # Treeview with scrollbars
        self.data_table = ttk.Treeview(self.table_frame, show="headings", style="Treeview")
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.data_table.yview)
        x_scrollbar = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.data_table.xview)
        self.data_table.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Pack scrollbars and table
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar.pack(side="bottom", fill="x")
        self.data_table.pack(fill="both", expand=True)

        return content

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.display_data(self.data)
                self.status_bar.configure(
                    text=f"Loaded: {os.path.basename(file_path)}"
                )
                self.detect_button.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def display_data(self, data):
        # Clear existing data in table
        self.data_table.delete(*self.data_table.get_children())

        # Set columns
        self.data_table["columns"] = list(data.columns)
        for col in data.columns:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, width=100)

        # Insert rows
        for _, row in data.iterrows():
            self.data_table.insert("", "end", values=row.tolist())

    def preprocess_data(self):
        # Handle missing data
        self.data.fillna(method="ffill", inplace=True)

        # Feature engineering
        self.data["balance_change"] = self.data["account_balance"].diff().fillna(0)
        self.data["percent_change"] = (
            self.data["balance_change"] / self.data["account_balance"].shift(1)
        ).fillna(0)
        scaler = RobustScaler()
        self.data["scaled_transaction_amount"] = scaler.fit_transform(
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

            self.display_data(self.anomalies)
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

    def save_report(self):
        if self.anomalies is None:
            messagebox.showerror("Error", "No anomalies to save!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            self.anomalies.to_csv(file_path, index=False)
            messagebox.showinfo("Success", "Anomalies report saved successfully!")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AdvancedAnomalyDetectorApp()
    app.run()



















