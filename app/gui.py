"""
GUI Application for Housing Price Prediction
Interactive interface for users to input house features and get price predictions
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import sys
import traceback

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Prediction Tool")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load the trained model
        self.model = None
        self.load_model()
        
        # Create the GUI elements
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model from the models directory"""
        try:
            # Look for the most recent model file (exclude _info.pkl files)
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            model_files = [f for f in os.listdir(models_dir) 
                          if f.startswith('best_model_') and f.endswith('.pkl') and '_info' not in f]
            
            if not model_files:
                raise FileNotFoundError("No trained model found. Please run the pipeline first.")
            
            # Use the most recent model
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
            
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully: {latest_model}")
            
            # Verify it's actually a model with predict method
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded object is not a valid trained model")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🏠 House Price Prediction Tool", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Instructions
        instruction_label = tk.Label(
            self.root,
            text="Enter the house characteristics below to get a price prediction:",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#34495e'
        )
        instruction_label.pack(pady=(0, 20))
        
        # Main frame for inputs
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(padx=40, pady=20, fill='both', expand=True)
        
        # Input fields
        self.create_input_fields(main_frame)
        
        # Prediction button
        predict_button = tk.Button(
            main_frame,
            text="🔮 Predict House Price",
            command=self.predict_price,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=20,
            pady=10
        )
        predict_button.pack(pady=20)
        
        # Result frame
        self.result_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.result_frame.pack(pady=20, fill='x')
        
        # Clear button
        clear_button = tk.Button(
            main_frame,
            text="🗑️ Clear All Fields",
            command=self.clear_fields,
            font=("Arial", 10),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            borderwidth=1,
            padx=15,
            pady=5
        )
        clear_button.pack(pady=10)
    
    def create_input_fields(self, parent):
        """Create input fields for house features"""
        # Store entry widgets
        self.entries = {}
        
        # Define input fields with their properties
        fields = [
            ("sqft_living", "Living Area (sq ft)", "2000", "Total square footage of living space"),
            ("floors", "Number of Floors", "1.0", "Number of floors (e.g., 1.0, 1.5, 2.0)"),
            ("condition", "Condition (1-5)", "3", "Property condition: 1=Poor, 2=Fair, 3=Average, 4=Good, 5=Excellent"),
            ("grade", "Grade (1-13)", "7", "Construction quality: 1-3=Poor, 4-6=Low, 7-9=Average, 10-13=High"),
            ("bedrooms", "Bedrooms", "3", "Number of bedrooms"),
            ("bathrooms", "Bathrooms", "2.0", "Number of bathrooms (e.g., 1.0, 1.5, 2.5)"),
            ("age", "Age (years)", "10", "Age of the property in years"),
            ("sqft_lot", "Lot Size (sq ft)", "7500", "Total lot size in square feet")
        ]
        
        for i, (field_name, label, default, tooltip) in enumerate(fields):
            # Create frame for each field
            field_frame = tk.Frame(parent, bg='#f0f0f0')
            field_frame.pack(fill='x', pady=5)
            
            # Label
            label_widget = tk.Label(
                field_frame,
                text=label + ":",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                fg='#2c3e50',
                width=20,
                anchor='w'
            )
            label_widget.pack(side='left')
            
            # Entry
            entry = tk.Entry(
                field_frame,
                font=("Arial", 10),
                width=15,
                relief='solid',
                borderwidth=1
            )
            entry.pack(side='left', padx=(10, 5))
            entry.insert(0, default)
            self.entries[field_name] = entry
            
            # Tooltip
            tooltip_label = tk.Label(
                field_frame,
                text=f"💡 {tooltip}",
                font=("Arial", 8),
                bg='#f0f0f0',
                fg='#7f8c8d',
                anchor='w'
            )
            tooltip_label.pack(side='left', padx=(5, 0))
    
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            # Get values from entries
            values = {}
            for field_name, entry in self.entries.items():
                value_str = entry.get().strip()
                if not value_str:
                    raise ValueError(f"Please enter a value for {field_name}")
                
                # Convert to float
                values[field_name] = float(value_str)
            
            # Basic validation
            if values['sqft_living'] <= 0:
                raise ValueError("Living area must be greater than 0")
            if values['floors'] <= 0:
                raise ValueError("Number of floors must be greater than 0")
            if not 1 <= values['condition'] <= 5:
                raise ValueError("Condition must be between 1 and 5")
            if not 1 <= values['grade'] <= 13:
                raise ValueError("Grade must be between 1 and 13")
            if values['bedrooms'] < 0:
                raise ValueError("Bedrooms cannot be negative")
            if values['bathrooms'] <= 0:
                raise ValueError("Bathrooms must be greater than 0")
            if values['age'] < 0:
                raise ValueError("Age cannot be negative")
            if values['sqft_lot'] <= 0:
                raise ValueError("Lot size must be greater than 0")
            
            return values
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None
    
    def engineer_features(self, values):
        """Engineer features as done in the training pipeline"""
        try:
            # Create engineered features that match the training pipeline
            engineered = values.copy()
            
            # Domain-specific features
            engineered['total_sqft'] = values['sqft_living'] + values['sqft_lot']
            engineered['quality_score'] = values['grade'] * values['condition']
            engineered['sqft_per_bedroom'] = values['sqft_living'] / max(values['bedrooms'], 1)
            
            # Size category encoding (simplified)
            if values['sqft_living'] < 1500:
                engineered['size_category_encoded'] = 0  # Small
            elif values['sqft_living'] < 2500:
                engineered['size_category_encoded'] = 1  # Medium
            elif values['sqft_living'] < 4000:
                engineered['size_category_encoded'] = 2  # Large
            else:
                engineered['size_category_encoded'] = 3  # XLarge
            
            # Binned features (simplified binning)
            engineered['sqft_living_binned'] = min(int(values['sqft_living'] / 500), 9)
            engineered['floors_binned'] = min(int(values['floors']), 3)
            engineered['grade_binned'] = min(int(values['grade'] / 3), 4)
            engineered['total_sqft_binned'] = min(int(engineered['total_sqft'] / 1000), 15)
            engineered['quality_score_binned'] = min(int(engineered['quality_score'] / 5), 12)
            engineered['size_category_encoded_binned'] = engineered['size_category_encoded']
            engineered['sqft_per_bedroom_binned'] = min(int(engineered['sqft_per_bedroom'] / 200), 10)
            
            # Select only the features used by the model (based on deployment config)
            model_features = [
                'sqft_living', 'floors', 'condition', 'grade', 'total_sqft',
                'quality_score', 'size_category_encoded', 'sqft_per_bedroom',
                'sqft_living_binned', 'floors_binned', 'grade_binned',
                'total_sqft_binned', 'quality_score_binned', 
                'size_category_encoded_binned', 'sqft_per_bedroom_binned'
            ]
            
            # Create feature vector in correct order
            feature_vector = [engineered[feature] for feature in model_features]
            
            return feature_vector, model_features
            
        except Exception as e:
            raise ValueError(f"Error in feature engineering: {str(e)}")
    
    def predict_price(self):
        """Make price prediction based on user inputs"""
        try:
            # Clear previous results
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # Validate inputs
            values = self.validate_inputs()
            if values is None:
                return
            
            # Engineer features
            feature_vector, feature_names = self.engineer_features(values)
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            
            # Display result
            self.display_result(prediction, values)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")
            print(f"Full error: {traceback.format_exc()}")
    
    def display_result(self, prediction, input_values):
        """Display the prediction result"""
        # Result title
        result_title = tk.Label(
            self.result_frame,
            text="🎯 Prediction Result",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        result_title.pack(pady=(0, 10))
        
        # Predicted price
        price_label = tk.Label(
            self.result_frame,
            text=f"Estimated House Price: ${prediction:,.0f}",
            font=("Arial", 16, "bold"),
            bg='#2ecc71',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=20,
            pady=10
        )
        price_label.pack(pady=10)
        
        # Price per square foot
        price_per_sqft = prediction / input_values['sqft_living']
        sqft_label = tk.Label(
            self.result_frame,
            text=f"Price per sq ft: ${price_per_sqft:.0f}",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#34495e'
        )
        sqft_label.pack()
        
        # Confidence note
        confidence_label = tk.Label(
            self.result_frame,
            text="⚠️ This prediction is based on a model with R² = 0.576\n" +
                 "Typical prediction accuracy: ±$88,859 (median error: 5.4%)",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#7f8c8d',
            justify='center'
        )
        confidence_label.pack(pady=(10, 0))
    
    def clear_fields(self):
        """Clear all input fields and results"""
        # Clear input fields
        defaults = {
            'sqft_living': '2000',
            'floors': '1.0',
            'condition': '3',
            'grade': '7',
            'bedrooms': '3',
            'bathrooms': '2.0',
            'age': '10',
            'sqft_lot': '7500'
        }
        
        for field_name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults.get(field_name, ''))
        
        # Clear results
        for widget in self.result_frame.winfo_children():
            widget.destroy()


def main():
    """Main function to run the GUI application"""
    try:
        root = tk.Tk()
        app = HousePricePredictorGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Application Error", f"Failed to start application:\n{str(e)}")


if __name__ == "__main__":
    main()
