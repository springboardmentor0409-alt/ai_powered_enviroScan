# source.py - Random Forest Model Training for Pollution Source Prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class PollutionSourcePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def generate_training_data(self, n_samples=2000):
        """Generate comprehensive training data with realistic patterns"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Generate realistic pollution data with clear source patterns
            city = np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore'])
            area_type = np.random.choice(['Industrial', 'Commercial', 'Residential', 'Mixed', 'Green'])
            
            # Time-based variations
            hour = np.random.randint(0, 24)
            is_rush_hour = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
            is_weekday = np.random.choice([0, 1], p=[0.3, 0.7])
            
            # Generate features based on realistic patterns
            if area_type == 'Industrial':
                pm25 = np.random.gamma(2.5, 12)
                pm10 = pm25 * 1.8 + np.random.normal(8, 3)
                no2 = np.random.gamma(1.2, 10)
                so2 = np.random.gamma(1.5, 8)
                co = np.random.gamma(1, 1.5)
                o3 = np.random.gamma(0.8, 10)
                true_source = 'Industrial'
                
            elif area_type == 'Commercial':
                pm25 = np.random.gamma(2, 10)
                pm10 = pm25 * 1.7 + np.random.normal(6, 2)
                no2 = np.random.gamma(1.8, 8) * (1 + 0.3 * is_rush_hour)
                so2 = np.random.gamma(0.8, 6)
                co = np.random.gamma(1.2, 1.8) * (1 + 0.2 * is_rush_hour)
                o3 = np.random.gamma(1, 12)
                true_source = 'Vehicular'
                
            elif area_type == 'Residential':
                pm25 = np.random.gamma(1.8, 8)
                pm10 = pm25 * 1.6 + np.random.normal(4, 2)
                no2 = np.random.gamma(1, 6)
                so2 = np.random.gamma(0.6, 4)
                co = np.random.gamma(1.5, 1.2)
                o3 = np.random.gamma(1.2, 10)
                true_source = 'Residential'
                
            elif area_type == 'Mixed':
                pm25 = np.random.gamma(2.2, 11)
                pm10 = pm25 * 1.75 + np.random.normal(7, 2.5)
                no2 = np.random.gamma(1.5, 9)
                so2 = np.random.gamma(1.1, 7)
                co = np.random.gamma(1.3, 1.6)
                o3 = np.random.gamma(1, 11)
                true_source = np.random.choice(['Industrial', 'Vehicular', 'Residential'])
                
            else:  # Green areas
                pm25 = np.random.gamma(1.2, 6)
                pm10 = pm25 * 1.4 + np.random.normal(2, 1)
                no2 = np.random.gamma(0.8, 4)
                so2 = np.random.gamma(0.4, 3)
                co = np.random.gamma(0.8, 1)
                o3 = np.random.gamma(1.5, 8)
                true_source = 'Natural'
            
            record = {
                'id': i + 1,
                'latitude': np.random.uniform(28.4, 28.9),
                'longitude': np.random.uniform(77.1, 77.3),
                'city': city,
                'area_type': area_type,
                'hour': hour,
                'is_rush_hour': is_rush_hour,
                'is_weekday': is_weekday,
                'PM2.5': max(5, pm25),
                'PM10': max(10, pm10),
                'NO2': max(2, no2),
                'SO2': max(1, so2),
                'CO': max(0.1, co),
                'O3': max(5, o3),
                'temperature': np.random.uniform(15, 35),
                'humidity': np.random.uniform(20, 95),
                'wind_speed': np.random.uniform(0, 20),
                'true_source': true_source
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def prepare_features(self, data):
        """Prepare features for model training"""
        # Select feature columns
        feature_columns = [
            'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
            'temperature', 'humidity', 'wind_speed',
            'hour', 'is_rush_hour', 'is_weekday'
        ]
        
        # Add city as encoded feature
        data['city_encoded'] = LabelEncoder().fit_transform(data['city'])
        feature_columns.append('city_encoded')
        
        # Add area type as encoded feature
        data['area_type_encoded'] = LabelEncoder().fit_transform(data['area_type'])
        feature_columns.append('area_type_encoded')
        
        # Select features
        X = data[feature_columns]
        self.feature_names = feature_columns
        
        # Prepare labels
        y = data['true_source']
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """Train the Random Forest model"""
        print("🔄 Training Random Forest model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training completed!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        
        return X_test_scaled, y_test, y_pred, accuracy
    
    def evaluate_model(self, X_test, y_test, y_pred):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("🤖 MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.model.classes_,
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix - Pollution Source Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Create assets directory if it doesn't exist
        os.makedirs('assets', exist_ok=True)
        plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("💾 Confusion matrix saved: assets/confusion_matrix.png")
        plt.close()
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_imp.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importances - Random Forest')
        plt.tight_layout()
        plt.savefig('assets/feature_importance.png', dpi=300, bbox_inches='tight')
        print("💾 Feature importance chart saved: assets/feature_importance.png")
        plt.close()
        
        return feature_imp
    
    def save_model(self, model_path='models/pollution_source_model.joblib'):
        """Save the trained model and preprocessing objects"""
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessing objects
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.model.classes_.tolist()
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Also save feature names separately
        feature_info = {
            'feature_names': self.feature_names,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        joblib.dump(feature_info, 'models/feature_info.joblib')
        print("💾 Feature information saved: models/feature_info.joblib")

def main():
    """Main function to run the model training"""
    print("🌍 POLLUTION SOURCE PREDICTION - RANDOM FOREST TRAINING")
    print("="*60)
    
    # Initialize predictor
    predictor = PollutionSourcePredictor()
    
    # Generate training data
    print("📊 Generating training data...")
    data = predictor.generate_training_data(2000)
    print(f"✅ Generated {len(data)} training samples")
    print(f"📋 Source distribution:\n{data['true_source'].value_counts()}")
    
    # Prepare features
    X, y = predictor.prepare_features(data)
    print(f"✅ Prepared {X.shape[1]} features for training")
    
    # Train model
    print("\n🚀 Training Random Forest model...")
    X_test, y_test, y_pred, accuracy = predictor.train_model(X, y)
    
    # Evaluate model
    feature_importance = predictor.evaluate_model(X_test, y_test, y_pred)
    
    # Save model
    predictor.save_model()
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 TRAINING SUMMARY")
    print("="*60)
    print(f"📊 Model: Random Forest Classifier")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"📈 Features: {len(predictor.feature_names)}")
    print(f"📦 Training samples: {len(data)}")
    print(f"💾 Model saved: models/pollution_source_model.joblib")
    print(f"🖼️  Charts saved: assets/")
    
    # Show top features
    print("\n🏆 Top 10 Most Important Features:")
    top_features = feature_importance.head(10)
    for _, row in top_features.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()