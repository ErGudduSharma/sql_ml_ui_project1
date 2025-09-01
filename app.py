
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import json
import traceback
import threading
import time
import bcrypt
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = 'student-stress-analyzer-secret-key-2025'
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# Database configuration
DB_CONFIG = {
    'user': 'root',
    'password': '050901',
    'host': 'localhost',
    'database': 'test_db'
}

# Global variables for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Not started',
    'results': None,
    'error': None
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

@app.route('/')
def serve_index():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        return html_content
    except FileNotFoundError:
        return "Error: index.html file not found", 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight request'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Simple authentication for demo
    if username == 'admin' and password == 'admin123':
        session['user'] = {'username': 'admin', 'is_admin': True}
        response = jsonify({
            'message': 'Login successful', 
            'user': {'username': 'admin', 'is_admin': True}
        })
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5000')
        return response
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user' in session:
        return jsonify({'authenticated': True, 'user': session['user']})
    return jsonify({'authenticated': False}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'message': 'Logout successful'})

@app.route('/api/data')
def get_data():
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students ORDER BY student_id DESC LIMIT 100")
        data = cursor.fetchall()
        return jsonify(data)
    except Error as e:
        return jsonify({'error': f'Database error: {e}'}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Validate input data
    required_fields = ['age', 'gender', 'academic_performance', 'sleep_quality', 
                      'anxiety_level', 'exercise_hours', 'study_hours', 'social_activity',
                      'financial_stress', 'part_time_job', 'screen_time', 'diet_quality',
                      'smoking', 'alcohol']
    
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    try:
        # Load the trained pipeline
        pipeline = joblib.load('stress_level_pipeline.pkl')
        
        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'age': data['age'],
            'gender': data['gender'],
            'academic_performance': data['academic_performance'],
            'sleep_quality': data['sleep_quality'],
            'anxiety_level': data['anxiety_level'],
            'exercise_hours': data['exercise_hours'],
            'study_hours': data['study_hours'],
            'social_activity': data['social_activity'],
            'financial_stress': data['financial_stress'],
            'part_time_job': data['part_time_job'],
            'screen_time': data['screen_time'],
            'diet_quality': data['diet_quality'],
            'smoking': data['smoking'],
            'alcohol': data['alcohol']
        }])
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        
        # Convert numpy types to Python native types for JSON serialization
        prediction = int(prediction)  # Convert numpy.int64 to Python int
        probabilities = [float(prob) for prob in probabilities]  # Convert numpy.float64 to Python float
        
        # Map stress level to human-readable labels
        stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        stress_label = stress_labels.get(prediction, 'Unknown')
        
        return jsonify({
            'prediction': prediction,
            'stress_level': stress_label,
            'probabilities': probabilities,
            'wellness_score': 100 - (prediction * 33)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predictions/save', methods=['POST'])
def save_prediction():
    data = request.get_json()
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO students 
            (name, age, gender, academic_performance, stress_level, sleep_quality, 
             anxiety_level, exercise_hours, study_hours, social_activity, 
             financial_stress, part_time_job, screen_time, diet_quality, smoking, alcohol)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'Predicted Student', data['age'], data['gender'], data['academic_performance'],
            data['predicted_stress_level'], data['sleep_quality'], data['anxiety_level'],
            data['exercise_hours'], data['study_hours'], data['social_activity'],
            data['financial_stress'], data['part_time_job'], data['screen_time'],
            data['diet_quality'], data['smoking'], data['alcohol']
        ))
        connection.commit()
        return jsonify({'message': 'Prediction saved successfully'})
    except Error as e:
        connection.rollback()
        return jsonify({'error': f'Database error: {e}'}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def train_model():
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['progress'] = 10
        training_status['message'] = 'Connecting to database...'
        
        # Connect to database
        engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        
        training_status['progress'] = 20
        training_status['message'] = 'Fetching data from database...'
        
        # Fetch data
        query = "SELECT * FROM students"
        df = pd.read_sql(query, engine)
        
        if len(df) == 0:
            training_status['error'] = 'No data found in database'
            training_status['is_training'] = False
            return
        
        training_status['progress'] = 30
        training_status['message'] = 'Preprocessing data...'
        
        # Drop unnecessary columns
        df = df.drop(['student_id', 'name'], axis=1, errors='ignore')
        
        # Separate features and target
        X = df.drop('stress_level', axis=1)
        y = df['stress_level']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        training_status['progress'] = 50
        training_status['message'] = 'Training models...'
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
            'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)
        }
        
        # Train and evaluate
        results = {}
        best_accuracy = 0
        best_pipeline = None
        best_model_name = ""
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in models.items():
            training_status['message'] = f'Training {name}...'
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pipeline = pipeline
                best_model_name = name
        
        training_status['progress'] = 90
        training_status['message'] = 'Saving model...'
        
        # Save the best pipeline
        if best_pipeline:
            joblib.dump(best_pipeline, 'stress_level_pipeline.pkl')
            
            training_status['results'] = {
                'best_model': best_model_name,
                'accuracy': best_accuracy,
                'all_results': results
            }
            
            training_status['progress'] = 100
            training_status['message'] = f'Training completed. Best model: {best_model_name}'
            
    except Exception as e:
        training_status['error'] = f'Training failed: {str(e)}'
        print(traceback.format_exc())
    finally:
        training_status['is_training'] = False

@app.route('/api/train', methods=['POST'])
def train():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 409
    
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started'})

@app.route('/api/train/status')
def train_status():
    return jsonify(training_status)

@app.route('/api/model/info')
def model_info():
    try:
        pipeline = joblib.load('stress_level_pipeline.pkl')
        return jsonify({
            'model_loaded': True,
            'model_type': type(pipeline.named_steps['model']).__name__
        })
    except:
        return jsonify({'model_loaded': False})

@app.route('/api/reports')
def get_reports():
    if training_status['results']:
        return jsonify(training_status['results'])
    else:
        return jsonify({'error': 'No training results available'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')