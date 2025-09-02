# Student Stress Analyzer

A comprehensive Flask-based web application that analyzes and predicts student stress levels using machine learning. The system provides an intuitive dashboard for visualizing student data, predicting stress levels, and generating detailed reports.

## 🌟 Features

- **User Authentication**: Secure login system with session management
- **Interactive Dashboard**: Overview of student statistics and stress distribution with dynamic charts
- **Stress Prediction**: Multi-step form with progress tracking to predict student stress levels
- **Data Management**: View all student data in a responsive, searchable table
- **Model Training**: Retrain machine learning models with current data
- **Performance Reports**: View detailed model performance metrics
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations and transitions
- **Real-time Updates**: Live data updates and training progress monitoring

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Bootstrap 5, Custom CSS with animations
- **Charts**: Chart.js for data visualization
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn
- **Database**: MySQL with SQLAlchemy ORM
- **API**: RESTful API design with JSON responses

## 📊 Project Workflow

### 1️⃣ Project Setup & Database Integration
- Created Flask application with SQLAlchemy for database operations
- Connected to MySQL database with student stress data
- Designed database schema with comprehensive student metrics

### 2️⃣ Machine Learning Pipeline
- Imported data from SQL database for analysis
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Implemented techniques to prevent overfitting:
  - Regularization
  - Proper train-test split (80/20)
  - Stratified cross-validation
- Trained multiple models including Random Forest and XGBoost
- Saved model artifacts for production use

### 3️⃣ Web Application Development
- Built responsive frontend with modern UI components
- Implemented smooth page transitions and animations
- Created interactive dashboard with real-time data visualization
- Developed multi-step prediction form with progress tracking

### 4️⃣ API Development & Integration
- Built RESTful API endpoints using Flask
- Implemented input validation and error handling
- Created comprehensive API documentation

### 5️⃣ Testing & Deployment
- Tested API endpoints with Postman
- Verified prediction accuracy with sample data
- Prepared application for deployment

## 🚀 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/student-stress-analyzer.git
cd student-stress-analyzer
Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up MySQL database:

Create a database named test_db

Import the SQL schema from database/schema.sql

Configure database connection in app.py if needed:

python
DB_CONFIG = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'database': 'test_db'
}
Run the application:

bash
python app.py
Access the application at http://localhost:5000

📖 Usage Guide
Login: Use the default credentials (admin/admin123)

Dashboard: View overview statistics and interactive charts

Predict Stress: Fill out the multi-step form to predict stress levels

Student Data: Browse all student records in a sortable, filterable table

Reports: View model performance metrics and training history

Train Model: Retrain the machine learning model with current data

🔌 API Endpoints
POST /api/login - User authentication

GET /api/data - Fetch student data

POST /api/predict - Predict stress level from input data

POST /api/predictions/save - Save prediction results to database

POST /api/train - Train machine learning model

GET /api/train/status - Check training progress

GET /api/model/info - Get model information

GET /api/reports - Get model performance reports

🗂️ Project Structure

stress_project/
│── app.py                      # Main Flask application
│── index.html                  # Frontend HTML file
│── ml_pipeline.ipynb           # Jupyter notebook for ML workflow
│── preprocessor.pkl            # Saved preprocessing pipeline
│── requirements.txt            # Python dependencies
│── sql_ml.session.sql          # SQL database schema/operations
│── stress_level_pipeline.pkl   # Full ML pipeline (preprocessing + model)
│── steps.txt                   # Setup/installation instructions
│── README.md                   # Project documentation
│── pyproject.toml              # Python project config (modern setup)
│── .gitignore                  # Git ignore rules
│── .env                        # Environment variables
│── .python-version             # Python version specification
│── uv.lock                     # Dependency lock file (using uv)
│── .vscode/                    # VS Code configuration
│── __pycache__/                # Python cache files
│── .venv/                      # Python virtual environment

🧠 Machine Learning Details
Data Preprocessing
Handled missing values with median/mode imputation

Scaled numerical features using StandardScaler

Encoded categorical variables using One-Hot Encoding

Addressed class imbalance with SMOTE

Model Training
Implemented Random Forest and XGBoost classifiers

Used Stratified K-Fold cross-validation (3 folds)

Selected best model based on accuracy metrics

Saved pipeline including preprocessing and model

Evaluation Metrics
Accuracy: Overall prediction correctness

Cross-validation scores: Model stability assessment

Precision/Recall: Performance on minority classes

🧪 Testing the API
Using Postman
Set request method to POST

Use endpoint: http://localhost:5000/api/predict

Set Headers: Content-Type: application/json

Send JSON body:

json
{
  "age": 20,
  "gender": "Male",
  "academic_performance": 4,
  "sleep_quality": 3,
  "anxiety_level": 6,
  "exercise_hours": 2.5,
  "study_hours": 5.0,
  "social_activity": 3,
  "financial_stress": 7,
  "part_time_job": "Yes",
  "screen_time": 8.0,
  "diet_quality": 5,
  "smoking": 0,
  "alcohol": 2.0
}
Expected response:

json
{
  "prediction": 1,
  "stress_level": "Medium",
  "probabilities": [0.2, 0.6, 0.2],
  "wellness_score": 67
}
🚦 Running the Application
Start the Flask server:

bash
python app.py
Access the application:

Open browser to: http://localhost:5000

Login with username: admin, password: admin123

Train the initial model:

Click on "Train Model" in the sidebar

Confirm the training operation

Monitor progress in the interface

Make predictions:

Navigate to "Predict Stress" in the sidebar

Fill out the multi-step form

View the prediction results

🛠️ Troubleshooting
Database connection errors: Check MySQL service is running and credentials are correct

Module import errors: Verify all dependencies are installed from requirements.txt

Model training failures: Ensure sufficient data exists in the database

Prediction errors: Check that all required fields are provided in the correct format

📈 Future Enhancements
User registration and role-based access control

Advanced visualization with interactive filters

Export functionality for reports and data

Real-time notifications for training completion

Model versioning and comparison

Automated scheduled retraining

Integration with additional data sources

👥 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Bootstrap for UI components

Chart.js for data visualization

Scikit-learn for machine learning algorithms

Font Awesome for icons

The research community for student stress analysis studies


Note: This application is designed for educational purposes. Always consult with mental health professionals for actual stress assessment and management.

