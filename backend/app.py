import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os, pandas as pd, pickle, io, datetime, numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import warnings
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'devsecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max uploads

# Configure CORS
from flask_cors import CORS

app.config['CORS_ORIGINS'] = ['http://localhost:5173']
CORS(app, 
     resources={r"/api/*": {"origins": ["http://localhost:5173"]}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Range", "X-Content-Range"])

# Configure SocketIO
socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173", async_mode='threading')

# Track training pause state per model
training_paused = {}

# Configure session
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.session_protection = "strong"

# ===== Models =====
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    file_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    description = db.Column(db.Text, nullable=True)
    # NOTE: keep model fields in sync with the existing DB schema.
    # Historical schema uses `target_feature` and `train_test_split`.
    # Avoid adding `target_variable`/`split_percent` here unless you run a DB migration.
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)
    input_features = db.Column(db.String(500), nullable=True)  # comma-separated feature names
    target_feature = db.Column(db.String(100), nullable=True)
    regression = db.Column(db.Boolean, default=False, nullable=True) # True if regression, False if classification
    train_test_split = db.Column(db.Float, nullable=True)
    
class ModelEntry(db.Model):
    """
    Stores models in DB. model_blob contains a pickled model object.
    params and metrics are pickled dicts.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    model_type = db.Column(db.String(80), nullable=False)
    # params and metrics stored as JSON strings
    params = db.Column(db.String(500), nullable=True)
    metrics = db.Column(db.String(500), nullable=True)
    model_blob = db.Column(db.LargeBinary, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)

def preprocess(df, input_features, target_feature, test_split):
    """
    Preprocess the dataset by removing missing values, duplicates, and splitting into train/test sets.
    
    Args:
        df: pandas DataFrame containing the dataset
        input_features: list of feature column names
        target_feature: name of the target column
        test_split: percentage of data to use for testing (e.g., 20 for 20%)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessing_info)
    """
    # Store original shape
    original_rows = df.shape[0]
    
    # Remove rows with missing values
    df_cleaned = df.dropna()
    rows_after_missing = df_cleaned.shape[0]
    missing_removed = original_rows - rows_after_missing
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    rows_after_duplicates = df_cleaned.shape[0]
    duplicates_removed = rows_after_missing - rows_after_duplicates
    
    # Separate features and target
    if input_features and all(col in df_cleaned.columns for col in input_features):
        X = df_cleaned[input_features]
    else:
        X = df_cleaned.drop(columns=[target_feature])

    if target_feature and target_feature in df_cleaned.columns:
        y = df_cleaned[target_feature]
        X = df_cleaned.drop(columns=[target_feature])
    else:
        # Fallback to last column as target
        X = df_cleaned.iloc[:, :-1]
        y = df_cleaned.iloc[:, -1]
    
    # Convert test_split percentage to decimal (e.g., 20 -> 0.20)
    test_size = test_split / 100.0
    # Ensure test_size is within valid range
    test_size = max(0.01, min(0.99, test_size))
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Create preprocessing info dictionary
    preprocessing_info = {
        'original_rows': original_rows,
        'missing_values_removed': missing_removed,
        'duplicates_removed': duplicates_removed,
        'final_rows': rows_after_duplicates,
        'test_split_percentage': test_split,
        'test_size_decimal': test_size,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return X_train, X_test, y_train, y_test, preprocessing_info
    
# ===== Model wrapper/helper =====
class ModelWrapper:
    """
    Thin wrapper to standardize training/predicting and DB serialization.
    model_type: one of ('linear_regression','logistic_regression','decision_tree',
                       'bagging','boosting','random_forest','svm','mlp')
    """
    def __init__(self, model_type, regression=True, model=None, params=None):
        self.model_type = model_type
        self.model = model
        self.params = params or {}
        self.metrics = {}
        self.regression = regression

    def train(self, X, y, progress_callback=None, pause_check=None, **train_kwargs):
        # X,y are pandas or numpy-like
        
        # For MLP: Always create a fresh model instance to avoid state pollution from previous training
        if self.model_type == 'mlp':
            self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)
        elif self.model is None:
            self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)

        # For MLP with streaming support
        if self.model_type == 'mlp' and progress_callback:
            print(f"[MLP Training] Starting epoch-wise training with progress callback")
            
            # Get max iterations from params (NOT from model which may have stale value)
            max_iter = self.params.get('max_iter', 200)
            print(f"[MLP Training] Max iterations from params: {max_iter}")
            
            # Enable warm_start and set max_iter to 1 for incremental training
            self.model.warm_start = True
            self.model.n_iter_no_change = max_iter  # Disable early stopping for streaming
            
            # Track previous loss for convergence
            prev_loss = float('inf')
            tol = self.model.tol if hasattr(self.model, 'tol') else 1e-4
            
            for epoch in range(max_iter):
                # Check if training is paused
                if pause_check and pause_check():
                    print(f"[MLP Training] Training paused at epoch {epoch + 1}")
                    import time
                    while pause_check():
                        time.sleep(0.5)
                    print(f"[MLP Training] Training resumed at epoch {epoch + 1}")
                
                # Set max_iter to train one more epoch
                self.model.max_iter = epoch + 1
                
                # Train (with warm_start, this continues from previous state)
                self.model.fit(X, y)
                
                # Calculate metrics for this epoch
                train_preds = self.model.predict(X)
                train_loss = self.model.loss_ if hasattr(self.model, 'loss_') else 0.0
                if self.regression:
                    train_metric = mean_squared_error(y, train_preds)
                else:
                    train_metric = accuracy_score(y, train_preds)

                print(f"[MLP Training] Epoch {epoch + 1}/{max_iter} - Loss: {train_loss:.6f}, {'MSE' if self.regression else 'Accuracy'}: {train_metric:.4f}")

                # Send progress update with regression flag
                progress_callback({
                    'epoch': epoch + 1,
                    'loss': float(train_loss),
                    'metric': float(train_metric),
                    'regression': self.regression
                })
                
                # Check for convergence (loss not improving)
                if epoch > 0 and abs(prev_loss - train_loss) < tol:
                    print(f"[MLP Training] Converged at epoch {epoch + 1}")
                    break
                
                prev_loss = train_loss
            
            print(f"[MLP Training] Training completed")
        else:
            # For scikit-learn style models (standard training)
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            else:
                # If the model is custom and doesn't have fit, raise
                raise ValueError("Model does not support .fit() - provide a fitted model blob for 'custom' models")

    def predict(self, X):
        if self.model is None:
            raise ValueError("No model loaded")
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # If model is pickled custom object without predict, try calling it as function
            try:
                return self.model(X)
            except Exception as e:
                raise ValueError(f"Model has no predict method and is not callable: {e}")

    def evaluate(self, X, y):
        preds = self.predict(X)
        if self.regression:
            self.metrics['mse'] = float(mean_squared_error(y, preds))
        else:
            # classification metrics
            try:
                self.metrics['accuracy'] = float(accuracy_score(y, preds))
            except Exception:
                self.metrics['accuracy'] = None
        return self.metrics

    def to_db_record(self, name, dataset_id, user_id):
        model_blob = pickle.dumps(self.model)
        # convert params and metrics to json strings
        params = json.dumps(self.params)
        metrics = json.dumps(self.metrics)
        return ModelEntry(
            name=name,
            model_type=self.model_type,
            params=params,
            metrics=metrics,
            model_blob=model_blob,
            dataset_id=dataset_id,
            user_id=user_id
        )

    @staticmethod
    def from_db_record(record: ModelEntry):
        # convert params and metrics from json strings to dicts
        params = json.loads(record.params) if record.params else {}
        metrics = json.loads(record.metrics) if record.metrics else {}
        model = pickle.loads(record.model_blob)
        dataset = Dataset.query.get(record.dataset_id)
        regression = dataset.regression if dataset else False
        wrapper = ModelWrapper(model_type=record.model_type, model=model, params=params, regression=regression)
        wrapper.metrics = metrics
        return wrapper

    @staticmethod
    def _instantiate_from_type(regression, model_type, params):
        # Map types to sklearn classes
        if model_type == 'linear_regression':
            if not regression:
                raise ValueError("Linear Regression model requires regression=True")
            return LinearRegression(**(params or {}))
        if model_type == 'logistic_regression':
            if regression:
                raise ValueError("Logistic Regression model requires regression=False")
            return LogisticRegression(**(params or {}))
        if model_type == 'decision_tree':
            if regression:
                return DecisionTreeRegressor(**(params or {}))
            return DecisionTreeClassifier(**(params or {}))
        if model_type == 'random_forest':
            if regression:
                return RandomForestRegressor(**(params or {}))
            return RandomForestClassifier(**(params or {}))
        if model_type == 'bagging':
            if regression:
                return BaggingRegressor(**(params or {}))
            return BaggingClassifier(**(params or {}))
        if model_type == 'boosting':
            if regression:
                return GradientBoostingRegressor(**(params or {}))
            return AdaBoostClassifier(**(params or {}))
        if model_type == 'svm':
            if regression:
                return SVR(**(params or {}))
            return SVC(**(params or {}))
        if model_type == 'mlp':
            # Convert hidden_layer_sizes from list to tuple for sklearn
            mlp_params = params.copy() if params else {}
            if 'hidden_layer_sizes' in mlp_params and isinstance(mlp_params['hidden_layer_sizes'], list):
                mlp_params['hidden_layer_sizes'] = tuple(mlp_params['hidden_layer_sizes'])
            if regression:
                return MLPRegressor(**mlp_params)
            return MLPClassifier(**mlp_params)
        raise ValueError(f"Unknown model_type: {model_type}")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/api/test')
def test():
    return "Backend is running! API version."

# ===== Routes =====
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email exists'}), 400
    user = User(email=data['email'], password_hash=generate_password_hash(data['password']))
    db.session.add(user)
    db.session.commit()
    return jsonify({'msg': 'Registered'}), 201

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    try:
        if request.method == 'OPTIONS':
            return jsonify({}), 200
            
        print("Login attempt received")
        data = request.json
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        print(f"Login attempt for email: {data.get('email')}")
        user = User.query.filter_by(email=data['email']).first()
        
        if not user:
            print("User not found")
            return jsonify({'error': 'Invalid credentials'}), 401
            
        if not check_password_hash(user.password_hash, data['password']):
            print("Invalid password")
            return jsonify({'error': 'Invalid credentials'}), 401
            
        login_user(user)
        print(f"User {user.id} logged in successfully")
        return jsonify({
            'msg': 'Logged in successfully',
            'user': {
                'id': user.id,
                'email': user.email
            }
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'msg': 'Logged out'})

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
@login_required
def upload():
    try:
        print("\n=== Upload Request ===")
        print(f"Request method: {request.method}")
        print(f"Request form data: {request.form}")
        print(f"Request files: {request.files}")
        
        if request.method == 'OPTIONS':
            return jsonify({}), 200
            
        file = request.files.get('file')
        custom_name = request.form.get('name')
        description = request.form.get('description')
        regression = request.form.get('regression', 'false').lower() == 'true'
        input_features = request.form.get('input_features')
        target_feature = request.form.get('target_feature')
        
        print(f"Received file: {file.filename if file else 'None'}")
        print(f"Custom name: {custom_name}")
        print(f"Description: {description}")
        print(f"Regression/Classification: {'Regression' if regression else 'Classification'}")
        print(f"Input features: {input_features}")
        print(f"Target feature: {target_feature}")
        
        if not input_features:
            input_features = None

        if not file:
            return jsonify({'error': 'No file provided'}), 400
        if not custom_name:
            return jsonify({'error': 'No dataset name provided'}), 400
            
        # Get original file extension
        original_ext = os.path.splitext(file.filename)[1].lower()
        print(f"File extension: {original_ext}")
        
        if original_ext not in ['.csv', '.txt', '.xlsx']:
            return jsonify({'error': f'Unsupported file format: {original_ext}. Supported formats: .csv, .txt, .xlsx'}), 400
        
        # Create safe filename with custom name and original extension
        filename = secure_filename(f"{custom_name}{original_ext}")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file with same name already exists
        if os.path.exists(path):
            return jsonify({'error': 'A dataset with this name already exists'}), 400
            
        try:
            file.save(path)
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
            
        try:
            # Save name with extension to ensure consistency when retrieving
            ds = Dataset(name=filename, file_path=path, user_id=current_user.id, description=description, regression=regression, input_features=input_features, target_feature=target_feature)
            db.session.add(ds)
            db.session.commit()
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            # Clean up the file if database operation fails
            if os.path.exists(path):
                os.remove(path)
            return jsonify({'error': f'Failed to save dataset to database: {str(e)}'}), 500
            
        try:
            # Read dataset info based on file extension
            if original_ext == '.csv':
                df = pd.read_csv(path)
            elif original_ext == '.txt':
                df = pd.read_csv(path, sep='\t')  # Assuming tab-separated values
            elif original_ext == '.xlsx':
                try:
                    import openpyxl
                    df = pd.read_excel(path, engine='openpyxl')
                except ImportError:
                    return jsonify({
                        'msg': 'Uploaded but could not read file contents',
                        'dataset': {
                            'id': ds.id,
                            'name': filename,
                            'file_path': path,
                            'upload_date': ds.created_at.isoformat(),
                            'error': 'Server missing openpyxl package required for Excel files'
                        }
                    })
                except Exception as excel_error:
                    return jsonify({
                        'msg': 'Uploaded but could not read file contents',
                        'dataset': {
                            'id': ds.id,
                            'name': filename,
                            'file_path': path,
                            'upload_date': ds.created_at.isoformat(),
                            'error': f'Error reading Excel file: {str(excel_error)}'
                        }
                    })
            else:
                return jsonify({'error': 'Unsupported file format'}), 400

            file_size = os.path.getsize(path)
            rows, features = df.shape
        except Exception as e:
            print(f"Error reading file contents: {str(e)}")
            # Don't delete the file or database entry here as they're valid
            # Just return basic info without the detailed stats
            return jsonify({
                'msg': 'Uploaded but could not read file contents',
                'dataset': {
                    'id': ds.id,
                    'name': filename,
                    'description': description,
                    'ext' : original_ext,
                    'file_path': path,
                    'file_size': file_size,
                    'rows': rows,
                    'features': features,
                    'regression': ds.regression,
                    'input_features': input_features,
                    'target_feature': target_feature,
                    'upload_date': ds.created_at.isoformat(),
                    'error': f'Could not read file contents: {str(e)}'
                }
            })

        return jsonify({
            'msg': 'Uploaded successfully',
            'dataset': {
                'id': ds.id,
                'name': filename,
                'description': description,
                'ext' : original_ext,
                'file_path': path,
                'file_size': file_size,
                'rows': rows,
                'features': features,
                'regression': ds.regression,
                'input_features': input_features,
                'target_feature': target_feature,
                'upload_date': ds.created_at.isoformat(),
                'models': 0
            }
        })
        
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# This after_request handler has been replaced by Flask-CORS

@app.route('/api/datasets', methods=['GET', 'OPTIONS'])
def get_datasets():
    try:
        print("\n=== Dataset Request ===")
        print("Received request for /api/datasets")
        print(f"Method: {request.method}")
        
        if request.method == 'OPTIONS':
            return jsonify({}), 200

        print("\nChecking authentication...")
        print(f"current_user type: {type(current_user)}")
        print(f"current_user dir: {dir(current_user)}")
        
        if not hasattr(current_user, 'is_authenticated'):
            print("Error: current_user doesn't have is_authenticated")
            return jsonify({'error': 'Authentication system not properly initialized'}), 500
            
        print(f"is_authenticated: {current_user.is_authenticated}")
        if not current_user.is_authenticated:
            print("User is not authenticated")
            return jsonify({'error': 'Not authenticated'}), 401
            
        print(f"Fetching datasets for user {current_user.id}")
        datasets = Dataset.query.filter_by(user_id=current_user.id).all()
        print(f"Found {len(datasets)} datasets")
        
        result = []
        for ds in datasets:
            try:
                print(f"Processing dataset: {ds.name}")
                # Add basic information even if file processing fails
                # Format timestamp safely
                try:
                    upload_date = ds.created_at.isoformat() if ds.created_at else None
                except AttributeError:
                    upload_date = None
                
                dataset_info = {
                    'id': ds.id,
                    'name': ds.name,
                    'description': ds.description,
                    'file_path': ds.file_path,
                    'upload_date': upload_date,
                    'file_size': 0,
                    'rows': 0,
                    'features': 0,
                    'regression': ds.regression,
                    'input_features': ds.input_features if ds.input_features else "",
                    'target_feature': ds.target_feature,
                    'models': ModelEntry.query.filter_by(dataset_id=ds.id).count()
                }
                
                if not ds.file_path:
                    print(f"No file path for dataset {ds.id}")
                    dataset_info['error'] = 'No file path available'
                    result.append(dataset_info)
                    continue
                    
                if not os.path.exists(ds.file_path):
                    print(f"File not found: {ds.file_path}")
                    dataset_info['error'] = 'File not found'
                    result.append(dataset_info)
                    continue
                    
                file_ext = os.path.splitext(ds.name)[1].lower()
                print(f"File extension: {file_ext}")
                
                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(ds.file_path)
                    elif file_ext == '.txt':
                        df = pd.read_csv(ds.file_path, sep='\t')
                    elif file_ext == '.xlsx':
                        try:
                            import openpyxl
                            df = pd.read_excel(ds.file_path, engine='openpyxl')
                        except ImportError:
                            print("openpyxl not installed - required for reading .xlsx files")
                            dataset_info['error'] = 'Server missing openpyxl package required for Excel files'
                            result.append(dataset_info)
                            continue
                        except Exception as excel_error:
                            print(f"Error reading Excel file: {str(excel_error)}")
                            dataset_info['error'] = f'Error reading Excel file: {str(excel_error)}'
                            result.append(dataset_info)
                            continue
                    else:
                        print(f"Unsupported file extension: {file_ext}")
                        dataset_info['error'] = 'Unsupported file format'
                        result.append(dataset_info)
                        continue
                        
                    file_size = os.path.getsize(ds.file_path)
                    rows, features = df.shape
                    
                    # Update with file stats
                    try:
                        dataset_info.update({
                            'file_size': file_size,
                            'rows': int(rows),
                            'features': int(features),
                            'error': None
                        })
                    except (ValueError, TypeError) as e:
                        dataset_info.update({
                            'error': f'Error processing file stats: {str(e)}'
                        })
                    result.append(dataset_info)
                    print(f"Successfully processed dataset {ds.id}")
                    
                except Exception as df_error:
                    print(f"Error reading file {ds.file_path}: {str(df_error)}")
                    dataset_info['error'] = f'Error reading file: {str(df_error)}'
                    result.append(dataset_info)
                    
            except Exception as e:
                print(f"Error processing dataset {ds.id}: {str(e)}")
                result.append({
                    'id': ds.id,
                    'name': ds.name,
                    'error': f'Processing error: {str(e)}'
                })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

from flask import send_from_directory

# get single dataset entry
@app.route('/api/datasets/<int:dataset_id>', methods=['GET'])
@login_required
def get_dataset(dataset_id):
    ds = Dataset.query.get_or_404(dataset_id)
    if ds.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    dataset_info = {
        'id': ds.id,
        'name': ds.name,
        'file_path': ds.file_path,
        'user_id': ds.user_id,
        'created_at': ds.created_at,
        'input_features': ds.input_features,
        'target_feature': ds.target_feature,
        'train_test_split': ds.train_test_split,
        'regression': ds.regression
    }

    return jsonify(dataset_info)

@app.route('/api/download/<int:dataset_id>', methods=['GET'])
@login_required
def download_dataset(dataset_id):
    ds = Dataset.query.get_or_404(dataset_id)
    if not ds.file_path or not os.path.exists(ds.file_path):
        return jsonify({'error': 'File not found'}), 404
    directory = os.path.dirname(ds.file_path)
    filename = os.path.basename(ds.file_path)
    return send_from_directory(directory, filename, as_attachment=True)

# Get dataset columns
@app.route('/api/datasets/<int:dataset_id>/columns', methods=['GET', 'OPTIONS'])
@login_required
def get_dataset_columns(dataset_id):
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    ds = Dataset.query.get_or_404(dataset_id)
    if ds.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    try:
        ext = os.path.splitext(ds.name)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(ds.file_path)
        elif ext == '.txt':
            df = pd.read_csv(ds.file_path, sep='\t')
        elif ext == '.xlsx':
            import openpyxl
            df = pd.read_excel(ds.file_path, engine='openpyxl')
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        return jsonify({'columns': df.columns.tolist()})
    except Exception as e:
        return jsonify({'error': f'Failed to read dataset: {str(e)}'}), 500

# Save dataset configuration
@app.route('/api/datasets/<int:dataset_id>/config', methods=['POST', 'OPTIONS'])
@login_required
def save_dataset_config(dataset_id):
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    ds = Dataset.query.get_or_404(dataset_id)
    if ds.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    data = request.json

    regression = data.get('regression', False)
    input_features = data.get('input_features') # expect a string (comma-separated features)
    target_feature = data.get('target_feature')
    train_test_split = data.get('train_test_split')
    
    if not input_features:
        print("Input features are required")
        return jsonify({'error': 'Input features required'}), 400

    if not target_feature or not isinstance(train_test_split, (int, float)):
        return jsonify({'error': 'Invalid configuration data'}), 400

    try:
        ds.regression = bool(regression)
        ds.input_features = input_features
        ds.target_feature = target_feature
        ds.train_test_split = float(train_test_split)
        db.session.commit()
        return jsonify({'msg': 'Configuration saved successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to save configuration: {str(e)}'}), 500

# Delete dataset (supports OPTIONS for CORS preflight)
@app.route('/api/datasets/<int:dataset_id>', methods=['DELETE', 'OPTIONS', 'GET'])
def delete_dataset(dataset_id):
    # Allow CORS preflight through
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    # GET request returns name of dataset
    if request.method == 'GET':
        ds = Dataset.query.get_or_404(dataset_id)
        if ds.user_id != current_user.id:
            return jsonify({'error': 'Forbidden'}), 403
        return jsonify({'name': ds.name}), 200

    # Require authentication for actual DELETE
    if not (hasattr(current_user, 'is_authenticated') and current_user.is_authenticated):
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        ds = Dataset.query.get_or_404(dataset_id)
        if ds.user_id != current_user.id:
            return jsonify({'error': 'Forbidden'}), 403

        # remove file if present
        try:
            if ds.file_path and os.path.exists(ds.file_path):
                os.remove(ds.file_path)
        except Exception as e:
            print(f"Failed to remove file {ds.file_path}: {str(e)}")

        db.session.delete(ds)
        db.session.commit()
        return jsonify({'msg': 'Deleted'}), 200
    except Exception as e:
        print(f"Error deleting dataset {dataset_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# ===== Model endpoints =====
@app.route('/api/models', methods=['GET'])
@login_required
def list_models():
    # List models owned by current user (optionally filter by dataset_id)
    dataset_id = request.args.get('dataset_id', type=int)
    query = ModelEntry.query.filter_by(user_id=current_user.id)
    if dataset_id:
        query = query.filter_by(dataset_id=dataset_id)
    models = query.order_by(ModelEntry.created_at.desc()).all()
    out = []
    for m in models:
        out.append({
            'id': m.id,
            'name': m.name,
            'description': m.description if hasattr(m, 'description') else 'No description',
            'model_type': m.model_type,
            'dataset_id': m.dataset_id,
            'created_at': m.created_at.isoformat() if m.created_at else None,
            'params': json.loads(m.params) if m.params else {},
            'metrics': json.loads(m.metrics) if m.metrics else {}
        })
    return jsonify(out)

def _load_dataset_for_model(m_entry: ModelEntry):
    ds = Dataset.query.get_or_404(m_entry.dataset_id)
    ext = os.path.splitext(ds.name)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(ds.file_path)
    elif ext == '.txt':
        df = pd.read_csv(ds.file_path, sep='\t')
    elif ext == '.xlsx':
        import openpyxl
        df = pd.read_excel(ds.file_path, engine='openpyxl')
    else:
        raise ValueError(f'Unsupported dataset format: {ext}')
    df.dropna(inplace=True)
    if ds.target_feature and ds.target_feature in df.columns:
        y = df[ds.target_feature]
        X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
    else:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # compute split
    test_size = 0.2
    if ds.train_test_split is not None:
        test_size = max(0.01, min(0.99, ds.train_test_split / 100.0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return ds, X, y, X_train, X_test, y_train, y_test

def _get_proba_or_score(model, X):
    # Try predict_proba, else decision_function, else None
    try:
        proba = model.predict_proba(X)
        # Return positive class probabilities for binary
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return None

def _classification_metrics(wrapper: ModelWrapper, X_test, y_test):
    preds = wrapper.predict(X_test)
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metrics['accuracy'] = float(accuracy_score(y_test, preds))
        # Use weighted for robustness across multiclass
        metrics['precision'] = float(precision_score(y_test, preds, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_test, preds, average='weighted', zero_division=0))
        metrics['f1'] = float(f1_score(y_test, preds, average='weighted', zero_division=0))
    # ROC-AUC and curves for binary classification if possible
    y_score = _get_proba_or_score(wrapper.model, X_test)
    try:
        if y_score is not None:
            # If y_test has exactly 2 classes, compute ROC-AUC and curve
            unique = np.unique(y_test)
            if unique.shape[0] == 2:
                # Convert y to {0,1}
                y_bin = (y_test == unique.max()).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_score)
                metrics['roc_auc'] = float(roc_auc_score(y_bin, y_score))
                metrics['roc_curve'] = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)]
                # PR curve and AUC
                prec, rec, _ = precision_recall_curve(y_bin, y_score)
                pr_points = [{'precision': float(p), 'recall': float(r)} for p, r in zip(prec, rec)]
                metrics['pr_curve'] = pr_points
                metrics['pr_auc'] = float(average_precision_score(y_bin, y_score))
            else:
                # multiclass ROC-AUC (no curve)
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_score, multi_class='ovr'))
    except Exception:
        pass
    return metrics

def _regression_metrics(wrapper: ModelWrapper, X_test, y_test):
    preds = wrapper.predict(X_test)
    return {
        'mse': float(mean_squared_error(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds)),
        'r2': float(r2_score(y_test, preds)),
    }

def _class_imbalance_info(y):
    try:
        values, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        min_pct = float(counts.min()) / float(total)
        imbalance_ratio = float(counts.max()) / float(counts.min()) if counts.min() > 0 else float('inf')
        return {
            'minority_class_percentage': min_pct * 100.0,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': min_pct < 0.35
        }
    except Exception:
        return None

@app.route('/api/models/compare', methods=['GET'])
@login_required
def compare_models_list():
    # Provide a simplified list for selection and leaderboard, reusing list_models data
    models = ModelEntry.query.filter_by(user_id=current_user.id).order_by(ModelEntry.created_at.desc()).all()
    out = []
    for m in models:
        out.append({
            'id': m.id,
            'name': m.name,
            'model_type': m.model_type,
            'dataset_id': m.dataset_id,
            'metrics': json.loads(m.metrics) if m.metrics else {},
        })
    return jsonify(out)

@app.route('/api/models/<int:model_id>/compare/<int:other_id>', methods=['GET'])
@login_required
def compare_two_models(model_id, other_id):
    m1 = ModelEntry.query.get_or_404(model_id)
    m2 = ModelEntry.query.get_or_404(other_id)
    if m1.user_id != current_user.id or m2.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    # Prepare wrappers and datasets
    w1 = ModelWrapper.from_db_record(m1)
    w2 = ModelWrapper.from_db_record(m2)

    # Load datasets and compute metrics separately (datasets may differ)
    def build_result(entry, wrapper):
        ds, X, y, X_train, X_test, y_train, y_test = _load_dataset_for_model(entry)
        # Use stored model as-is for evaluation
        result_metrics = {}
        if ds.regression:
            result_metrics.update(_regression_metrics(wrapper, X_test, y_test))
        else:
            result_metrics.update(_classification_metrics(wrapper, X_test, y_test))
        # Add preprocessing/imbalance info
        prep = {
            'original_rows': int(X.shape[0]),
            'final_rows': int(X.shape[0]),
            'missing_values_removed': 0,
            'duplicates_removed': 0,
        }
        imb = _class_imbalance_info(y_test) if not ds.regression else None
        if imb:
            prep['imbalance'] = imb
        # CV (lightweight)
        cv = None
        try:
            params = json.loads(entry.params) if entry.params else {}
            est = ModelWrapper._instantiate_from_type(ds.regression, entry.model_type, params)
            if ds.regression:
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                cv = {
                    'r2_mean': float(cross_val_score(est, X, y, cv=kf, scoring='r2').mean()),
                    'mse_mean': float((-cross_val_score(est, X, y, cv=kf, scoring='neg_mean_squared_error')).mean()),
                }
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cv = {
                        'accuracy_mean': float(cross_val_score(est, X, y, cv=skf, scoring='accuracy').mean()),
                        'f1_mean': float(cross_val_score(est, X, y, cv=skf, scoring='f1_weighted').mean()),
                    }
        except Exception:
            pass

        # Compose output
        return {
            'id': entry.id,
            'name': entry.name,
            'model_type': entry.model_type,
            'metrics': {**(json.loads(entry.metrics) if entry.metrics else {}), **result_metrics, 'preprocessing': prep},
            'cv': cv
        }

    data = {
        'model1': build_result(m1, w1),
        'model2': build_result(m2, w2),
    }

    if data['model1']['cv']:
        existing_metrics = json.loads(m1.metrics) if m1.metrics else {}
        # for any cv metric not already in stored metrics, add it
        for k, v in data['model1']['cv'].items():
            potential_key = k.replace('mean', '').replace('_', " ").strip()
            if potential_key not in existing_metrics:
                existing_metrics[potential_key] = v
        m1.metrics = json.dumps(existing_metrics)
        db.session.merge(m1)
        db.session.commit()
    if data['model2']['cv']:
        existing_metrics = json.loads(m2.metrics) if m2.metrics else {}
        for k, v in data['model2']['cv'].items():
            potential_key = k.replace('mean', '').replace('_', " ").strip()
            if potential_key not in existing_metrics:
                existing_metrics[potential_key] = v
        m2.metrics = json.dumps(existing_metrics)
        db.session.merge(m2)
        db.session.commit()
        
    return jsonify(data)

# create model from parameters
@app.route('/api/models', methods=['POST'])
@login_required
def create_model():
    data = request.json
    print(f"\n=== Create Model Request ===\nData: {data}")
    name = data.get('name')
    description = data.get('description')
    model_type = data.get('model_type')
    dataset_id = data.get('dataset_id')
    params = data.get('params', {})

    if not name or not model_type:
        return jsonify({'error': 'name and model_type required'}), 400
    if model_type not in ['linear_regression','logistic_regression','decision_tree',
                          'bagging','boosting','random_forest','svm','mlp','custom']:
        return jsonify({'error': 'Invalid model_type'}), 400

    ds = None
    if dataset_id:
        ds = Dataset.query.get_or_404(dataset_id)
        if ds.user_id != current_user.id:
            return jsonify({'error': 'Forbidden'}), 403
    reg = ds.regression if ds else False

    try:
        wrapper = ModelWrapper(model_type=model_type, regression=reg, model=None, params=params)
        entry = wrapper.to_db_record(name=name, dataset_id=ds.id if ds else None, user_id=current_user.id)
        entry.description = description
        db.session.add(entry)
        db.session.commit()
        return jsonify({'msg': 'Model created', 'model_id': entry.id}), 201
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return jsonify({'error': f'Failed to create model: {str(e)}'}), 500

@app.route('/api/models/<int:model_id>', methods=['GET', 'PUT'])
@login_required
def get_model(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    if request.method == 'PUT':
        # update model params from params arg
        data = request.json
        params = data.get('params')
        if not isinstance(params, dict):
            return jsonify({'error': 'params must be a dict'}), 400
        wrapper = ModelWrapper.from_db_record(m)
        wrapper = ModelWrapper(model_type=m.model_type, regression=wrapper.regression, model=wrapper.model, params=params)
        entry = wrapper.to_db_record(name=m.name, dataset_id=m.dataset_id, user_id=m.user_id)
        entry.id = m.id  # keep same ID
        entry.description = m.description
        db.session.merge(entry)
        db.session.commit()
        print(f"Model {m.id} updated with new params: {params}")
        return jsonify({'msg': 'Model updated'}), 200
    if request.method == 'GET':
        return jsonify({
            'id': m.id,
            'name': m.name,
            'description': m.description if hasattr(m, 'description') else 'No description',
            'model_type': m.model_type,
            'dataset_id': m.dataset_id,
            'created_at': m.created_at.isoformat() if m.created_at else None,
            'params': json.loads(m.params) if m.params else {},
            'metrics': json.loads(m.metrics) if m.metrics else {}
        })

@app.route('/api/models/<int:model_id>/download', methods=['GET'])
@login_required
def download_model_blob(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    # return pickled model as binary download
    blob = m.model_blob
    return (blob, 200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': f'attachment; filename=model_{m.id}.pkl'
    })

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
@login_required
def delete_model(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    db.session.delete(m)
    db.session.commit()
    return jsonify({'msg': 'Model deleted'})

def read_dataset_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.txt':
            return pd.read_csv(file_path, sep='\t')
        elif ext == '.xlsx':
            import openpyxl
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f'Unsupported file format: {ext}')
    except Exception as e:
        raise ValueError(f'Failed to read dataset: {str(e)}')
    
@app.route('/api/datasets/<int:dataset_id>/preprocess', methods=['POST', 'OPTIONS'])
@login_required
def preprocess_dataset(dataset_id):
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    ds = Dataset.query.get_or_404(dataset_id)
    if ds.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    if not ds.train_test_split or not ds.target_feature:
        print("Dataset must have target_feature and train_test_split configured")
        return jsonify({'error': 'Dataset must have target_feature and train_test_split configured'}), 400

    if not ds.input_features:
        print("Input features are required")
        return jsonify({'error': 'Input features required'}), 400

    try:
        # Read file temporarily (df is not stored)
        df = read_dataset_file(ds.file_path)
        
        # Get original shape for reporting
        original_rows = df.shape[0]
        
        # Remove missing values
        df_cleaned = df.dropna()
        rows_after_missing = df_cleaned.shape[0]
        missing_removed = original_rows - rows_after_missing
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        rows_after_duplicates = df_cleaned.shape[0]
        duplicates_removed = rows_after_missing - rows_after_duplicates
        
        # Validate target feature exists
        if ds.target_feature not in df_cleaned.columns:
            print(f"Target feature '{ds.target_feature}' not found in dataset")
            return jsonify({'error': f'Target feature "{ds.target_feature}" not found in dataset'}), 400
        
        if not all(col in df_cleaned.columns for col in ds.input_features.split(',')):
            print(f"One or more input features not found in dataset: {ds.input_features}")
            return jsonify({'error': 'One or more input features not found in dataset'}), 400
        
        # Separate features and target
        y = df_cleaned[ds.target_feature]
        X = df_cleaned[ds.input_features.split(',')]
        
        
        # Convert percentage to decimal
        test_size = ds.train_test_split / 100.0
        test_size = max(0.01, min(0.99, test_size))
        
        # Perform train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        preprocessing_info = {
            'original_rows': int(original_rows),
            'missing_values_removed': int(missing_removed),
            'duplicates_removed': int(duplicates_removed),
            'final_rows': int(rows_after_duplicates),
            'test_split_percentage': float(ds.train_test_split),
            'test_size_decimal': float(test_size),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(X.columns),
            'input_features': ds.input_features,
            'target_feature': ds.target_feature
        }
        
        return jsonify({
            'msg': 'Dataset preprocessed successfully',
            'preprocessing': preprocessing_info
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500


@app.route('/api/train/<int:model_id>', methods=['POST'])
@login_required
def train(model_id):
    """
    Train a model on a dataset and save it to DB.
    Expected JSON form-data:
      - hyperparams: optional dict of hyperparameters (JSON)
    """
    # Accept form-data and JSON both
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # hyperparams = request.form.get('hyperparams') or (request.json and request.json.get('hyperparams'))
    # # hyperparams may be a JSON string if form-data; try to parse
    # if isinstance(hyperparams, str):
    #     try:
    #         import json
    #         hyperparams = json.loads(hyperparams)
    #     except Exception:
    #         hyperparams = {}
    # hyperparams = hyperparams or {}
    
    m_entry = ModelEntry.query.get_or_404(model_id)
    if m_entry.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    
    ds = Dataset.query.get_or_404(m_entry.dataset_id)
    if not ds:
        return jsonify({'error': 'Model has no associated dataset to train on'}), 400
    if ds.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    # read dataset and train server-side
    try:
        ext = os.path.splitext(ds.name)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(ds.file_path)
        elif ext == '.txt':
            df = pd.read_csv(ds.file_path, sep='\t')
        elif ext == '.xlsx':
            import openpyxl
            df = pd.read_excel(ds.file_path, engine='openpyxl')
        else:
            print(f"Error: Unsupported dataset format '{ext}' for training")
            return jsonify({'error': 'Unsupported dataset format for training'}), 400
    except Exception as e:
        print(f"Error reading dataset for training: {str(e)}")
        return jsonify({'error': f'Could not read dataset: {e}'}), 500

    if df.shape[1] < 2:
        print(f"Error: Dataset must have at least 2 columns (features + target)")
        return jsonify({'error': 'Dataset must have at least 2 columns (features + target)'}), 400
    
    df.dropna(inplace=True)

    if ds.target_feature and ds.target_feature in df.columns:
        # Use the configured target feature
        y = df[ds.target_feature]
        X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
    else:
        # Fallback to last column as target
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    # Convert percentage (e.g., 20) to decimal (0.20) for sklearn
    test_size = 0.2  # default fallback
    if ds.train_test_split is not None:
        test_size = ds.train_test_split / 100.0
        # Ensure test_size is within valid range (0, 1)
        test_size = max(0.01, min(0.99, test_size))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    try:
        wrapper = ModelWrapper.from_db_record(m_entry)
        wrapper.train(X_train, y_train)
        metrics = wrapper.evaluate(X_test, y_test)
        entry = wrapper.to_db_record(name=m_entry.name, dataset_id=ds.id, user_id=current_user.id)
        entry.id = m_entry.id
        entry.metrics = json.dumps(metrics)
        db.session.merge(entry)
        db.session.commit()
        return jsonify({
            'msg': 'Trained and saved model',
            'model_id': entry.id,
            'metrics': wrapper.metrics,
            'test_size_used': test_size  # Return the actual test size used
        }), 201
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return jsonify({'error': f'Failed to train model: {str(e)}'}), 500

# WebSocket training endpoint for streaming progress
@socketio.on('start_training')
def handle_training(data):
    """
    WebSocket handler for training with real-time progress updates.
    Expects: { 'model_id': int }
    """
    print(f"\n[WebSocket] Received start_training event with data: {data}")
    try:
        model_id = data.get('model_id')
        if not model_id:
            print("[WebSocket] Error: model_id is required")
            emit('training_error', {'message': 'model_id is required'})
            return
        
        # Initialize pause state for this model
        training_paused[model_id] = False
        
        print(f"[WebSocket] Starting training for model_id: {model_id}")
        
        # Verify user is authenticated (SocketIO session)
        # Note: For production, implement proper auth via session or token
        
        m_entry = ModelEntry.query.get(model_id)
        if not m_entry:
            print(f"[WebSocket] Error: Model {model_id} not found")
            emit('training_error', {'message': 'Model not found'})
            return
        
        print(f"[WebSocket] Found model: {m_entry.name} (type: {m_entry.model_type})")
        
        ds = Dataset.query.get(m_entry.dataset_id)
        if not ds:
            print("[WebSocket] Error: Dataset not found")
            emit('training_error', {'message': 'Dataset not found'})
            return
        
        print(f"[WebSocket] Found dataset: {ds.name}")
        
        # Read dataset
        ext = os.path.splitext(ds.name)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(ds.file_path)
        elif ext == '.txt':
            df = pd.read_csv(ds.file_path, sep='\t')
        elif ext == '.xlsx':
            import openpyxl
            df = pd.read_excel(ds.file_path, engine='openpyxl')
        else:
            print(f"[WebSocket] Error: Unsupported format {ext}")
            emit('training_error', {'message': 'Unsupported dataset format'})
            return
        
        print(f"[WebSocket] Loaded dataset with shape: {df.shape}")
        
        df.dropna(inplace=True)
        
        if ds.target_feature and ds.target_feature in df.columns:
            y = df[ds.target_feature]
            X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
        else:
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
        
        print(f"[WebSocket] X shape: {X.shape}, y shape: {y.shape}")
        
        test_size = 0.2
        if ds.train_test_split is not None:
            test_size = ds.train_test_split / 100.0
            test_size = max(0.01, min(0.99, test_size))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        print(f"[WebSocket] Train/test split complete - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Define progress callback
        def progress_callback(metrics):
            print(f"[WebSocket] Emitting metrics: epoch={metrics['epoch']}, loss={metrics['loss']:.6f}, metric={metrics.get('metric', 0):.4f}")
            socketio.emit('training_metrics', {
                'type': 'metrics',
                'epoch': metrics['epoch'],
                'loss': metrics['loss'],
                'metric': metrics.get('metric'),
                'regression': metrics.get('regression', False)
            })
        
        # Define pause check function
        def pause_check():
            return training_paused.get(model_id, False)
        
        print("[WebSocket] Starting model training...")
        # Train with streaming
        wrapper = ModelWrapper.from_db_record(m_entry)
        wrapper.train(X_train, y_train, progress_callback=progress_callback, pause_check=pause_check)
        
        print("[WebSocket] Training complete, evaluating...")
        # Evaluate and save
        final_metrics = wrapper.evaluate(X_test, y_test)
        entry = wrapper.to_db_record(name=m_entry.name, dataset_id=ds.id, user_id=m_entry.user_id)
        entry.id = m_entry.id
        entry.metrics = json.dumps(final_metrics)
        db.session.merge(entry)
        db.session.commit()
        
        print(f"[WebSocket] Emitting training_complete with metrics: {final_metrics}")
        socketio.emit('training_complete', {
            'type': 'complete',
            'message': 'Training completed successfully',
            'metrics': final_metrics
        })
        
    except Exception as e:
        import traceback
        print(f"[WebSocket] Exception occurred:")
        traceback.print_exc()
        socketio.emit('training_error', {
            'type': 'error',
            'message': f'Training failed: {str(e)}'
        })
    finally:
        # Clean up pause state
        if model_id in training_paused:
            del training_paused[model_id]

@socketio.on('pause_training')
def handle_pause(data):
    """Pause training for a specific model"""
    model_id = data.get('model_id')
    if model_id:
        training_paused[model_id] = True
        emit('training_paused', {'model_id': model_id})
        print(f"[WebSocket] Training paused for model {model_id}")

@socketio.on('resume_training')
def handle_resume(data):
    """Resume training for a specific model"""
    model_id = data.get('model_id')
    if model_id:
        training_paused[model_id] = False
        emit('training_resumed', {'model_id': model_id})
        print(f"[WebSocket] Training resumed for model {model_id}")

@app.route('/api/models/<int:model_id>/predict', methods=['POST'])
@login_required
def model_predict(model_id):
    """
    Make predictions with a stored model. Client sends JSON:
      { "input": [[...], [...], ...] }  # list of rows
    Returns predictions (list).
    """
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    wrapper = ModelWrapper.from_db_record(m)
    data = request.json
    if not data or 'input' not in data:
        return jsonify({'error': 'Provide "input" in JSON body as list-of-rows'}), 400
    import numpy as np
    X_in = np.array(data['input'])
    try:
        preds = wrapper.predict(X_in)
        # convert numpy arrays to python lists
        if hasattr(preds, 'tolist'):
            preds = preds.tolist()
        return jsonify({'predictions': preds})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# reset DB/uploads/both - for testing purposes, add fail-safe
@app.route('/api/reset/<string:thing>', methods=['POST'])
def reset(thing):
    key = request.args.get('key')
    if key != 'supersecretresetkey':
        return jsonify({'error': 'Unauthorized'}), 401
    if thing not in ['db', 'uploads', 'both']:
        return jsonify({'error': 'Invalid reset option. Use "db", "uploads", or "both".'}), 400
    if thing == 'db' or thing == 'both':
        db.drop_all()
        db.create_all()
        print("Database reset completed")
    if thing == 'uploads' or thing == 'both':
        folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
        print("Uploads folder reset completed")
    return jsonify({'msg': f'Reset {thing} completed'})

# Create necessary directories
if not os.path.exists('instance'):
    os.makedirs('instance')
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    print(f"\nError occurred: {str(error)}")
    print(f"Error type: {type(error)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(error)}), 500

# Remove any duplicate CORS headers that might have been added
@app.after_request
def cleanup_response(response):
    # Remove any duplicate CORS headers
    for header in ['Access-Control-Allow-Origin', 'Access-Control-Allow-Credentials',
                  'Access-Control-Allow-Methods', 'Access-Control-Allow-Headers']:
        if header in response.headers and response.headers.get_all(header):
            # Keep only the last value if there are duplicates
            response.headers[header] = response.headers.get_all(header)[-1]
    return response

# Initialize database
with app.app_context():
    try:
        # Only create tables if they don't exist
        db.create_all()
        print("Database tables created if they didn't exist")
        
        # Create test user if it doesn't exist
        if not User.query.filter_by(email='test@example.com').first():
            test_user = User(
                email='test@example.com',
                password_hash=generate_password_hash('password123')
            )
            db.session.add(test_user)
            db.session.commit()
            print("Created test user: test@example.com / password123")
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            print("Created uploads directory")
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

@app.route('/api/migrate/add-model-description', methods=['POST'])
def migrate_add_model_description():
    key = request.args.get('key')
    if key != 'supersecretresetkey':
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Add description column if it doesn't exist
        with db.engine.connect() as conn:
            conn.execute(db.text('ALTER TABLE model_entry ADD COLUMN description TEXT'))
            conn.commit()
        return jsonify({'msg': 'Migration completed successfully'})
    except Exception as e:
        # Column might already exist
        return jsonify({'msg': f'Migration note: {str(e)}'})
    

# ===== Main =====
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)
