import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, pandas as pd, pickle, io, datetime, numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import warnings
from flask_socketio import SocketIO, emit
import threading
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'devsecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max uploads

# ===== SHAP performance knobs (tunable via environment) =====
# Smaller background/eval samples and fewer nsamples make KernelExplainer much faster.
# You can tweak these without code changes using environment variables.
SHAP_BG_N = int(os.environ.get('SHAP_BG_N', '20'))        # default background sample size (was 80)
SHAP_EVAL_N = int(os.environ.get('SHAP_EVAL_N', '60'))     # default eval sample size (was 120)
SHAP_NSAMPLES = int(os.environ.get('SHAP_NSAMPLES', '40')) # default nsamples for KernelExplainer (was 80)
SHAP_N_CHUNKS = int(os.environ.get('SHAP_N_CHUNKS', '20')) # progress chunks (UI smoothness)
SHAP_USE_SAMPLING = os.environ.get('SHAP_USE_SAMPLING', '0') == '1'  # opt-in faster SamplingExplainer

# Initialize DB
db = SQLAlchemy(app)

# Configure CORS
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
# Track early stop requests per model
training_early_stopped = {}

# Configure session
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production

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
    data_source = db.Column(db.Text, nullable=False)  # required: where the data came from
    license_info = db.Column(db.Text, nullable=False)  # required: license terms
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
    early_stopped = db.Column(db.Boolean, default=False, nullable=True)
    current_epoch = db.Column(db.Integer, nullable=True)  # Track last completed epoch for resuming

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

    def train(self, X, y, progress_callback=None, pause_check=None, early_stop_check=None, start_epoch=0, keep_model=False, **train_kwargs):
        # X,y are pandas or numpy-like
        
        # For MLP: Create a fresh model instance unless keep_model=True (for early-stopped retraining)
        if self.model_type == 'mlp':
            if not keep_model:
                # Starting fresh - create new model
                self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)
            # else: keep existing model from blob (for early-stopped retraining)
        elif self.model is None:
            self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)

        # Track the final epoch reached for resuming later
        self.final_epoch = start_epoch

        # For MLP with streaming support
        if self.model_type == 'mlp' and progress_callback:
            print(f"[MLP Training] Starting epoch-wise training with progress callback (start_epoch={start_epoch})")

            # Get max iterations from params (NOT from model which may have stale value)
            max_iter = self.params.get('max_iter', 200)
            print(f"[MLP Training] Max iterations from params: {max_iter}")

            # If the model is a Pipeline (classification path), adjust the final estimator's params
            estimator = None
            is_pipeline = hasattr(self.model, 'named_steps') and 'estimator' in getattr(self.model, 'named_steps', {})
            try:
                if is_pipeline:
                    # Configure warm start and 1-iteration training per fit call on the estimator
                    self.model.set_params(estimator__warm_start=True)
                    # Disable early stopping to avoid validation-split delays and premature stop
                    self.model.set_params(estimator__early_stopping=False)
                    self.model.set_params(estimator__max_iter=1)
                    estimator = self.model.named_steps['estimator']
                    # Fit the preprocessor ONCE to avoid repeated expensive refits per epoch
                    preprocessor = self.model.named_steps.get('preprocess', None)
                    X_fit = X
                    if preprocessor is not None:
                        try:
                            preprocessor.fit(X, y)
                            X_fit = preprocessor.transform(X)
                        except Exception as pe:
                            print(f"[MLP Training] Preprocessor fit/transform failed, falling back to pipeline.fit each epoch: {pe}")
                            preprocessor = None
                    # Store prepared features for reuse in the loop
                    prepared = {'X': X_fit, 'pre': preprocessor}
                else:
                    # MLPRegressor path (no pipeline)
                    self.model.warm_start = True
                    # Disable early stopping (MLP defaults to False, but be explicit)
                    if hasattr(self.model, 'early_stopping'):
                        self.model.early_stopping = False
                    # Train one iteration per fit call
                    self.model.max_iter = 1
                    estimator = self.model
                    prepared = {'X': X}
            except Exception as e:
                print(f"[MLP Training] Warning: Failed to configure warm-start incremental training: {e}")
                estimator = self.model
                prepared = {'X': X}

            # Track previous loss for convergence
            prev_loss = float('inf')
            # Pull tolerance from the actual estimator if available
            try:
                tol = getattr(estimator, 'tol', 1e-4)
            except Exception:
                tol = 1e-4

            for epoch in range(start_epoch, max_iter):
                # Check if early stop was requested
                if early_stop_check and early_stop_check():
                    print(f"[MLP Training] Early stop requested - saving progress at epoch {self.final_epoch}")
                    break

                # Check if training is paused
                if pause_check and pause_check():
                    print(f"[MLP Training] Training paused at epoch {epoch + 1}")
                    import time
                    while pause_check():
                        if early_stop_check and early_stop_check():
                            print(f"[MLP Training] Early stop requested during pause - saving progress at epoch {self.final_epoch}")
                            break
                        time.sleep(0.5)
                    print(f"[MLP Training] Training resumed at epoch {epoch + 1}")

                # Train exactly one iteration per call (warm_start keeps weights)
                try:
                    if is_pipeline and prepared.get('pre') is not None:
                        # Train only the estimator with pre-transformed features
                        estimator.fit(prepared['X'], y)
                    else:
                        # Fallback: train the model (pipeline or raw estimator)
                        self.model.fit(X, y)
                except Exception as fit_err:
                    print(f"[MLP Training] Fit error at epoch {epoch + 1}: {fit_err}")
                    raise

                # Calculate metrics for this epoch
                if is_pipeline and prepared.get('pre') is not None:
                    train_preds = estimator.predict(prepared['X'])
                else:
                    train_preds = self.model.predict(X)
                # Get loss from the underlying estimator when using a Pipeline
                try:
                    current_estimator = estimator
                    # Refresh reference in case Pipeline cloned estimator internally
                    if is_pipeline and hasattr(self.model, 'named_steps') and 'estimator' in self.model.named_steps:
                        current_estimator = self.model.named_steps['estimator']
                    train_loss = getattr(current_estimator, 'loss_', 0.0)
                except Exception:
                    train_loss = 0.0

                if self.regression:
                    train_metric = mean_squared_error(y, train_preds)
                else:
                    train_metric = accuracy_score(y, train_preds)

                print(f"[MLP Training] Epoch {epoch + 1}/{max_iter} - Loss: {float(train_loss):.6f}, {'MSE' if self.regression else 'Accuracy'}: {train_metric:.4f}")

                # Send progress update with regression flag
                progress_callback({
                    'epoch': epoch + 1,
                    'loss': float(train_loss),
                    'metric': float(train_metric),
                    'regression': self.regression
                })

                # Update final epoch after successful completion
                self.final_epoch = epoch + 1

                # Check for convergence (loss not improving)
                if epoch > start_epoch and abs(prev_loss - float(train_loss)) < tol:
                    print(f"[MLP Training] Converged at epoch {epoch + 1}")
                    break

                prev_loss = float(train_loss)

            print(f"[MLP Training] Training completed at epoch {self.final_epoch}")
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
            # Add imputers and encoding for categorical
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            base = LinearRegression(**(params or {}))
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'logistic_regression':
            if regression:
                raise ValueError("Logistic Regression model requires regression=False")
            base = LogisticRegression(**(params or {}))
            pre = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))],
                remainder='passthrough'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'decision_tree':
            if regression:
                # Add imputers and encoding for categorical
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = DecisionTreeRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = DecisionTreeClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'random_forest':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = RandomForestRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = RandomForestClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'bagging':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = BaggingRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = BaggingClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'boosting':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = GradientBoostingRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = AdaBoostClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'svm':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = SVR(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            # Ensure we don't pass duplicate 'probability' kwarg
            _params = dict(params or {})
            if 'probability' not in _params:
                _params['probability'] = True
            base = SVC(**_params)
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'mlp':
            # Convert hidden_layer_sizes from list to tuple for sklearn
            mlp_params = params.copy() if params else {}
            if 'hidden_layer_sizes' in mlp_params and isinstance(mlp_params['hidden_layer_sizes'], list):
                mlp_params['hidden_layer_sizes'] = tuple(mlp_params['hidden_layer_sizes'])
            if regression:
                # Wrap MLPRegressor in a preprocessing pipeline for robust handling of mixed types
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = MLPRegressor(**mlp_params)
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = MLPClassifier(**mlp_params)
            pre = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))],
                remainder='passthrough'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
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
        data_source = request.form.get('data_source')
        license_info = request.form.get('license_info')
        
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
        if not data_source or not data_source.strip():
            return jsonify({'error': 'Data source is required'}), 400
        if not license_info or not license_info.strip():
            return jsonify({'error': 'License information is required'}), 400
            
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
            ds = Dataset(
                name=filename,
                file_path=path,
                user_id=current_user.id,
                description=description,
                data_source=data_source.strip(),
                license_info=license_info.strip(),
                regression=regression,
                input_features=input_features,
                target_feature=target_feature
            )
            db.session.add(ds)
            db.session.commit()
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            # Clean up the file if database operation fails
            if os.path.exists(path):
                os.remove(path)
            return jsonify({'error': f'Failed to save dataset to database: {str(e)}'}), 500
            
        try:
            # Read dataset info (robust encoding for CSV/TXT)
            df = read_dataset_file(path)

            file_size = os.path.getsize(path)
            rows, features = df.shape
            # Schema/size checks -> warnings (non-fatal)
            schema_warnings = []
            if features < 2:
                schema_warnings.append('Dataset has fewer than 2 columns; models require features and a target.')
            if rows == 0:
                schema_warnings.append('Dataset appears to be empty after reading.')
            if rows > 1_000_000:
                schema_warnings.append('Dataset has over 1,000,000 rows and may impact performance.')
            if features > 200:
                schema_warnings.append('Dataset has over 200 features and may impact performance.')
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
                        'data_source': data_source,
                        'license_info': license_info,
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

        # Optionally compute class imbalance immediately for classification
        imbalance_info = None
        try:
            if not regression:
                if target_feature and target_feature in df.columns:
                    y_tmp = df[target_feature].dropna()
                else:
                    # Fallback: use last column as proxy target
                    y_tmp = df.iloc[:, -1].dropna()
                imbalance_info = _class_imbalance_info(y_tmp)
        except Exception as _:
            imbalance_info = None

        return jsonify({
            'msg': 'Uploaded successfully',
            'dataset': {
                'id': ds.id,
                'name': filename,
                'description': description,
                'data_source': data_source,
                'license_info': license_info,
                'ext' : original_ext,
                'file_path': path,
                'file_size': file_size,
                'rows': rows,
                'features': features,
                'regression': ds.regression,
                'input_features': input_features,
                'target_feature': target_feature,
                'upload_date': ds.created_at.isoformat(),
                'imbalance': imbalance_info,
                'schema_warnings': schema_warnings if 'schema_warnings' in locals() else [],
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
                    'data_source': getattr(ds, 'data_source', None),
                    'license_info': getattr(ds, 'license_info', None),
                    'file_path': ds.file_path,
                    'upload_date': upload_date,
                    'file_size': 0,
                    'rows': 0,
                    'features': 0,
                    'regression': ds.regression,
                    'input_features': ds.input_features if ds.input_features else "",
                    'target_feature': ds.target_feature,
                    'models': ModelEntry.query.filter_by(dataset_id=ds.id).count(),
                    'imbalance': None
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
                    df = read_dataset_file(ds.file_path)
                        
                    file_size = os.path.getsize(ds.file_path)
                    rows, features = df.shape

                    # Compute class imbalance for classification; if target unknown, use last column as proxy
                    try:
                        if not ds.regression:
                            if ds.target_feature and ds.target_feature in df.columns:
                                y_tmp = df[ds.target_feature].dropna()
                            else:
                                y_tmp = df.iloc[:, -1].dropna()
                            dataset_info['imbalance'] = _class_imbalance_info(y_tmp)
                    except Exception:
                        dataset_info['imbalance'] = None
                    
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
                    # Schema/size warnings
                    warnings_list = []
                    try:
                        if int(features) < 2:
                            warnings_list.append('Dataset has fewer than 2 columns; models require features and a target.')
                        if int(rows) == 0:
                            warnings_list.append('Dataset appears to be empty after reading.')
                        if int(rows) > 1_000_000:
                            warnings_list.append('Dataset has over 1,000,000 rows and may impact performance.')
                        if int(features) > 200:
                            warnings_list.append('Dataset has over 200 features and may impact performance.')
                    except Exception:
                        pass
                    if warnings_list:
                        dataset_info['schema_warnings'] = warnings_list
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
        'regression': ds.regression,
        'description': ds.description,
        'data_source': getattr(ds, 'data_source', None),
        'license_info': getattr(ds, 'license_info', None)
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
        df = read_dataset_file(ds.file_path)
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
@app.route('/api/datasets/<int:dataset_id>', methods=['DELETE', 'OPTIONS', 'GET', 'PATCH'])
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

    # PATCH request updates metadata (data_source and license_info)
    if request.method == 'PATCH':
        if not (hasattr(current_user, 'is_authenticated') and current_user.is_authenticated):
            return jsonify({'error': 'Not authenticated'}), 401
        
        try:
            ds = Dataset.query.get_or_404(dataset_id)
            if ds.user_id != current_user.id:
                return jsonify({'error': 'Forbidden'}), 403
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            data_source = data.get('data_source', '').strip()
            license_info = data.get('license_info', '').strip()
            
            if not data_source:
                return jsonify({'error': 'Data source is required'}), 400
            if not license_info:
                return jsonify({'error': 'License information is required'}), 400
            
            ds.data_source = data_source
            ds.license_info = license_info
            db.session.commit()
            
            return jsonify({
                'msg': 'Metadata updated successfully',
                'data_source': ds.data_source,
                'license_info': ds.license_info
            }), 200
        except Exception as e:
            print(f"Error updating dataset metadata: {str(e)}")
            return jsonify({'error': str(e)}), 500

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

        # Cascade delete: remove all models associated with this dataset (for this user)
        try:
            associated_models = ModelEntry.query.filter_by(dataset_id=ds.id, user_id=current_user.id).all()
            for m in associated_models:
                db.session.delete(m)
        except Exception as cascade_err:
            print(f"Error deleting associated models for dataset {dataset_id}: {cascade_err}")

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
    # Centralized, encoding-robust read
    df = read_dataset_file(ds.file_path)
    # Coerce numeric-like strings (currency, percents, commas) to numeric
    df = coerce_numeric_like(df)
    # Select features/target
    if ds.target_feature and ds.target_feature in df.columns:
        y = df[ds.target_feature]
        X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
    else:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Drop rows with missing target only; let pipelines impute X
    if hasattr(y, 'notna'):
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
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
        print(f"[DEBUG] predict_proba returned shape: {proba.shape}, ndim: {proba.ndim}")
        # For binary, return just positive class probabilities
        if proba.ndim == 2 and proba.shape[1] == 2:
            print(f"[DEBUG] Binary classification detected, returning column 1")
            return proba[:, 1]
        # For multiclass, return all class probabilities
        print(f"[DEBUG] Multiclass detected, returning full proba array")
        return proba
    except Exception as e:
        print(f"[DEBUG] predict_proba failed: {e}")
        try:
            df = model.decision_function(X)
            print(f"[DEBUG] decision_function returned, shape: {df.shape if hasattr(df, 'shape') else 'scalar'}")
            return df
        except Exception as e2:
            print(f"[DEBUG] decision_function also failed: {e2}")
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
    print(f"[DEBUG _classification_metrics] y_score is None: {y_score is None}")
    if y_score is not None:
        print(f"[DEBUG _classification_metrics] y_score shape: {y_score.shape if hasattr(y_score, 'shape') else 'scalar'}, ndim: {y_score.ndim if hasattr(y_score, 'ndim') else 'N/A'}")
    try:
        if y_score is not None:
            # If y_test has exactly 2 classes, compute ROC-AUC and curve
            unique = np.unique(y_test)
            print(f"[DEBUG _classification_metrics] unique classes in y_test: {unique}, count: {unique.shape[0]}")
            if unique.shape[0] == 2:
                print(f"[DEBUG _classification_metrics] Binary classification - computing ROC curve")
                # Convert y to {0,1}
                y_bin = (y_test == unique.max()).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_score)
                metrics['roc_auc'] = float(roc_auc_score(y_bin, y_score))
                metrics['roc_curve'] = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)]
                print(f"[DEBUG _classification_metrics] ROC curve computed with {len(metrics['roc_curve'])} points, AUC: {metrics['roc_auc']}")
                # PR curve and AUC
                prec, rec, _ = precision_recall_curve(y_bin, y_score)
                pr_points = [{'precision': float(p), 'recall': float(r)} for p, r in zip(prec, rec)]
                metrics['pr_curve'] = pr_points
                metrics['pr_auc'] = float(average_precision_score(y_bin, y_score))
                print(f"[DEBUG _classification_metrics] PR curve computed with {len(pr_points)} points, AUC: {metrics['pr_auc']}")
            else:
                print(f"[DEBUG _classification_metrics] Multiclass classification - computing ROC AUC only")
                # multiclass ROC-AUC (no curve)
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_score, multi_class='ovr'))
                print(f"[DEBUG _classification_metrics] Multiclass ROC AUC: {metrics['roc_auc']}")
    except Exception as e:
        print(f"[DEBUG _classification_metrics] Exception during ROC computation: {e}")
        pass
    return metrics

def _regression_metrics(wrapper: ModelWrapper, X_test, y_test):
    preds = wrapper.predict(X_test)
    mse_val = float(mean_squared_error(y_test, preds))
    rmse_val = float(np.sqrt(mse_val))
    return {
        'mse': mse_val,
        'rmse': rmse_val,
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
        # Determine task type from associated dataset
        ds = None
        try:
            ds = Dataset.query.get(m.dataset_id) if m.dataset_id else None
        except Exception:
            ds = None
        task_type = 'regression' if (ds and getattr(ds, 'regression', False)) else 'classification'

        # Ensure essential scalar metrics exist; if missing, compute quickly and persist
        try:
            metrics_dict = json.loads(m.metrics) if m.metrics else {}
            need_cls = (task_type == 'classification') and not all(k in metrics_dict for k in ['accuracy','precision','recall','f1'])
            need_reg = (task_type == 'regression') and not all(k in metrics_dict for k in ['mse','mae','r2'])
            if (need_cls or need_reg) and ds is not None:
                # Compute lightweight evaluation on current stored model
                wrapper = ModelWrapper.from_db_record(m)
                try:
                    _, X, y, X_train, X_test, y_train, y_test = _load_dataset_for_model(m)
                except Exception:
                    # Fallback: load via dataset id
                    ds2, X, y, X_train, X_test, y_train, y_test = _load_dataset_for_model(m)
                if task_type == 'classification':
                    fresh = _classification_metrics(wrapper, X_test, y_test)
                else:
                    fresh = _regression_metrics(wrapper, X_test, y_test)
                # Merge scalar values only
                for k, v in fresh.items():
                    if isinstance(v, (int, float)):
                        metrics_dict[k] = v
                m.metrics = json.dumps(metrics_dict)
                db.session.merge(m)
                db.session.commit()
        except Exception:
            pass
        out.append({
            'id': m.id,
            'name': m.name,
            'model_type': m.model_type,
            'dataset_id': m.dataset_id,
            'metrics': json.loads(m.metrics) if m.metrics else {},
            'type': task_type,
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
            'metrics': {**(json.loads(entry.metrics) if entry.metrics else {}), **result_metrics},
            'cv': cv
        }

    data = {
        'model1': build_result(m1, w1),
        'model2': build_result(m2, w2),
    }
    # Persist scalar metrics and CV summaries back to each model for leaderboard visibility
    try:
        # Model 1
        existing_metrics_1 = json.loads(m1.metrics) if m1.metrics else {}
        scalar_metrics_1 = {k: v for k, v in data['model1']['metrics'].items() if isinstance(v, (int, float))}
        existing_metrics_1.update(scalar_metrics_1)
        if data['model1']['cv']:
            for k, v in data['model1']['cv'].items():
                potential_key = k.replace('mean', '').replace('_', " ").strip()
                if potential_key not in existing_metrics_1:
                    existing_metrics_1[potential_key] = v
        m1.metrics = json.dumps(existing_metrics_1)
        db.session.merge(m1)
        # Model 2
        existing_metrics_2 = json.loads(m2.metrics) if m2.metrics else {}
        scalar_metrics_2 = {k: v for k, v in data['model2']['metrics'].items() if isinstance(v, (int, float))}
        existing_metrics_2.update(scalar_metrics_2)
        if data['model2']['cv']:
            for k, v in data['model2']['cv'].items():
                potential_key = k.replace('mean', '').replace('_', " ").strip()
                if potential_key not in existing_metrics_2:
                    existing_metrics_2[potential_key] = v
        m2.metrics = json.dumps(existing_metrics_2)
        db.session.merge(m2)
        db.session.commit()
    except Exception:
        pass
        
    return jsonify(data)

# Simple in-memory store for SHAP async jobs
from threading import Thread, Lock
shap_jobs = {}
shap_jobs_lock = Lock()

# ===== Experiments / Evaluation endpoint =====
@app.route('/api/models/<int:model_id>/experiments', methods=['GET'])
@login_required
def model_experiments(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    try:
        # Prepare model and data
        wrapper = ModelWrapper.from_db_record(m)
        ds, X, y, X_train, X_test, y_train, y_test = _load_dataset_for_model(m)

        # Guard: model not trained / missing
        if wrapper.model is None:
            return jsonify({'error': "Model doesn't exist or has not trained"}), 400
        # Quick sanity check: try a tiny predict to catch NotFittedError early
        try:
            sample_X = X_test.iloc[:1] if len(X_test) > 0 else (X_train.iloc[:1] if len(X_train) > 0 else None)
            if sample_X is not None and len(sample_X) > 0:
                _ = wrapper.predict(sample_X)
        except NotFittedError:
            return jsonify({'error': "Model doesn't exist or has not trained"}), 400
        except ValueError as e:
            if 'No model loaded' in str(e):
                return jsonify({'error': "Model doesn't exist or has not trained"}), 400
            # else, allow other ValueErrors to be handled by existing logic later

        result = {
            'model_id': m.id,
            'model_name': m.name,
            'model_type': m.model_type,
            'type': 'regression' if ds.regression else 'classification',
            'metrics': {},
        }

        # Add preprocessing summary here (moved from compare endpoint)
        try:
            df_full = read_dataset_file(ds.file_path)
            original_rows = int(df_full.shape[0])
            rows_after_missing = int(df_full.dropna().shape[0])
            missing_removed = original_rows - rows_after_missing
            rows_after_duplicates = int(df_full.dropna().drop_duplicates().shape[0])
            duplicates_removed = rows_after_missing - rows_after_duplicates
            test_pct = float(ds.train_test_split) if ds.train_test_split is not None else 20.0
            result['preprocessing'] = {
                'original_rows': original_rows,
                'missing_values_removed': missing_removed,
                'duplicates_removed': duplicates_removed,
                'final_rows': rows_after_duplicates,
                'test_split_percentage': test_pct,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(X.columns),
                'input_features': ds.input_features,
                'target_feature': ds.target_feature,
            }
        except Exception:
            pass

        if ds.regression:
            # Regression metrics
            reg_metrics = _regression_metrics(wrapper, X_test, y_test)
            result['metrics'].update(reg_metrics)
        else:
            # Classification metrics + curves
            cls_metrics = _classification_metrics(wrapper, X_test, y_test)
            result['metrics'].update(cls_metrics)

            # Confusion matrix
            try:
                preds = wrapper.predict(X_test)
                labels = sorted(list(map(lambda v: v.item() if hasattr(v, 'item') else v, np.unique(y_test))))
                cm = confusion_matrix(y_test, preds, labels=labels)
                result['confusion_matrix'] = {
                    'labels': [str(l) for l in labels],
                    'matrix': cm.astype(int).tolist()
                }
            except Exception:
                pass

            # Class imbalance info for PR curve visibility guidance
            imb = _class_imbalance_info(y_test)
            if imb:
                result['imbalance'] = imb

            # Class distribution (overall)
            try:
                values, counts = np.unique(y, return_counts=True)
                total = counts.sum() if counts.sum() > 0 else 1
                result['class_distribution'] = [
                    {'label': str(v), 'count': int(c), 'percentage': float(c) * 100.0 / float(total)}
                    for v, c in zip(values, counts)
                ]
            except Exception:
                pass

            # Multiclass ROC curves (one-vs-rest) if applicable
            try:
                # Attempt to get score/probabilities
                y_score = _get_proba_or_score(wrapper.model, X_test)
                unique = np.unique(y_test)
                print(f"[DEBUG multiclass ROC] y_score is None: {y_score is None}, unique classes: {unique.shape[0]}")
                if y_score is not None:
                    print(f"[DEBUG multiclass ROC] y_score shape: {y_score.shape if hasattr(y_score, 'shape') else 'scalar'}, ndim: {y_score.ndim if hasattr(y_score, 'ndim') else 'N/A'}")
                if y_score is not None and unique.shape[0] > 2:
                    print(f"[DEBUG multiclass ROC] Proceeding with multiclass ROC curves (>2 classes)")
                    # ensure 2D scores: shape (n_samples, n_classes)
                    scores = y_score
                    if scores.ndim == 1:
                        print(f"[DEBUG multiclass ROC] scores is 1D, cannot plot multiclass ROC")
                        # cannot plot multiclass ROC with 1D score
                        scores = None
                    if scores is not None:
                        print(f"[DEBUG multiclass ROC] scores is 2D with shape {scores.shape}")
                        try:
                            # Try classes_ from the model/pipeline, else fallback to sorted unique labels
                            classes = getattr(wrapper.model, 'classes_', None)
                            if classes is None:
                                classes = sorted(list(unique))
                            print(f"[DEBUG multiclass ROC] classes: {classes}")
                            # Binarize labels per class and compute curve
                            roc_curves_ovr = []
                            roc_auc_ovr = []
                            pr_curves_ovr = []
                            pr_auc_ovr = []
                            # Map class label order to score columns if possible
                            # If classes is an array, assume scores columns align to that order
                            for idx, cls in enumerate(classes):
                                try:
                                    # binarize y_test for current class
                                    y_bin = (y_test == cls).astype(int)
                                    # pick column idx if available, else attempt to find matching column
                                    if scores.shape[1] > idx:
                                        s = scores[:, idx]
                                    else:
                                        # fallback: use decision_function result if shape mismatch
                                        s = scores[:, 0]
                                    fpr, tpr, _ = roc_curve(y_bin, s)
                                    roc_curves_ovr.append({
                                        'class_label': str(cls),
                                        'curve': [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)]
                                    })
                                    roc_auc_ovr.append({'class_label': str(cls), 'auc': float(roc_auc_score(y_bin, s))})
                                    # PR curve (OvR)
                                    prec, rec, _ = precision_recall_curve(y_bin, s)
                                    pr_curves_ovr.append({
                                        'class_label': str(cls),
                                        'curve': [{'recall': float(r), 'precision': float(p)} for p, r in zip(prec, rec)]
                                    })
                                    pr_auc_ovr.append({'class_label': str(cls), 'ap': float(average_precision_score(y_bin, s))})
                                    print(f"[DEBUG multiclass ROC] Class {cls} ROC curve computed with {len(roc_curves_ovr[-1]['curve'])} points")
                                except Exception as e:
                                    print(f"[DEBUG multiclass ROC] Failed to compute ROC for class {cls}: {e}")
                                    continue
                            if roc_curves_ovr:
                                result['metrics']['roc_curves_ovr'] = roc_curves_ovr
                                result['metrics']['roc_auc_ovr'] = roc_auc_ovr
                                result['metrics']['pr_curves_ovr'] = pr_curves_ovr
                                result['metrics']['pr_auc_ovr'] = pr_auc_ovr
                                print(f"[DEBUG multiclass ROC] Added {len(roc_curves_ovr)} ROC curves to result")
                            else:
                                print(f"[DEBUG multiclass ROC] No ROC curves computed")
                        except Exception as e:
                            print(f"[DEBUG multiclass ROC] Exception in inner try: {e}")
                            pass
                else:
                    print(f"[DEBUG multiclass ROC] Skipping multiclass ROC (not >2 classes or y_score is None)")
            except Exception as e:
                print(f"[DEBUG multiclass ROC] Exception in outer try: {e}")
                pass

        # SHAP: optionally compute asynchronously so the response is fast and UI can render while SHAP is computed
        shap_async = request.args.get('shap_async', '1') != '0'
        if shap_async:
            # enqueue job and return pending
            try:
                key = f"{current_user.id}:{m.id}"
                with shap_jobs_lock:
                    job = shap_jobs.get(key)
                    if not job or job.get('status') in ('done','error'):
                        shap_jobs[key] = {'status': 'queued', 'progress': 0, 'result': None, 'error': None}

                def run_shap_job():
                    try:
                        import shap  # type: ignore
                        import pandas as pd
                        with shap_jobs_lock:
                            if key in shap_jobs:
                                shap_jobs[key]['status'] = 'running'
                                shap_jobs[key]['progress'] = 5

                        # samples
                        bg_sample = X_train.sample(min(SHAP_BG_N, len(X_train)), random_state=42) if len(X_train) > 0 else X_test
                        eval_sample = X_test.sample(min(SHAP_EVAL_N, len(X_test)), random_state=42) if len(X_test) > 0 else X_train
                        model = wrapper.model
                        feature_names = list(bg_sample.columns)

                        def to_df(Xk):
                            try:
                                if isinstance(Xk, pd.DataFrame):
                                    if list(Xk.columns) != feature_names:
                                        try:
                                            return Xk[feature_names]
                                        except Exception:
                                            return Xk
                                    return Xk
                                return pd.DataFrame(Xk, columns=feature_names)
                            except Exception:
                                return pd.DataFrame(Xk)

                        if hasattr(model, 'predict_proba'):
                            f = lambda Xk: model.predict_proba(to_df(Xk))
                        else:
                            f = lambda Xk: model.predict(to_df(Xk))

                        # Choose explainer: optional faster SamplingExplainer, else KernelExplainer
                        explainer = shap.SamplingExplainer(f, bg_sample) if SHAP_USE_SAMPLING else shap.KernelExplainer(f, bg_sample)
                        with shap_jobs_lock:
                            if key in shap_jobs:
                                    shap_jobs[key]['progress'] = 10

                        # compute in smaller chunks to provide smoother progress updates
                        n_chunks = SHAP_N_CHUNKS
                        chunks = np.array_split(np.arange(len(eval_sample)), n_chunks) if len(eval_sample) > 0 else []
                        mean_abs_accum = np.zeros(len(eval_sample.columns)) if len(eval_sample.columns) > 0 else None
                        total_samples = 0
                        for i, idxs in enumerate(chunks):
                            if len(idxs) == 0:
                                continue
                            part = eval_sample.iloc[idxs]
                            shap_vals = explainer.shap_values(part, nsamples=SHAP_NSAMPLES)
                            
                            # Handle multi-class (list of arrays) vs binary/regression (single array)
                            if isinstance(shap_vals, list):
                                # For multi-class: average absolute SHAP values across all classes
                                # Each element in shap_vals has shape (n_samples, n_features)
                                print(f"[SHAP DEBUG] Multi-class detected. Number of classes: {len(shap_vals)}")
                                print(f"[SHAP DEBUG] Shape of each class: {[sv.shape for sv in shap_vals]}")
                                abs_vals = [np.abs(sv) for sv in shap_vals]
                                arr = np.mean(abs_vals, axis=0)  # Average across classes -> (n_samples, n_features)
                                print(f"[SHAP DEBUG] After averaging: {arr.shape}")
                            else:
                                # For binary/regression: use absolute values directly
                                print(f"[SHAP DEBUG] Binary/regression. SHAP values shape: {shap_vals.shape}")
                                arr = np.abs(shap_vals)
                                print(f"[SHAP DEBUG] After abs: {arr.shape}")
                                
                                # If 3D (n_samples, n_features, n_classes), average across classes
                                if arr.ndim == 3:
                                    print(f"[SHAP DEBUG] Detected 3D array (binary classification with probabilities), averaging across classes")
                                    arr = arr.mean(axis=2)  # Average across classes -> (n_samples, n_features)
                                    print(f"[SHAP DEBUG] After averaging classes: {arr.shape}")
                            
                            # Ensure arr is 2D: (n_samples, n_features)
                            if arr.ndim == 1:
                                print(f"[SHAP DEBUG] Reshaping 1D array {arr.shape} to 2D")
                                arr = arr.reshape(-1, 1)
                            
                            print(f"[SHAP DEBUG] Final arr shape: {arr.shape}")                            # accumulate sums to compute mean later
                            chunk_sum = arr.sum(axis=0)  # Sum across samples -> (n_features,)
                            print(f"[SHAP DEBUG] chunk_sum shape: {chunk_sum.shape}")
                            
                            if mean_abs_accum is None:
                                print(f"[SHAP DEBUG] Initializing mean_abs_accum with shape: {chunk_sum.shape}")
                                mean_abs_accum = chunk_sum
                            else:
                                print(f"[SHAP DEBUG] Adding chunk_sum {chunk_sum.shape} to mean_abs_accum {mean_abs_accum.shape}")
                                mean_abs_accum += chunk_sum
                            total_samples += arr.shape[0]
                            with shap_jobs_lock:
                                if key in shap_jobs:
                                        shap_jobs[key]['progress'] = 10 + int((i+1) / max(1, n_chunks) * 90)
                        
                        if mean_abs_accum is None or total_samples == 0:
                            pairs = []
                        else:
                            mean_abs = mean_abs_accum / float(total_samples)
                            features = list(eval_sample.columns)
                            pairs = sorted([{'feature': f, 'importance': float(v)} for f, v in zip(features, mean_abs)], key=lambda x: x['importance'], reverse=True)[:20]

                        with shap_jobs_lock:
                            if key in shap_jobs:
                                shap_jobs[key]['status'] = 'done'
                                shap_jobs[key]['result'] = {'feature_importance': pairs, 'total_features': len(eval_sample.columns)}
                                shap_jobs[key]['progress'] = 100
                    except Exception as e:
                        with shap_jobs_lock:
                            if key in shap_jobs:
                                shap_jobs[key]['status'] = 'error'
                                shap_jobs[key]['error'] = str(e)
                                shap_jobs[key]['progress'] = 100

                # start background thread
                with shap_jobs_lock:
                    if shap_jobs[key]['status'] == 'queued':
                        t = Thread(target=run_shap_job, daemon=True)
                        shap_jobs[key]['thread'] = t
                        t.start()

                result['shap'] = {'feature_importance': [], 'status': 'pending'}
            except Exception as e:
                result['shap'] = {'feature_importance': [], 'error': str(e)}
        else:
            # synchronous SHAP (fallback)
            try:
                import shap  # type: ignore
                import pandas as pd  # ensure we can construct DataFrames inside SHAP wrapper
                bg_sample = X_train.sample(min(SHAP_BG_N, len(X_train)), random_state=42) if len(X_train) > 0 else X_test
                eval_sample = X_test.sample(min(SHAP_EVAL_N, len(X_test)), random_state=42) if len(X_test) > 0 else X_train
                model = wrapper.model
                feature_names = list(bg_sample.columns)
                def to_df(Xk):
                    try:
                        if isinstance(Xk, pd.DataFrame):
                            if list(Xk.columns) != feature_names:
                                try:
                                    return Xk[feature_names]
                                except Exception:
                                    return Xk
                            return Xk
                        return pd.DataFrame(Xk, columns=feature_names)
                    except Exception:
                        return pd.DataFrame(Xk)
                if hasattr(model, 'predict_proba'):
                    f = lambda Xk: model.predict_proba(to_df(Xk))
                else:
                    f = lambda Xk: model.predict(to_df(Xk))
                # Choose explainer: optional faster SamplingExplainer, else KernelExplainer
                explainer = shap.SamplingExplainer(f, bg_sample) if SHAP_USE_SAMPLING else shap.KernelExplainer(f, bg_sample)
                shap_vals = explainer.shap_values(eval_sample, nsamples=SHAP_NSAMPLES)
                
                # Handle multi-class (list of arrays) vs binary/regression (single array)
                if isinstance(shap_vals, list):
                    # For multi-class: average absolute SHAP values across all classes
                    abs_vals = [np.abs(sv) for sv in shap_vals]
                    arr = np.mean(abs_vals, axis=0)  # Average across classes -> (n_samples, n_features)
                    mean_abs = arr.mean(axis=0)  # Mean across samples -> (n_features,)
                else:
                    # For binary/regression: compute mean of absolute values
                    arr = np.abs(shap_vals)
                    # If 3D (n_samples, n_features, n_classes), average across classes
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2)  # Average across classes -> (n_samples, n_features)
                    # Ensure 2D for consistent handling
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    mean_abs = arr.mean(axis=0)
                
                features = list(eval_sample.columns)
                pairs = sorted([{'feature': f, 'importance': float(v)} for f, v in zip(features, mean_abs)], key=lambda x: x['importance'], reverse=True)
                result['shap'] = {'feature_importance': pairs[:20], 'total_features': len(features)}
            except Exception as e:
                    result['shap'] = {'feature_importance': [], 'error': str(e)}

        # Persist scalar evaluation metrics back to the model so they're available in compare/leaderboard
        try:
            scalar_metrics = {k: v for k, v in result.get('metrics', {}).items() if isinstance(v, (int, float))}
            if scalar_metrics:
                existing = json.loads(m.metrics) if m.metrics else {}
                existing.update(scalar_metrics)
                m.metrics = json.dumps(existing)
                db.session.merge(m)
                db.session.commit()
        except Exception:
            # Do not block the response if persisting fails
            pass

        return jsonify(result)
    except Exception as e:
        print(f"Error in model_experiments: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint to poll SHAP job status/result
@app.route('/api/models/<int:model_id>/experiments/shap', methods=['GET'])
@login_required
def model_experiments_shap(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    key = f"{current_user.id}:{m.id}"
    with shap_jobs_lock:
        job = shap_jobs.get(key)
        if not job:
            return jsonify({'status': 'none', 'progress': 0})
        out = {'status': job.get('status'), 'progress': job.get('progress', 0)}
        if job.get('status') == 'done' and job.get('result') is not None:
            out['shap'] = job['result']
        if job.get('status') == 'error':
            out['error'] = job.get('error')
        return jsonify(out)

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
            'metrics': json.loads(m.metrics) if m.metrics else {},
            'early_stopped': m.early_stopped if hasattr(m, 'early_stopped') else False,
            'current_epoch': m.current_epoch if hasattr(m, 'current_epoch') else None
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
    # Cascade delete: remove any experiments associated with this model if stored separately in future
    # Currently, experiments are derived/evaluated on demand and stored in model.metrics/history on frontend.
    # If a server-side Experiment table is added, delete rows here filtering by model_id and user.
    db.session.delete(m)
    db.session.commit()
    return jsonify({'msg': 'Model deleted'})

def _read_text_table_with_encoding(file_path: str, sep: str = ','):
    """Read a text table (CSV/TXT) with resilient encoding handling.
    Tries utf-8, then chardet-detected encoding, then latin-1 as a last resort.
    """
    # Try UTF-8 first
    try:
        return pd.read_csv(file_path, sep=sep, encoding='utf-8')
    except UnicodeDecodeError:
        pass
    # Try chardet detection if available
    try:
        import chardet  # type: ignore
        with open(file_path, 'rb') as f:
            raw = f.read(64 * 1024)
        guess = chardet.detect(raw)
        enc = guess.get('encoding') or 'latin-1'
        return pd.read_csv(file_path, sep=sep, encoding=enc)
    except Exception:
        # Last resort
        return pd.read_csv(file_path, sep=sep, encoding='latin-1', engine='python')

def read_dataset_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            return _read_text_table_with_encoding(file_path, ',')
        elif ext == '.txt':
            return _read_text_table_with_encoding(file_path, '\t')
        elif ext == '.xlsx':
            import openpyxl
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f'Unsupported file format: {ext}')
    except Exception as e:
        raise ValueError(f'Failed to read dataset: {str(e)}')

def coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort conversion of numeric-looking string columns to numbers.
    - Strips currency symbols ($), thousand separators (comma), percent signs
    - Converts parentheses to negatives, e.g., (123) -> -123
    - Converts to float when at least ~60% of non-empty values are numeric-like
    - If majority had percent signs, divides by 100
    """
    try:
        out = df.copy()
        for col in out.columns:
            s = out[col]
            # Only attempt on object/string-like columns
            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                continue
            try:
                s_str = s.astype(str).str.strip()
                if s_str.empty:
                    continue
                pct_mask = s_str.str.contains('%', regex=False, na=False)
                # Normalize: remove $ and commas, remove %, convert (x) to -x
                cleaned = s_str.str.replace(r'[\$,]', '', regex=True)
                cleaned = cleaned.str.replace('%', '', regex=False)
                cleaned = cleaned.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
                # Try conversion
                as_num = pd.to_numeric(cleaned, errors='coerce')
                # Heuristic: convert if at least 60% become numbers (ignoring NaNs)
                conv_ratio = as_num.notna().mean()
                if conv_ratio >= 0.60:
                    # If majority values had percent, scale down
                    if pct_mask.mean() >= 0.60:
                        as_num = as_num / 100.0
                    out[col] = as_num
            except Exception:
                # Best-effort; leave column as-is on failure
                continue
        return out
    except Exception:
        return df
    
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
        df = read_dataset_file(ds.file_path)
        df = coerce_numeric_like(df)
    except Exception as e:
        print(f"Error reading dataset for training: {str(e)}")
        return jsonify({'error': f'Could not read dataset: {e}'}), 500

    if df.shape[1] < 2:
        print(f"Error: Dataset must have at least 2 columns (features + target)")
        return jsonify({'error': 'Dataset must have at least 2 columns (features + target)'}), 400
    
    if ds.target_feature and ds.target_feature in df.columns:
        # Use the configured target feature
        y = df[ds.target_feature]
        X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
    else:
        # Fallback to last column as target
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Drop rows with missing targets only; let imputers handle feature NaNs
    if hasattr(y, 'notna'):
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    
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
        
        print(f"[WebSocket] Starting training for model_id: {model_id}")
        
        # Verify user is authenticated (SocketIO session)
        # Note: For production, implement proper auth via session or token
        
        m_entry = ModelEntry.query.get(model_id)
        if not m_entry:
            print(f"[WebSocket] Error: Model {model_id} not found")
            emit('training_error', {'message': 'Model not found'})
            return
        
        print(f"[WebSocket] Found model: {m_entry.name} (type: {m_entry.model_type})")
        
        # Initialize pause state for this model
        # If resuming early-stopped model, always start in paused state
        if m_entry.early_stopped:
            training_paused[model_id] = True  # Start paused when resuming early-stopped model
            epoch_info = f" from epoch {m_entry.current_epoch}" if m_entry.current_epoch is not None else ""
            print(f"[WebSocket] Initializing paused state for early-stopped model resume{epoch_info}")
            # Notify frontend immediately
            emit('training_paused', {'model_id': model_id})
        else:
            training_paused[model_id] = False
        training_early_stopped[model_id] = False
        
        ds = Dataset.query.get(m_entry.dataset_id)
        if not ds:
            print("[WebSocket] Error: Dataset not found")
            emit('training_error', {'message': 'Dataset not found'})
            return
        
        print(f"[WebSocket] Found dataset: {ds.name}")
        
        # Read dataset (centralized, robust) and coerce numeric-like strings
        try:
            df = read_dataset_file(ds.file_path)
            df = coerce_numeric_like(df)
        except Exception as e:
            print(f"[WebSocket] Error reading dataset: {e}")
            emit('training_error', {'message': f'Could not read dataset: {str(e)}'})
            return
        
        print(f"[WebSocket] Loaded dataset with shape: {df.shape}")
        
        if ds.target_feature and ds.target_feature in df.columns:
            y = df[ds.target_feature]
            X = df[ds.input_features.split(',')] if ds.input_features else df.drop(columns=[ds.target_feature])
        else:
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
        # Drop rows with missing target; keep feature NaNs for imputers
        if hasattr(y, 'notna'):
            mask = y.notna()
            X = X.loc[mask]
            y = y.loc[mask]
        
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
        
        # Define early stop check function
        def early_stop_check():
            return training_early_stopped.get(model_id, False)
        
        print("[WebSocket] Starting model training...")
        # Train with streaming
        wrapper = ModelWrapper.from_db_record(m_entry)
        
        # For early-stopped models, start epoch numbering from 0 (not resuming from saved epoch)
        # The model weights are preserved, but epoch counter resets
        start_epoch = 0
        keep_model = m_entry.early_stopped  # Keep model weights if this was early stopped
        print(f"[WebSocket] Starting training from epoch {start_epoch}, keep_model={keep_model}")
        
        wrapper.train(X_train, y_train, progress_callback=progress_callback, pause_check=pause_check, early_stop_check=early_stop_check, start_epoch=start_epoch, keep_model=keep_model)
        
        print("[WebSocket] Training complete, evaluating...")
        # Evaluate and save
        final_metrics = wrapper.evaluate(X_test, y_test)
        entry = wrapper.to_db_record(name=m_entry.name, dataset_id=ds.id, user_id=m_entry.user_id)
        entry.id = m_entry.id
        entry.metrics = json.dumps(final_metrics)
        # Mark if this was early stopped (or clear flag if training completed normally)
        was_early_stopped = training_early_stopped.get(model_id, False)
        entry.early_stopped = was_early_stopped
        # Save current epoch if early stopped, otherwise clear it
        if was_early_stopped:
            entry.current_epoch = wrapper.final_epoch if hasattr(wrapper, 'final_epoch') else None
            print(f"[WebSocket] Early stopped at epoch {entry.current_epoch}")
        else:
            entry.current_epoch = None  # Clear epoch on successful completion
        db.session.merge(entry)
        db.session.commit()
        
        print(f"[WebSocket] Emitting training_complete with metrics: {final_metrics}, early_stopped: {was_early_stopped}")
        socketio.emit('training_complete', {
            'type': 'complete',
            'message': 'Training completed successfully',
            'metrics': final_metrics,
            'early_stopped': was_early_stopped
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
        if model_id in training_early_stopped:
            del training_early_stopped[model_id]

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

@socketio.on('early_stop_training')
def handle_early_stop(data):
    """Early stop training for a specific model - saves current progress"""
    model_id = data.get('model_id')
    if model_id:
        training_early_stopped[model_id] = True
        emit('training_early_stopped', {'model_id': model_id})
        print(f"[WebSocket] Early stop requested for model {model_id}")

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
    # Build a DataFrame with proper column names so Pipelines with encoders work
    ds = Dataset.query.get_or_404(m.dataset_id)
    rows = data['input']
    # Determine column names
    if ds.input_features:
        cols = [c.strip() for c in ds.input_features.split(',') if c.strip()]
    else:
        # Fallback: try to read dataset columns (excluding target)
        try:
            df_full = read_dataset_file(ds.file_path)
            if ds.target_feature and ds.target_feature in df_full.columns:
                cols = [c for c in df_full.columns if c != ds.target_feature]
            else:
                cols = list(df_full.columns[:-1])
        except Exception:
            # Last resort: generic column names
            cols = [f'f{i}' for i in range(len(rows[0]) if rows else 0)]
    X_in = pd.DataFrame(rows, columns=cols)
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
        # Best-effort migration: add data_source and license_info to dataset if missing
        try:
            with db.engine.connect() as conn:
                # Add data_source
                try:
                    conn.execute(db.text('ALTER TABLE dataset ADD COLUMN data_source TEXT'))
                    conn.commit()
                    print("Migrated: added dataset.data_source column")
                except Exception as e:
                    # Column may already exist
                    pass
                # Add license_info
                try:
                    conn.execute(db.text('ALTER TABLE dataset ADD COLUMN license_info TEXT'))
                    conn.commit()
                    print("Migrated: added dataset.license_info column")
                except Exception as e:
                    # Column may already exist
                    pass
        except Exception as e:
            print(f"Dataset metadata migration skipped/failed: {e}")
        
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

@app.route('/api/migrate/add-early-stopped', methods=['POST'])
def migrate_add_early_stopped():
    key = request.args.get('key')
    if key != 'supersecretresetkey':
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Add early_stopped column if it doesn't exist
        with db.engine.connect() as conn:
            conn.execute(db.text('ALTER TABLE model_entry ADD COLUMN early_stopped BOOLEAN DEFAULT 0'))
            conn.commit()
        return jsonify({'msg': 'Migration completed successfully - added early_stopped column'})
    except Exception as e:
        # Column might already exist
        return jsonify({'msg': f'Migration note: {str(e)}'})

@app.route('/api/migrate/add-current-epoch', methods=['POST'])
def migrate_add_current_epoch():
    key = request.args.get('key')
    if key != 'supersecretresetkey':
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Add current_epoch column if it doesn't exist
        with db.engine.connect() as conn:
            conn.execute(db.text('ALTER TABLE model_entry ADD COLUMN current_epoch INTEGER'))
            conn.commit()
        return jsonify({'msg': 'Migration completed successfully - added current_epoch column'})
    except Exception as e:
        # Column might already exist
        return jsonify({'msg': f'Migration note: {str(e)}'})

@app.route('/api/migrate/add-dataset-metadata', methods=['POST'])
def migrate_add_dataset_metadata():
    key = request.args.get('key')
    if key != 'supersecretresetkey':
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        with db.engine.connect() as conn:
            try:
                conn.execute(db.text('ALTER TABLE dataset ADD COLUMN data_source TEXT'))
            except Exception as e:
                pass
            try:
                conn.execute(db.text('ALTER TABLE dataset ADD COLUMN license_info TEXT'))
            except Exception as e:
                pass
            conn.commit()
        return jsonify({'msg': 'Migration completed successfully - added dataset metadata columns'})
    except Exception as e:
        return jsonify({'msg': f'Migration note: {str(e)}'})
    

# ===== Main =====
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)
