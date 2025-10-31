from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os, pandas as pd, pickle, io, datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

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
    target_feature = db.Column(db.String(100), nullable=True)
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
    params = db.Column(db.LargeBinary, nullable=True)
    metrics = db.Column(db.LargeBinary, nullable=True)
    model_blob = db.Column(db.LargeBinary, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)
    
# ===== Model wrapper/helper =====
class ModelWrapper:
    """
    Thin wrapper to standardize training/predicting and DB serialization.
    model_type: one of ('linear_regression','logistic_regression','decision_tree',
                       'bagging','boosting','random_forest','svm','mlp','custom')
    For 'custom' we accept an uploaded pickled model file which will be stored as-is.
    """
    def __init__(self, model_type, model=None, params=None):
        self.model_type = model_type
        self.model = model
        self.params = params or {}
        self.metrics = {}

    def train(self, X, y, **train_kwargs):
        # X,y are pandas or numpy-like
        if self.model is None:
            self.model = self._instantiate_from_type(self.model_type, self.params)

        # For scikit-learn style models
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
        # Simple heuristic: if target dtype is numeric and model_type suggests regression, use MSE
        if self.model_type in ['linear_regression']:
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
        params_blob = pickle.dumps(self.params)
        metrics_blob = pickle.dumps(self.metrics)
        return ModelEntry(
            name=name,
            model_type=self.model_type,
            params=params_blob,
            metrics=metrics_blob,
            model_blob=model_blob,
            dataset_id=dataset_id,
            user_id=user_id
        )

    @staticmethod
    def from_db_record(record: ModelEntry):
        params = pickle.loads(record.params) if record.params else {}
        metrics = pickle.loads(record.metrics) if record.metrics else {}
        model = pickle.loads(record.model_blob)
        wrapper = ModelWrapper(model_type=record.model_type, model=model, params=params)
        wrapper.metrics = metrics
        return wrapper

    @staticmethod
    def _instantiate_from_type(model_type, params):
        # Map types to sklearn classes
        if model_type == 'linear_regression':
            return LinearRegression(**(params or {}))
        if model_type == 'logistic_regression':
            return LogisticRegression(**(params or {}))
        if model_type == 'decision_tree':
            return DecisionTreeClassifier(**(params or {}))
        if model_type == 'random_forest':
            return RandomForestClassifier(**(params or {}))
        if model_type == 'bagging':
            return BaggingClassifier(**(params or {}))
        if model_type == 'boosting':
            # AdaBoost for classification
            return AdaBoostClassifier(**(params or {}))
        if model_type == 'svm':
            return SVC(**(params or {}))
        if model_type == 'mlp':
            return MLPClassifier(**(params or {}))
        # 'custom' must be provided as an already pickled/fitted model by user
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
        target_feature = request.form.get('target_feature')
        
        print(f"Received file: {file.filename if file else 'None'}")
        print(f"Custom name: {custom_name}")
        print(f"Description: {description}")
        print(f"Target feature: {target_feature}")

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
            ds = Dataset(name=filename, file_path=path, user_id=current_user.id, description=description, target_feature=target_feature)
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

# ...existing code...

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
    target_feature = data.get('target_feature')
    train_test_split = data.get('train_test_split')

    if not target_feature or not isinstance(train_test_split, (int, float)):
        return jsonify({'error': 'Invalid configuration data'}), 400

    try:
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
            'metrics': pickle.loads(m.metrics) if m.metrics else {}
        })
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

    try:
        wrapper = ModelWrapper(model_type=model_type, model=None, params=params)
        entry = wrapper.to_db_record(name=name, dataset_id=ds.id if ds else None, user_id=current_user.id)
        entry.description = description
        db.session.add(entry)
        db.session.commit()
        return jsonify({'msg': 'Model created', 'model_id': entry.id}), 201
    except Exception as e:
        return jsonify({'error': f'Failed to create model: {str(e)}'}), 500

@app.route('/api/models/<int:model_id>', methods=['GET'])
@login_required
def get_model(model_id):
    m = ModelEntry.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify({
        'id': m.id,
        'name': m.name,
        'description': m.description if hasattr(m, 'description') else 'No description',
        'model_type': m.model_type,
        'dataset_id': m.dataset_id,
        'created_at': m.created_at.isoformat() if m.created_at else None,
        'metrics': pickle.loads(m.metrics) if m.metrics else {}
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
            return jsonify({'error': 'Unsupported dataset format for training'}), 400
    except Exception as e:
        return jsonify({'error': f'Could not read dataset: {e}'}), 500

    if df.shape[1] < 2:
        return jsonify({'error': 'Dataset must have at least 2 columns (features + target)'}), 400

    if ds.target_feature and ds.target_feature in df.columns:
        # Use the configured target feature
        y = df[ds.target_feature]
        X = df.drop(columns=[ds.target_feature])
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
        # replace metrics blob with actual metrics (to ensure updated)
        entry.metrics = pickle.dumps(wrapper.metrics)
        db.session.add(entry)
        db.session.commit()
        return jsonify({
            'msg': 'Trained and saved model',
            'model_id': entry.id,
            'metrics': wrapper.metrics,
            'test_size_used': test_size  # Return the actual test size used
        }), 201
    except Exception as e:
        return jsonify({'error': f'Failed to train model: {str(e)}'}), 500

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
    app.run(port=5000, debug=True)
