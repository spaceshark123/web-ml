from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.config['SECRET_KEY'] = 'devsecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

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
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)

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
        
        print(f"Received file: {file.filename if file else 'None'}")
        print(f"Custom name: {custom_name}")
        
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
            ds = Dataset(name=filename, file_path=path, user_id=current_user.id)
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
                    'file_path': path,
                    'upload_date': ds.created_at.isoformat(),
                    'error': f'Could not read file contents: {str(e)}'
                }
            })

        return jsonify({
            'msg': 'Uploaded successfully',
            'dataset': {
                'id': ds.id,
                'name': filename,
                'ext' : original_ext,
                'file_path': path,
                'file_size': file_size,
                'rows': rows,
                'features': features,
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
        print(f"Headers: {dict(request.headers)}")
        
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
                    'file_path': ds.file_path,
                    'upload_date': upload_date,
                    'file_size': 0,
                    'rows': 0,
                    'features': 0,
                    'models': 0
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

@app.route('/api/train/<int:dataset_id>', methods=['POST'])
@login_required
def train(dataset_id):
    ds = Dataset.query.get_or_404(dataset_id)
    df = pd.read_csv(ds.file_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return jsonify({'accuracy': acc})

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

# ===== Main =====
if __name__ == '__main__':
    app.run(port=5000, debug=True)
