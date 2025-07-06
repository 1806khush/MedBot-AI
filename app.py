from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_cors import CORS
from document_processor import DocumentProcessor
from vector_db import VectorDB
from rag_handler import RAGHandler
import os
import uuid
from pathlib import Path
import logging
import concurrent.futures
import time
from functools import wraps
from datetime import datetime, timedelta
from database import db
from werkzeug.security import generate_password_hash, check_password_hash
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
Path("temp_files").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Initialize Flask app with session timeout
app = Flask(__name__)
try:
    flask_config = config.get_flask_config()
    app.secret_key = flask_config['secret_key']
    app.permanent_session_lifetime = timedelta(minutes=flask_config['session_timeout_minutes'])
except Exception as e:
    logger.error(f"Failed to load Flask configuration: {str(e)}")
    # Fallback values if config loading fails
    app.secret_key = 'fallback_secret_key_please_change_me'
    app.permanent_session_lifetime = timedelta(minutes=2)
CORS(app)

document_processor = DocumentProcessor()
vector_db = VectorDB()
rag_handler = RAGHandler()

# Thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Performance monitoring
performance_stats = {
    'upload_times': [],
    'query_times': [],
    'last_optimized': datetime.now()
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def monitor_performance(endpoint_name):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                # Store performance data
                if endpoint_name == 'upload':
                    performance_stats['upload_times'].append(duration)
                    if len(performance_stats['upload_times']) > 100:
                        performance_stats['upload_times'].pop(0)
                elif endpoint_name == 'query':
                    performance_stats['query_times'].append(duration)
                    if len(performance_stats['query_times']) > 100:
                        performance_stats['query_times'].pop(0)
                
                # Add performance header to response
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
                    response, status = result
                    response['performance'] = {'duration': round(duration, 3)}
                    return response, status
                elif isinstance(result, dict):
                    result['performance'] = {'duration': round(duration, 3)}
                    return result
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error in {endpoint_name}: {str(e)} (took {duration:.2f}s)")
                raise e
        return wrapped
    return decorator

def optimize_index_if_needed():
    """Check if index optimization is needed based on performance stats"""
    now = datetime.now()
    hours_since_last_optimization = (now - performance_stats['last_optimized']).total_seconds() / 3600
    
    # Optimize if it's been more than 24 hours or if query times are slowing down
    if hours_since_last_optimization > 24:
        logger.info("Performing scheduled index optimization")
        vector_db.optimize_index()
        performance_stats['last_optimized'] = now
    elif len(performance_stats['query_times']) > 20:
        avg_query_time = sum(performance_stats['query_times']) / len(performance_stats['query_times'])
        if avg_query_time > 2.0:  # If average query time exceeds 2 seconds
            logger.info(f"Performing performance-based index optimization (avg query time: {avg_query_time:.2f}s)")
            vector_db.optimize_index()
            performance_stats['last_optimized'] = now
            performance_stats['query_times'] = []  # Reset stats after optimization

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('main_app'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET'])
def login():
    if 'user_id' in session:
        return redirect(url_for('main_app'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def process_login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    user = db.get_user_by_email(email)
    if not user or not check_password_hash(user['Password'], password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    # Update last login time
    db.update_last_login(user['CustomerID'])
    
    # Create permanent session with timeout
    session.permanent = True
    session['user_id'] = user['CustomerID']
    session['user_name'] = user['Name']
    session['user_email'] = user['Email_ID']
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'redirect': '/app'
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('main_app'))
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def process_signup():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['fullName', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    name = data['fullName']
    email = data['email']
    password = data['password']
    
    # Hash password before storing
    hashed_password = generate_password_hash(password)
    
    try:
        # Check if email already exists
        existing_user = db.get_user_by_email(email)
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        success = db.create_user(name, email, hashed_password)
        if not success:
            return jsonify({'error': 'Failed to create user'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'redirect': '/login'
        })
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/app')
@login_required
def main_app():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
@monitor_performance('upload')
def upload_document():
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endsWith('.pdf'):
        temp_path = os.path.join("temp_files", f"temp_{uuid.uuid4().hex}.pdf")
        file.save(temp_path)
        
        try:
            logger.info(f"Processing file: {file.filename}")
            
            # Extract text and metadata in parallel
            text_future = executor.submit(document_processor.extract_text_from_pdf, temp_path)
            text, metadata = text_future.result()
            
            # Clean text and extract keywords in parallel
            clean_future = executor.submit(document_processor.clean_text, text)
            keywords_future = executor.submit(document_processor.extract_keywords, text)
            
            cleaned_text = clean_future.result()
            if not metadata.get('keywords'):
                metadata['keywords'] = keywords_future.result()
            
            # Chunk document
            sections = document_processor.chunk_document(cleaned_text)
            
            # Prepare documents for vector DB in parallel
            documents = []
            futures = []
            
            # Process traditional sections
            for section_name, section_text in sections.items():
                if section_text and section_name not in ['full_text', 'title'] and not section_name.endswith('_chunks'):
                    doc_id = f"{uuid.uuid4().hex}_{section_name}"
                    documents.append({
                        'id': doc_id,
                        'text': section_text,
                        'title': sections.get('title', 'Untitled'),
                        'section': section_name,
                        'source': file.filename,
                        'authors': metadata['authors'],
                        'year': metadata['year'],
                        'keywords': metadata['keywords']
                    })
            
            # Process semantic chunks in parallel
            for section in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']:
                if f'{section}_chunks' in sections:
                    for chunk in sections[f'{section}_chunks']:
                        doc_id = f"{uuid.uuid4().hex}_{section}_chunk"
                        documents.append({
                            'id': doc_id,
                            'text': chunk['text'],
                            'title': sections.get('title', 'Untitled'),
                            'section': section,
                            'source': file.filename,
                            'authors': metadata['authors'],
                            'year': metadata['year'],
                            'keywords': metadata['keywords'],
                            'is_semantic': True
                        })
            
            # Add full combined context
            combined_text = document_processor.combine_sections(sections, 
                ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion'])
            if combined_text:
                documents.append({
                    'id': f"{uuid.uuid4().hex}_combined",
                    'text': combined_text,
                    'title': sections.get('title', 'Untitled'),
                    'section': 'combined',
                    'source': file.filename,
                    'authors': metadata['authors'],
                    'year': metadata['year'],
                    'keywords': metadata['keywords']
                })
            
            logger.info(f"Upserting {len(documents)} document chunks")
            
            # Upsert documents in parallel batches
            vector_db.upsert_documents(documents)
            
            return jsonify({
                'success': True,
                'message': 'Document processed successfully',
                'title': sections.get('title', 'Untitled'),
                'authors': metadata['authors'],
                'year': metadata['year'],
                'keywords': metadata['keywords'],
                'sections': list(sections.keys()),
                'fileId': str(uuid.uuid4())  # Generate a unique ID for the file
            })
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f"Failed to process document: {str(e)}"}), 500
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    else:
        logger.error("Invalid file type uploaded")
        return jsonify({'error': 'Invalid file type. Only PDFs are supported.'}), 400

@app.route('/query', methods=['POST'])
@login_required
@monitor_performance('query')
def handle_query():
    data = request.get_json()
    if not data or 'query' not in data:
        logger.error("No query provided")
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query'].strip()
    if not query:
        logger.error("Empty query provided")
        return jsonify({'error': 'Empty query provided'}), 400
    
    try:
        optimize_index_if_needed()
        
        logger.info(f"Processing query: {query}")
        similar_docs = vector_db.search_similar(query, threshold=0.4)
        
        if not similar_docs:
            logger.warning("No relevant documents found")
            return jsonify({'error': 'No relevant documents found. Try uploading more research papers or rephrasing your question.'}), 404
        
        logger.info(f"Found {len(similar_docs)} relevant documents")
        response = rag_handler.generate_response(query, similar_docs)
        
        # Save chat history
        db.save_chat_history(session['user_id'], query, response)
        
        return jsonify({
            'success': True,
            'response': response,
            'sources': [{
                'title': doc['title'],
                'authors': doc.get('authors', []),
                'year': doc.get('year'),
                'section': doc['section'],
                'score': float(doc['score']),
                'source': doc['source'],
                'keywords': doc.get('keywords', []),
                'fileId': data.get('fileId')  # Pass through the file ID if provided
            } for doc in similar_docs]
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)