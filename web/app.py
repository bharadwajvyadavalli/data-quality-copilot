"""
Flask Web Application for Data Quality Co-pilot
Provides web interface for data quality inspection with real-time processing visualization.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import pandas as pd
import tempfile
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import DataQualityPipeline


# Configure Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'data-quality-copilot-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the data quality pipeline."""
    global pipeline
    if pipeline is None:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        pipeline = DataQualityPipeline(openai_api_key=openai_api_key)
        logger.info("Pipeline initialized")


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'}), 400
            
        # Save uploaded file
        session_id = str(uuid.uuid4())
        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{session_id}_{file.filename}"
        file.save(file_path)
        
        # Store session info
        session['file_path'] = str(file_path)
        session['filename'] = file.filename
        session['session_id'] = session_id
        session['upload_time'] = datetime.now().isoformat()
        
        logger.info(f"File uploaded: {file.filename} -> {file_path}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': file.filename,
            'message': 'File uploaded successfully. Starting analysis...'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Start data quality analysis."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id != session.get('session_id'):
            return jsonify({'error': 'Invalid session'}), 400
            
        file_path = session.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400
            
        # Initialize pipeline if needed
        initialize_pipeline()
        
        # Start analysis
        logger.info(f"Starting analysis for session {session_id}")
        
        # Run pipeline
        results = pipeline.run_pipeline(file_path)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 500
            
        # Store results in session
        session['analysis_results'] = results
        session['analysis_complete'] = True
        session['analysis_time'] = datetime.now().isoformat()
        
        # Prepare response
        response_data = {
            'success': True,
            'session_id': session_id,
            'pipeline_summary': results['pipeline_summary'],
            'dashboard_data': results['reporting_results']['dashboard_data'],
            'message': 'Analysis completed successfully'
        }
        
        logger.info(f"Analysis completed for session {session_id}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/status/<session_id>')
def get_status(session_id):
    """Get analysis status and progress."""
    try:
        if session_id != session.get('session_id'):
            return jsonify({'error': 'Invalid session'}), 400
            
        # Get agent thoughts for real-time updates
        if pipeline:
            thoughts = pipeline.get_agent_thoughts()
            processing_summaries = pipeline.get_processing_summaries()
        else:
            thoughts = {}
            processing_summaries = {}
            
        status_data = {
            'session_id': session_id,
            'analysis_complete': session.get('analysis_complete', False),
            'upload_time': session.get('upload_time'),
            'analysis_time': session.get('analysis_time'),
            'filename': session.get('filename'),
            'agent_thoughts': thoughts,
            'processing_summaries': processing_summaries
        }
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/results/<session_id>')
def get_results(session_id):
    """Get analysis results."""
    try:
        if session_id != session.get('session_id'):
            return jsonify({'error': 'Invalid session'}), 400
            
        results = session.get('analysis_results')
        if not results:
            return jsonify({'error': 'No results available'}), 404
            
        # Prepare results for frontend
        response_data = {
            'session_id': session_id,
            'pipeline_summary': results['pipeline_summary'],
            'dashboard_data': results['reporting_results']['dashboard_data'],
            'validation_results': results['validation_results'],
            'schema_results': results['schema_results'],
            'ingestion_results': results['ingestion_results'],
            'agent_thoughts': pipeline.get_agent_thoughts() if pipeline else {},
            'mcp_context': results.get('mcp_context', {})
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Results error: {str(e)}")
        return jsonify({'error': f'Results retrieval failed: {str(e)}'}), 500


@app.route('/download/<session_id>/<report_type>')
def download_report(session_id, report_type):
    """Download generated reports."""
    try:
        if session_id != session.get('session_id'):
            return jsonify({'error': 'Invalid session'}), 400
            
        results = session.get('analysis_results')
        if not results:
            return jsonify({'error': 'No results available'}), 404
            
        reporting_results = results.get('reporting_results', {})
        file_paths = reporting_results.get('file_paths', {})
        
        if report_type not in file_paths:
            return jsonify({'error': f'Report type {report_type} not available'}), 404
            
        file_path = file_paths[report_type]
        if not os.path.exists(file_path):
            return jsonify({'error': 'Report file not found'}), 404
            
        # Determine filename
        filename = f"data_quality_report_{session_id}.{report_type}"
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/sample-data')
def get_sample_data():
    """Generate and return sample data for testing."""
    try:
        # Create sample data
        import numpy as np
        
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, 51),
            'name': [f'Customer_{i}' for i in range(1, 51)],
            'email': [f'customer{i}@example.com' for i in range(1, 51)],
            'age': np.random.randint(18, 80, 50),
            'income': np.random.normal(50000, 15000, 50),
            'purchase_amount': np.random.exponential(100, 50),
            'customer_type': np.random.choice(['Premium', 'Standard', 'Basic'], 50),
            'satisfaction_score': np.random.randint(1, 11, 50)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce quality issues
        df.loc[5:8, 'email'] = None
        df.loc[15:18, 'age'] = None
        df.loc[25, 'income'] = 1000000  # Outlier
        df.loc[30, 'satisfaction_score'] = 15  # Invalid value
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # Read file content
        with open(temp_file.name, 'r') as f:
            content = f.read()
            
        # Clean up
        os.unlink(temp_file.name)
        
        return jsonify({
            'success': True,
            'filename': 'sample_customer_data.csv',
            'content': content,
            'rows': len(df),
            'columns': len(df.columns)
        })
        
    except Exception as e:
        logger.error(f"Sample data error: {str(e)}")
        return jsonify({'error': f'Sample data generation failed: {str(e)}'}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_initialized': pipeline is not None
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize pipeline on startup
    initialize_pipeline()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Data Quality Co-pilot web application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug) 