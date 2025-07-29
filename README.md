# Data Quality Co-pilot

An AI-powered data quality inspection tool that automates the process of analyzing datasets using a multi-agent architecture with LLM integration.

## 🚀 Features

- **Multi-Agent Architecture**: Chain-of-Thought workflow with specialized agents
- **LLM-Powered Schema Generation**: Automated data dictionary creation using OpenAI GPT
- **Comprehensive Quality Validation**: Statistical analysis, outlier detection, and constraint validation
- **Model Context Protocol (MCP) Integration**: Structured agent communication and context management
- **Web Dashboard**: Real-time processing visualization with interactive charts
- **Console Interface**: Command-line demo with detailed reporting
- **Multiple Output Formats**: JSON, YAML, HTML, and console reports

## 🏗️ Architecture

### Multi-Agent System

1. **Ingestion Agent**: Handles file reading, data loading, and preprocessing
2. **Schema Agent**: Uses LLM to generate comprehensive data schemas and dictionaries
3. **Validation Agent**: Applies quality rules and performs statistical analysis
4. **Reporting Agent**: Generates comprehensive reports in multiple formats

### MCP Integration

- Chunked prompt/context management
- Tool call injection capabilities
- Intermediate thought collection
- Structured agent communication

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for schema generation)
- Modern web browser (for web interface)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data-quality-copilot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

## 🚀 Quick Start

### Console Demo

Run the console demo with sample data:

```bash
python demo/console_demo.py --sample
```

Or analyze your own file:

```bash
python demo/console_demo.py --file your_data.csv
```

### Web Interface

Start the web application:

```bash
cd web
python app.py
```

Then open your browser to `http://localhost:5000`

## 📖 Usage Examples

### Console Interface

```bash
# Basic usage with sample data
python demo/console_demo.py --sample

# Analyze specific file
python demo/console_demo.py --file data.csv

# Verbose mode with detailed agent thoughts
python demo/console_demo.py --file data.csv --verbose

# Use custom OpenAI API key
python demo/console_demo.py --file data.csv --openai-key "your-key"
```

### Web Interface

1. **Upload File**: Drag and drop or click to upload CSV/Excel files
2. **Real-time Processing**: Watch agent thoughts and processing progress
3. **Interactive Dashboard**: View quality metrics, charts, and detailed analysis
4. **Download Reports**: Get comprehensive reports in JSON, YAML, or HTML formats

## 📊 Sample Output

### Console Report
```
================================================================================
DATA QUALITY INSPECTION REPORT
================================================================================
Generated: 2024-01-15 14:30:25

DATASET OVERVIEW
----------------------------------------
File: sample_customer_data.csv
Rows: 103
Columns: 10
File Size: 15,432 bytes

QUALITY ASSESSMENT
----------------------------------------
Overall Quality Score: 78.50/100
Quality Level: Fair

ISSUES SUMMARY
----------------------------------------
Critical Issues: 5
Warnings: 8
Outliers: 12
```

### Web Dashboard
- Interactive quality score visualization
- Real-time agent thought process
- Statistical analysis charts
- Schema summary and validation results

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for LLM integration
- `FLASK_ENV`: Set to 'development' for debug mode
- `PORT`: Web server port (default: 5000)

### Pipeline Configuration

The pipeline can be customized by modifying agent parameters:

```python
from src.pipeline import DataQualityPipeline

pipeline = DataQualityPipeline(
    openai_api_key="your-key",
    output_dir="custom_reports"
)
```

## 📁 Project Structure

```
data-quality-copilot/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Base agent class
│   │   ├── ingestion_agent.py     # Data ingestion agent
│   │   ├── schema_agent.py        # LLM schema generation
│   │   ├── validation_agent.py    # Quality validation
│   │   └── reporting_agent.py     # Report generation
│   ├── mcp/
│   │   └── client.py              # MCP client implementation
│   └── pipeline.py                # Main pipeline orchestrator
├── demo/
│   └── console_demo.py            # Console demo application
├── web/
│   ├── app.py                     # Flask web application
│   └── templates/
│       └── index.html             # Web dashboard
├── reports/                       # Generated reports
├── uploads/                       # Uploaded files
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔍 Agent Details

### Ingestion Agent
- Supports CSV and Excel files
- Automatic data preprocessing
- Basic quality issue detection
- Metadata generation

### Schema Agent
- LLM-powered schema generation
- Data type inference
- Constraint identification
- Business rule extraction

### Validation Agent
- Statistical analysis
- Outlier detection (IQR method)
- Constraint validation
- Quality scoring algorithm

### Reporting Agent
- Multiple output formats
- Interactive visualizations
- Comprehensive summaries
- Dashboard data generation

## 🧪 Testing

### Sample Data Generation

The tool includes built-in sample data generation with known quality issues:

- Missing values
- Duplicate rows
- Outliers
- Invalid data types
- Constraint violations

### Running Tests

```bash
# Console demo with sample data
python demo/console_demo.py --sample --verbose

# Web interface with sample data
# Use the "Load Sample Data" button in the web interface
```

## 📈 Quality Scoring

The quality score (0-100) is calculated based on:

- **Critical Issues** (5 points each): Missing files, schema violations, etc.
- **Warnings** (2 points each): Moderate quality concerns
- **Validation Issues** (3 points each): Schema constraint violations
- **Outliers** (0.1 points each, capped at 10): Statistical outliers
- **Consistency Issues** (4 points each): Data consistency problems

## 🔧 API Endpoints

### Web Application

- `GET /`: Main dashboard
- `POST /upload`: File upload
- `POST /analyze`: Start analysis
- `GET /status/<session_id>`: Get processing status
- `GET /results/<session_id>`: Get analysis results
- `GET /download/<session_id>/<type>`: Download reports
- `GET /api/sample-data`: Get sample data
- `GET /api/health`: Health check

## 🚨 Error Handling

The system includes comprehensive error handling:

- File validation and format checking
- LLM API error recovery
- Graceful degradation for missing dependencies
- Detailed error reporting and logging

## 🔒 Security Considerations

- File size limits (50MB max)
- File type validation
- Session-based file handling
- Secure file cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   - Set the `OPENAI_API_KEY` environment variable
   - Schema generation will be limited without it

2. **File Upload Fails**
   - Check file size (max 50MB)
   - Ensure file is CSV or Excel format
   - Verify file is not corrupted

3. **Web Interface Not Loading**
   - Check if Flask is running on correct port
   - Verify all dependencies are installed
   - Check browser console for JavaScript errors

4. **Analysis Takes Too Long**
   - Large files may take several minutes
   - LLM calls can be slow depending on API response time
   - Check network connectivity for API calls

### Debug Mode

Enable verbose logging:

```bash
python demo/console_demo.py --file data.csv --verbose
```

Or set Flask debug mode:

```bash
export FLASK_ENV=development
python web/app.py
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the console output for error messages
3. Enable verbose mode for detailed logging
4. Open an issue on the repository

## 🔮 Future Enhancements

- Support for more file formats (Parquet, JSON, etc.)
- Additional LLM providers (Claude, Gemini, etc.)
- Advanced visualization options
- Custom quality rule definitions
- Batch processing capabilities
- Integration with data warehouses
- Real-time data streaming support