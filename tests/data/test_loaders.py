"""
Comprehensive tests for data loaders.

Tests various data loading scenarios including file, database, API,
and credit-specific data loading with error handling.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import numpy as np

from src.data.loaders import (
    DataLoader,
    FileDataLoader,
    DatabaseDataLoader,
    APIDataLoader,
    CreditDataLoader,
    DataLoaderFactory
)


class TestDataLoader:
    """Test suite for abstract DataLoader class."""
    
    def test_abstract_methods(self):
        """Test that DataLoader is abstract."""
        with pytest.raises(TypeError):
            DataLoader()


class TestFileDataLoader:
    """Test suite for FileDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = FileDataLoader()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_supported_formats(self):
        """Test supported file formats."""
        expected_formats = {'.csv', '.json', '.parquet', '.xlsx', '.tsv'}
        assert self.loader.supported_formats == expected_formats
    
    def test_validate_source_valid_csv(self):
        """Test validation of valid CSV file."""
        # Create a test CSV file
        csv_file = self.temp_dir / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        
        assert self.loader.validate_source(str(csv_file)) == True
    
    def test_validate_source_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_file = self.temp_dir / "nonexistent.csv"
        assert self.loader.validate_source(str(nonexistent_file)) == False
    
    def test_validate_source_unsupported_format(self):
        """Test validation of unsupported file format."""
        txt_file = self.temp_dir / "test.txt"
        txt_file.write_text("some text")
        
        assert self.loader.validate_source(str(txt_file)) == False
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        # Create test CSV
        csv_content = "name,age,score\nAlice,25,85\nBob,30,92\n"
        csv_file = self.temp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        data = self.loader.load(str(csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert list(data.columns) == ["name", "age", "score"]
        assert data.iloc[0]["name"] == "Alice"
    
    def test_load_json_file(self):
        """Test loading JSON file."""
        # Create test JSON
        json_data = [
            {"name": "Alice", "age": 25, "score": 85},
            {"name": "Bob", "age": 30, "score": 92}
        ]
        json_file = self.temp_dir / "test.json"
        json_file.write_text(json.dumps(json_data))
        
        data = self.loader.load(str(json_file))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert data.iloc[0]["name"] == "Alice"
    
    def test_load_tsv_file(self):
        """Test loading TSV file."""
        # Create test TSV
        tsv_content = "name\tage\tscore\nAlice\t25\t85\nBob\t30\t92\n"
        tsv_file = self.temp_dir / "test.tsv"
        tsv_file.write_text(tsv_content)
        
        data = self.loader.load(str(tsv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert list(data.columns) == ["name", "age", "score"]
    
    def test_load_csv_with_encoding_fallback(self):
        """Test CSV loading with encoding fallback."""
        # Create CSV with special characters
        csv_content = "name,description\nAlice,café\nBob,naïve\n"
        csv_file = self.temp_dir / "test.csv"
        
        # Write with latin-1 encoding
        with open(csv_file, 'w', encoding='latin-1') as f:
            f.write(csv_content)
        
        # Should succeed with encoding fallback
        data = self.loader.load(str(csv_file))
        assert len(data) == 2
    
    def test_load_invalid_file(self):
        """Test loading invalid file."""
        nonexistent_file = self.temp_dir / "nonexistent.csv"
        
        with pytest.raises(ValueError, match="Invalid or inaccessible file"):
            self.loader.load(str(nonexistent_file))
    
    def test_metadata_update(self):
        """Test metadata update after loading."""
        # Create test CSV
        csv_content = "col1,col2\n1,2\n3,4\n"
        csv_file = self.temp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        data = self.loader.load(str(csv_file))
        metadata = self.loader.get_metadata()
        
        assert metadata["source"] == str(csv_file)
        assert metadata["shape"] == (2, 2)
        assert "col1" in metadata["columns"]
        assert "load_time" in metadata


class TestDatabaseDataLoader:
    """Test suite for DatabaseDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'connection': {
                'type': 'sqlite',
                'path': ':memory:'
            }
        }
        self.loader = DatabaseDataLoader(self.config)
    
    def test_validate_source_valid_query(self):
        """Test validation of valid SQL query."""
        query = "SELECT * FROM users WHERE age > 25"
        assert self.loader.validate_source(query) == True
    
    def test_validate_source_table_name(self):
        """Test validation of table name."""
        table_name = "users"
        assert self.loader.validate_source(table_name) == True
    
    def test_validate_source_dangerous_sql(self):
        """Test validation rejects dangerous SQL."""
        dangerous_queries = [
            "DROP TABLE users",
            "DELETE FROM users",
            "INSERT INTO users VALUES (1, 'hack')",
            "UPDATE users SET name = 'hacked'"
        ]
        
        for query in dangerous_queries:
            assert self.loader.validate_source(query) == False
    
    def test_validate_source_empty_string(self):
        """Test validation of empty string."""
        assert self.loader.validate_source("") == False
        assert self.loader.validate_source(None) == False
    
    @patch('pandas.read_sql_query')
    @patch('sqlite3.connect')
    def test_load_sql_query(self, mock_connect, mock_read_sql):
        """Test loading data from SQL query."""
        # Setup mocks
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        mock_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        mock_read_sql.return_value = mock_data
        
        # Test query execution
        query = "SELECT * FROM users"
        data = self.loader.load(query)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        mock_read_sql.assert_called_once_with(query, mock_connection)
    
    @patch('pandas.read_sql_table')
    @patch('sqlite3.connect')
    def test_load_table_name(self, mock_connect, mock_read_sql):
        """Test loading data from table name."""
        # Setup mocks
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        mock_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_read_sql.return_value = mock_data
        
        # Test table loading
        table_name = "users"
        data = self.loader.load(table_name)
        
        assert isinstance(data, pd.DataFrame)
        mock_read_sql.assert_called_once_with(table_name, mock_connection)


class TestAPIDataLoader:
    """Test suite for APIDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = APIDataLoader()
    
    def test_validate_source_valid_url(self):
        """Test validation of valid URLs."""
        valid_urls = [
            "https://api.example.com/data",
            "http://localhost:8080/api/users"
        ]
        
        for url in valid_urls:
            assert self.loader.validate_source(url) == True
    
    def test_validate_source_invalid_url(self):
        """Test validation of invalid URLs."""
        invalid_urls = [
            "ftp://example.com/file",
            "not-a-url",
            "",
            None
        ]
        
        for url in invalid_urls:
            assert self.loader.validate_source(url) == False
    
    @patch('requests.Session.request')
    def test_load_json_api(self, mock_request):
        """Test loading JSON data from API."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = [
            {'id': 1, 'name': 'Alice', 'age': 25},
            {'id': 2, 'name': 'Bob', 'age': 30}
        ]
        mock_request.return_value = mock_response
        
        # Test API call
        url = "https://api.example.com/users"
        data = self.loader.load(url)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert data.iloc[0]['name'] == 'Alice'
    
    @patch('requests.Session.request')
    def test_load_json_api_with_data_key(self, mock_request):
        """Test loading JSON with nested data structure."""
        # Setup mock response with nested data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {
            'data': [
                {'id': 1, 'name': 'Alice'},
                {'id': 2, 'name': 'Bob'}
            ],
            'status': 'success'
        }
        mock_request.return_value = mock_response
        
        data = self.loader.load("https://api.example.com/users")
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
    
    @patch('requests.Session.request')
    def test_load_csv_api(self, mock_request):
        """Test loading CSV data from API."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/csv'}
        mock_response.text = "name,age\nAlice,25\nBob,30\n"
        mock_request.return_value = mock_response
        
        data = self.loader.load("https://api.example.com/users.csv")
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert list(data.columns) == ['name', 'age']
    
    @patch('requests.Session.request')
    def test_load_api_error(self, mock_request):
        """Test API loading error handling."""
        # Setup mock error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        mock_request.return_value = mock_response
        
        with pytest.raises(Exception):
            self.loader.load("https://api.example.com/nonexistent")
    
    def test_load_unsupported_content_type(self):
        """Test handling of unsupported content type."""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/xml'}
            mock_request.return_value = mock_response
            
            with pytest.raises(ValueError, match="Unsupported content type"):
                self.loader.load("https://api.example.com/data.xml")


class TestCreditDataLoader:
    """Test suite for CreditDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'required_columns': ['age', 'income', 'credit_score'],
            'protected_attributes': ['age'],
            'target_column': 'approved'
        }
        self.loader = CreditDataLoader(self.config)
        
        # Create test data
        self.temp_dir = Path(tempfile.mkdtemp())
        
        credit_data = {
            'age': [25, 35, 45, 55],
            'income': [40000, 50000, 60000, 70000],
            'credit_score': [650, 700, 750, 800],
            'debt_to_income': [0.3, 0.2, 0.1, 0.15],
            'approved': [0, 1, 1, 1]
        }
        self.credit_df = pd.DataFrame(credit_data)
        
        # Save to CSV
        self.credit_csv = self.temp_dir / "credit_data.csv"
        self.credit_df.to_csv(self.credit_csv, index=False)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_credit_data(self):
        """Test loading credit data with validation."""
        data = self.loader.load(str(self.credit_csv))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 4
        assert 'protected' in data.columns  # Should be created from age
        assert all(col in data.columns for col in self.config['required_columns'])
    
    def test_load_with_split(self):
        """Test loading with train/test split."""
        X_train, X_test, y_train, y_test = self.loader.load_with_split(
            str(self.credit_csv),
            test_size=0.5,
            random_state=42
        )
        
        assert len(X_train) == 2
        assert len(X_test) == 2
        assert len(y_train) == 2
        assert len(y_test) == 2
        assert self.config['target_column'] not in X_train.columns
        assert self.config['target_column'] not in X_test.columns
    
    def test_validate_credit_data_missing_columns(self):
        """Test validation with missing required columns."""
        # Create data missing required columns
        incomplete_data = {
            'age': [25, 35],
            'income': [40000, 50000]
            # Missing credit_score
        }
        incomplete_df = pd.DataFrame(incomplete_data)
        incomplete_csv = self.temp_dir / "incomplete.csv"
        incomplete_df.to_csv(incomplete_csv, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.loader.load(str(incomplete_csv))
    
    def test_validate_feature_ranges(self):
        """Test feature range validation."""
        # Create data with out-of-range values
        invalid_data = {
            'age': [25, 150],  # 150 is out of range
            'income': [40000, -5000],  # Negative income
            'credit_score': [650, 1000],  # 1000 is out of range
            'approved': [0, 1]
        }
        invalid_df = pd.DataFrame(invalid_data)
        invalid_csv = self.temp_dir / "invalid.csv"
        invalid_df.to_csv(invalid_csv, index=False)
        
        # Should load but with warnings
        data = self.loader.load(str(invalid_csv))
        assert len(data) == 2  # Should still load the data
    
    def test_preprocess_credit_data(self):
        """Test credit data preprocessing."""
        data = self.loader.load(str(self.credit_csv))
        
        # Check protected attribute creation
        assert 'protected' in data.columns
        assert data['protected'].dtype in [int, 'int64']
        
        # Check for missing value handling
        assert data.isnull().sum().sum() == 0
    
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        # Create data without target column
        no_target_data = {
            'age': [25, 35],
            'income': [40000, 50000],
            'credit_score': [650, 700]
        }
        no_target_df = pd.DataFrame(no_target_data)
        no_target_csv = self.temp_dir / "no_target.csv"
        no_target_df.to_csv(no_target_csv, index=False)
        
        # Should create synthetic target
        data = self.loader.load(str(no_target_csv))
        assert self.config['target_column'] in data.columns
    
    async def test_load_async(self):
        """Test asynchronous loading."""
        data = await self.loader.load_async(str(self.credit_csv))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 4


class TestDataLoaderFactory:
    """Test suite for DataLoaderFactory class."""
    
    def test_create_file_loader(self):
        """Test creating file data loader."""
        loader = DataLoaderFactory.create_loader('file')
        assert isinstance(loader, FileDataLoader)
    
    def test_create_database_loader(self):
        """Test creating database data loader."""
        loader = DataLoaderFactory.create_loader('database')
        assert isinstance(loader, DatabaseDataLoader)
    
    def test_create_api_loader(self):
        """Test creating API data loader."""
        loader = DataLoaderFactory.create_loader('api')
        assert isinstance(loader, APIDataLoader)
    
    def test_create_credit_loader(self):
        """Test creating credit data loader."""
        loader = DataLoaderFactory.create_loader('credit')
        assert isinstance(loader, CreditDataLoader)
    
    def test_create_loader_with_config(self):
        """Test creating loader with configuration."""
        config = {'timeout': 60}
        loader = DataLoaderFactory.create_loader('api', config)
        assert loader.config == config
    
    def test_create_unknown_loader(self):
        """Test creating unknown loader type."""
        with pytest.raises(ValueError, match="Unknown source type"):
            DataLoaderFactory.create_loader('unknown')


# Integration tests
@pytest.mark.integration
class TestDataLoadersIntegration:
    """Integration tests for data loaders."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_to_credit_loader_workflow(self):
        """Test workflow from file loader to credit loader."""
        # Create test credit data
        credit_data = {
            'age': [25, 35, 45],
            'income': [40000, 50000, 60000],
            'credit_score': [650, 700, 750],
            'debt_to_income': [0.3, 0.2, 0.1],
            'approved': [0, 1, 1]
        }
        credit_df = pd.DataFrame(credit_data)
        credit_csv = self.temp_dir / "credit.csv"
        credit_df.to_csv(credit_csv, index=False)
        
        # 1. Load with file loader
        file_loader = FileDataLoader()
        raw_data = file_loader.load(str(credit_csv))
        
        # 2. Process with credit loader
        credit_loader = CreditDataLoader()
        processed_data = credit_loader._preprocess_credit_data(raw_data)
        
        assert 'protected' in processed_data.columns
        assert len(processed_data) == len(raw_data)
    
    def test_multiple_format_support(self):
        """Test loading the same data in different formats."""
        # Create test data
        test_data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [85, 92, 78]
        }
        df = pd.DataFrame(test_data)
        
        # Save in different formats
        csv_file = self.temp_dir / "data.csv"
        json_file = self.temp_dir / "data.json"
        
        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient='records')
        
        # Load with file loader
        loader = FileDataLoader()
        
        csv_data = loader.load(str(csv_file))
        json_data = loader.load(str(json_file))
        
        # Should produce equivalent data
        assert len(csv_data) == len(json_data)
        assert list(csv_data.columns) == list(json_data.columns)


# Fixtures
@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    csv_file = temp_dir / "sample.csv"
    
    data = {
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85, 92, 78]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    yield str(csv_file)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return [
        {'id': 1, 'name': 'Alice', 'age': 25},
        {'id': 2, 'name': 'Bob', 'age': 30},
        {'id': 3, 'name': 'Charlie', 'age': 35}
    ]