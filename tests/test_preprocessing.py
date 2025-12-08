# ============================================================
# Tests for prepare_features()
# ============================================================

class TestPrepareFeatures:
    """Tests for the prepare_features function."""
    
    def test_removes_target_column(self, sample_raw_data):
        """Test that target column 'y' is removed."""
        cleaned = clean_data(sample_raw_data)
        result = prepare_features(cleaned)
        
        assert 'y' not in result.columns
    
    def test_keeps_feature_columns(self, sample_raw_data):
        """Test that feature columns are kept."""
        cleaned = clean_data(sample_raw_data)
        result = prepare_features(cleaned)
        
        # Check some expected columns exist
        assert 'age' in result.columns
        assert 'job' in result.columns
        assert 'balance' in result.columns



class TestCreatePreprocessor:
    """Tests for the create_preprocessor function."""
    
    def test_returns_column_transformer(self):
        """Test that function returns a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        
        preprocessor = create_preprocessor()
        
        assert isinstance(preprocessor, ColumnTransformer)
    
   

# ============================================================
# Run tests
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
