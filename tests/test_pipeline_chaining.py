"""
Tests for Pipeline Chaining functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.processing.pipeline import PipelineChain
from src.processing.base import EnhancementPipeline, ProcessingAborted
from src.models.unified_registry import UnifiedModelRegistry
@pytest.fixture
def mock_pipeline_class():
    """Create a mock pipeline class for testing"""
    class MockPipeline(EnhancementPipeline):
        def __init__(self, model_name="TestModel", device="auto", **kwargs):
            super().__init__(device)
            self.model_name = model_name
            self.load_count = 0
            self.process_count = 0
            self._model = None
            
        def load_model(self, model_path=None, **kwargs):
            self.load_count += 1
            self._model = "mock_model"
            return True
            
        def process(self, input_path, output_path, **kwargs):
            self.process_count += 1
            return {
                "success": True,
                "input_path": input_path,
                "output_path": output_path,
                "model": self.model_name,
                "processed": True
            }
    
    return MockPipeline
@pytest.fixture
def temp_files():
    """Create temporary files for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.txt"
        output_path = Path(tmpdir) / "output.txt"
        
        input_path.write_text("test input")
        yield str(input_path), str(output_path)
@pytest.fixture
def mock_pipelines(mock_pipeline_class):
    """Create mock pipeline instances"""
    pipeline1 = mock_pipeline_class("model1", "cpu")
    pipeline2 = mock_pipeline_class("model2", "cpu")
    pipeline3 = mock_pipeline_class("model3", "cpu")
    
    return [pipeline1, pipeline2, pipeline3]
class TestPipelineChain:
    """Test the PipelineChain class"""
    
    def test_pipeline_chain_initialization(self, mock_pipeline_class):
        """Test that PipelineChain can be initialized with pipelines"""
        pipeline1 = mock_pipeline_class()
        pipeline2 = mock_pipeline_class()
        
        chain = PipelineChain([pipeline1, pipeline2])
        
        assert chain.pipelines == [pipeline1, pipeline2]
        assert len(chain.pipelines) == 2
    
    def test_pipeline_chain_process_single_pipeline(self, mock_pipeline_class, temp_files):
        """Test PipelineChain with single pipeline"""
        input_path, output_path = temp_files
        pipeline = mock_pipeline_class("single_model")
        
        chain = PipelineChain([pipeline])
        result = chain.process(input_path, output_path)
        
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["output_path"] == output_path
        assert "pipelines_executed" in result
        assert pipeline.process_count == 1
        assert pipeline.load_count == 1
    
    def test_pipeline_chain_process_multiple_pipelines(self, mock_pipelines, temp_files):
        """Test PipelineChain with multiple pipelines"""
        input_path, output_path = temp_files
        chain = PipelineChain(mock_pipelines)
        
        result = chain.process(input_path, output_path)
        
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["output_path"] == output_path
        assert "pipelines_executed" in result
        
        # Check that each pipeline was processed
        for pipeline in mock_pipelines:
            assert pipeline.process_count == 1
            assert pipeline.load_count == 1
        
        # Check result structure
        assert len(result["pipelines_executed"]) == 3
        assert mock_pipelines[0].__class__.__name__ in result["pipelines_executed"]
    
    def test_pipeline_chain_empty_pipelines(self, temp_files):
        """Test PipelineChain with empty pipeline list"""
        input_path, output_path = temp_files
        
        chain = PipelineChain([])
        result = chain.process(input_path, output_path)
        
        # Should still return success but process zero pipelines
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["output_path"] == output_path
        assert result["pipelines_executed"] == []
    
    def test_pipeline_chain_processing_aborted(self, mock_pipelines, temp_files):
        """Test that ProcessingAborted exception is propagated"""
        input_path, output_path = temp_files
        
        # Make first pipeline raise ProcessingAborted
        mock_pipelines[0].process = Mock(side_effect=ProcessingAborted("Test abort"))
        
        chain = PipelineChain(mock_pipelines)
        
        with pytest.raises(ProcessingAborted):
            chain.process(input_path, output_path)
        
        # First pipeline should have been processed (load called)
        assert mock_pipelines[0].load_count == 1
    
    def test_pipeline_chain_generic_exception(self, mock_pipelines, temp_files):
        """Test that generic exceptions are raised"""
        input_path, output_path = temp_files
        
        # Make second pipeline raise a generic exception
        mock_pipelines[1].process = Mock(side_effect=RuntimeError("Test error"))
        
        chain = PipelineChain(mock_pipelines)
        
        with pytest.raises(RuntimeError):
            chain.process(input_path, output_path)
        
        # First pipeline should have been processed
        assert mock_pipelines[0].process_count == 1
class TestUnifiedModelRegistry:
    """Test the UnifiedModelRegistry"""
    
    def test_registry_initialization(self):
        """Test that registry can be initialized"""
        registry = UnifiedModelRegistry()
        
        assert registry is not None
        assert hasattr(registry, '_load_model_registry')
        assert hasattr(registry, 'get_registry')
        assert hasattr(registry, 'get_models_by_feature')
    
    def test_get_registry(self, monkeypatch):
        """Test registry data retrieval"""
        registry = UnifiedModelRegistry()
        
        # Mock the internal method
        mock_registry_data = {
            "Upscale": [
                {"name": "Real-ESRGAN", "is_builtin": True},
                {"name": "RealBasicVSR", "is_builtin": True}
            ],
            "Face Recovery": [
                {"name": "CodeFormer", "is_builtin": True}
            ]
        }
        
        monkeypatch.setattr(registry, '_merge_catalog_and_hardcoded', return_value=mock_registry_data)
        
        result = registry.get_registry()
        
        assert result == mock_registry_data
        assert "Upscale" in result
        assert "Face Recovery" in result
        assert len(result["Upscale"]) == 2
        assert len(result["Face Recovery"]) == 1
    
    def test_get_models_by_feature(self, monkeypatch):
        """Test getting models by feature"""
        registry = UnifiedModelRegistry()
        
        mock_registry_data = {
            "Upscale": [{"name": "Real-ESRGAN", "is_builtin": True}],
            "Face Recovery": [{"name": "CodeFormer", "is_builtin": True}]
        }
        
        monkeypatch.setattr(registry, 'get_registry', return_value=mock_registry_data)
        
        result = registry.get_models_by_feature("Upscale")
        
        assert len(result) == 1
        assert result[0]["name"] == "Real-ESRGAN"
        assert result[0]["is_builtin"] is True
    
    def test_get_model_by_feature_and_name(self, monkeypatch):
        """Test getting a specific model by feature and name"""
        registry = UnifiedModelRegistry()
        
        mock_registry_data = {
            "Upscale": [
                {"name": "Real-ESRGAN", "is_builtin": True},
                {"name": "RealBasicVSR", "is_builtin": True}
            ]
        }
        
        monkeypatch.setattr(registry, 'get_models_by_feature', return_value=mock_registry_data["Upscale"])
        
        result = registry.get_model_by_feature_and_name("Upscale", "Real-ESRGAN")
        
        assert result is not None
        assert result["name"] == "Real-ESRGAN"
        assert result["is_builtin"] is True
        
        # Test non-existent model
        result2 = registry.get_model_by_feature_and_name("Upscale", "NonExistent")
        assert result2 is None
    
    def test_get_all_features(self, monkeypatch):
        """Test getting all available features"""
        registry = UnifiedModelRegistry()
        
        mock_registry_data = {
            "Upscale": [],
            "Face Recovery": [],
            "Sharpen / Deblur": []
        }
        
        monkeypatch.setattr(registry, 'get_registry', return_value=mock_registry_data)
        
        result = registry.get_all_features()
        
        assert set(result) == {"Upscale", "Face Recovery", "Sharpen / Deblur"}
        assert len(result) == 3
@pytest.fixture
def mock_processing_module():
    """Mock the processing module for integration tests"""
    with patch('src.processing.get_unified_registry') as mock_registry:
        mock_registry.return_value = UnifiedModelRegistry()
        
        with patch('src.processing.get_models_by_feature') as mock_get_models:
            mock_get_models.return_value = [
                {"name": "Real-ESRGAN", "is_builtin": True}
            ]
            
            yield
class TestPipelineChainingIntegration:
    """Integration tests for pipeline chaining"""
    
    def test_pipeline_chain_with_real_pipelines(self, mock_processing_module):
        """Test that pipeline chaining works with real pipeline classes"""
        from src.processing import PipelineChain, UpscalePipeline, FaceRecoveryPipeline
        
        # Create mock instances
        upscale_pipeline = Mock(spec=UpscalePipeline)
        face_pipeline = Mock(spec=FaceRecoveryPipeline)
        
        upscale_pipeline.load_model.return_value = True
        upscale_pipeline.process.return_value = {
            "success": True,
            "input_path": "input.jpg",
            "output_path": "output1.jpg"
        }
        
        face_pipeline.load_model.return_value = True
        face_pipeline.process.return_value = {
            "success": True,
            "input_path": "output1.jpg",
            "output_path": "output2.jpg"
        }
        
        chain = PipelineChain([upscale_pipeline, face_pipeline])
        
        # Mock the dependencies that PipelineChain expects
        with patch('src.processing.pipeline.PipelineChain') as MockPipelineChain:
            mock_chain_instance = Mock()
            mock_chain_instance.process.return_value = {
                "success": True,
                "input_path": "input.jpg",
                "output_path": "output2.jpg"
            }
            MockPipelineChain.return_value = mock_chain_instance
            
            result = chain.process("input.jpg", "output2.jpg")
            
            # Verify chain was created and processed
            assert result["success"] is True
            assert result["input_path"] == "input.jpg"
            assert result["output_path"] == "output2.jpg"