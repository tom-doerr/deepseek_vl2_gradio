import pytest
from PIL import Image
import io
import torch
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_model, draw_bounding_boxes, MODEL_PATHS

@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    return img

@pytest.fixture
def mock_model_output():
    return "<|det|>person<box>0.1,0.1,0.5,0.5</box><|/det|><|det|>car<box>0.6,0.6,0.9,0.9</box><|/det|>"

def test_draw_bounding_boxes(sample_image, mock_model_output):
    # Test bounding box drawing functionality
    result = draw_bounding_boxes(sample_image, mock_model_output)
    
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size
    assert result != sample_image  # Should be a different image due to drawings

def test_draw_bounding_boxes_invalid_coords(sample_image):
    # Test with invalid coordinates
    invalid_output = "<|det|>person<box>invalid,coords</box><|/det|>"
    result = draw_bounding_boxes(sample_image, invalid_output)
    
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size

def test_draw_bounding_boxes_empty_text(sample_image):
    # Test with no detection tags
    result = draw_bounding_boxes(sample_image, "No detections here")
    
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size

def is_flash_attn_installed():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_flash_attn_installed(), reason="flash_attn not installed")
def test_load_model():
    # Test model loading for tiny model
    vl_chat_processor, tokenizer, vl_gpt = load_model("tiny")
    
    assert vl_chat_processor is not None
    assert tokenizer is not None
    assert vl_gpt is not None
    assert next(vl_gpt.parameters()).device.type == 'cuda'

def test_model_paths():
    # Test that all model paths are valid
    assert "tiny" in MODEL_PATHS
    assert "small" in MODEL_PATHS
    assert "base" in MODEL_PATHS
    assert all(isinstance(path, str) for path in MODEL_PATHS.values())
    assert all(path.startswith("deepseek-ai/deepseek-vl2") for path in MODEL_PATHS.values())
