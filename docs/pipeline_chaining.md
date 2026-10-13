# Pipeline Chaining - Documentation

## Overview

Pipeline Chaining is a new feature in openVideoAI that enables the creation of complex, multi-stage video and image enhancement workflows. Instead of applying a single enhancement operation, you can now chain multiple processing stages together (e.g., Denoise → Upscale → Face Recovery → Sharpen) to achieve more sophisticated results.

This document provides comprehensive information about pipeline chaining, including API endpoints, configuration options, and usage examples.

## What is Pipeline Chaining?

Pipeline Chaining allows you to execute multiple EnhancementPipeline instances sequentially, where the output of one pipeline becomes the input to the next. This enables complex workflows that combine different types of enhancement:

### Common Pipeline Patterns

1. **Image Enhancement Pipeline**
   - Denoise → Upscale → Face Recovery → Sharpen
   - Best for restoring old photos or heavily compressed images

2. **Video Enhancement Pipeline**
   - Frame Interpolation → Slow Motion → Sharpen
   - Best for creating cinematic slow-motion effects

3. **Color Correction Pipeline**
   - Low Light → Match Color → Sharpen
   - Best for correcting color issues in underexposed images

4. **Professional Production Pipeline**
   - Denoise → Upscale → Sharpen → Color Match → Face Recovery
   - Industry-standard multi-stage workflow

## API Endpoints

### POST `/api/processing/pipeline/file`

Execute a multi-stage pipeline on an uploaded file.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | `UploadFile` | ✅ Yes | The image or video file to process |
| `pipeline_config` | `Dict[str, Any]` | ✅ Yes | JSON configuration for the pipeline |
| `device` | `str` | ❌ No | Processing device (auto, cuda, cpu, metal, hip) |

#### Pipeline Configuration Schema

The `pipeline_config` should be a dictionary containing:

```json
{
  "features": ["denoise", "upscale", "face-recovery"],
  "denoise_model": "NAFNet",
  "denoise_strength": 0.6,
  "upscale_model": "Real-ESRGAN",
  "upscale_scale": 4,
  "face_recovery_model": "CodeFormer",
  "face_fidelity_weight": 0.7,
  "device": "cuda",
  "output_path": "optional_custom_path.jpg"
}
```

#### Available Features

| Feature | Description | Required Parameters |
|---------|-------------|---------------------|
| `denoise` | Image/video denoising | `denoise_model`, `denoise_strength` |
| `upscale` | Image/video upscaling | `upscale_model`, `upscale_scale` |
| `face-recovery` | Face restoration | `face_recovery_model`, `face_fidelity_weight` |
| `sharpen` | Sharpening/deblurring | `sharpen_model`, `sharpen_strength` |
| `frame-interpolation` | Frame interpolation | `frame_interp_model`, `frame_interp_multiplier` |
| `slow-motion` | Slow motion creation | `slow_motion_model`, `slowdown_factor` |
| `match-color` | Color matching | `match_color_model`, `match_color_strength` |
| `low-light` | Low-light enhancement | `low_light_model`, `low_light_strength` |

## Pipeline Chaining Examples

### 1. Basic Image Enhancement

```bash
curl -X POST http://localhost:8000/api/processing/pipeline/file \
  -H "Content-Type: multipart/form-data" \
  -F "file=@input.jpg" \
  -F 'pipeline_config={
    "features": ["denoise", "upscale", "face-recovery"],
    "denoise_model": "NAFNet",
    "denoise_strength": 0.6,
    "upscale_model": "Real-ESRGAN",
    "upscale_scale": 4,
    "face_recovery_model": "CodeFormer",
    "face_fidelity_weight": 0.7,
    "device": "cuda"
  }'
```

### 2. Video Enhancement Pipeline

```bash
curl -X POST http://localhost:8000/api/processing/pipeline/file \
  -H "Content-Type: multipart/form-data" \
  -F "file=@input_video.mp4" \
  -F 'pipeline_config={
    "features": ["frame-interpolation", "slow-motion"],
    "frame_interp_model": "RIFE",
    "frame_interp_multiplier": 2,
    "slow_motion_model": "RIFE-SloMo",
    "slowdown_factor": 4.0,
    "device": "cuda"
  }'
```

### 3. Color Correction Pipeline

```bash
curl -X POST http://localhost:8000/api/processing/pipeline/file \
  -H "Content-Type: multipart/form-data" \
  -F "file=@input.jpg" \
  -F 'pipeline_config={
    "features": ["low-light", "match-color", "sharpen"],
    "low_light_model": "Zero-DCE",
    "low_light_strength": 0.8,
    "match_color_model": "Reinhard Transfer",
    "match_color_strength": 1.5,
    "match_color_preserve_luminance": true,
    "sharpen_model": "MPRNet",
    "sharpen_strength": 0.8,
    "device": "cuda"
  }'
```

## Pipeline Configuration Options

### Feature-Specific Parameters

#### Denoise Feature
```json
{
  "denoise_model": "SCUNet",  // or "NAFNet", "DeepScatter"
  "denoise_strength": 0.5     // 0.1 to 1.0 (higher = more aggressive)
}
```

#### Upscale Feature
```json
{
  "upscale_model": "Real-ESRGAN",  // or "RealBasicVSR", "Real-CUGAN"
  "upscale_scale": 4              // 2, 4, or 8
}
```

#### Face Recovery Feature
```json
{
  "face_recovery_model": "CodeFormer",  // or "GFPGAN", "RestoreFormer"
  "face_fidelity_weight": 0.7          // 0.1 to 1.0 (higher = more preservation)
}
```

#### Sharpen Feature
```json
{
  "sharpen_model": "NAFNet",           // or "MPRNet", "FFTformer"
  "sharpen_strength": 1.0,            // 0.1 to 5.0
  "sharpen_overlap": 128,            // 32 to 256
  "sharpen_unsharp_amount": 0.5      // 0.0 to 2.0
}
```

#### Frame Interpolation Feature
```json
{
  "frame_interp_model": "RIFE",       // or "RIFE", "FILM"
  "frame_interp_multiplier": 2      // 1 to 4 (2 = 60fps → 120fps)
}
```

#### Slow Motion Feature
```json
{
  "slow_motion_model": "RIFE-SloMo",  // or "FILM"
  "slowdown_factor": 4.0             // 1.0 to 10.0
}
```

#### Match Color Feature
```json
{
  "match_color_model": "Reinhard Transfer",  // or "WCT2", "NLUT"
  "match_color_strength": 1.0,              // 0.1 to 3.0
  "match_color_preserve_luminance": true   // boolean
}
```

#### Low Light Feature
```json
{
  "low_light_model": "Zero-DCE",       // or other models
  "low_light_strength": 0.7           // 0.1 to 1.0
}
```

### Global Pipeline Options

```json
{
  "features": ["feature1", "feature2"],
  "device": "cuda",                    // auto, cuda, cpu, metal, hip
  "output_path": "custom_output.jpg",   // optional custom output
  "enable_progress_tracking": true,
  "max_retries": 3,
  "skip_failed_features": true
}
```

## Feature Order and Dependencies

### Important Considerations

1. **Order Matters**: The order in the `features` array determines execution sequence
2. **Device Compatibility**: All pipelines use the same device for consistency
3. **Model Availability**: Ensure downloaded models match your configuration
4. **Memory Usage**: Chained pipelines use more VRAM - monitor GPU memory

### Recommended Feature Orders

#### Image Enhancement:
```
[Denoise] → [Upscale] → [Face Recovery] → [Sharpen]
```

#### Video Enhancement:
```
[Frame Interpolation] → [Slow Motion] → [Sharpen]
```

#### Color Enhancement:
```
[Low Light] → [Match Color] → [Sharpen]
```

#### Professional Pipeline:
```
[Denoise] → [Upscale] → [Sharpen] → [Color Match] → [Face Recovery]
```

## API Response Format

### Success Response
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

### Task Status Response
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "pipelines_executed": ["DenoisePipeline", "UpscalePipeline"],
  "current_pipeline": "UpscalePipeline"
}
```

### Final Result
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "pipelines_executed": ["DenoisePipeline", "UpscalePipeline", "FaceRecoveryPipeline"],
  "result": {
    "input_path": "input.jpg",
    "output_path": "output.jpg",
    "pipelines_executed": ["DenoisePipeline", "UpscalePipeline", "FaceRecoveryPipeline"],
    "last_pipeline_result": {
      "success": true,
      "input_path": "output_intermediate.jpg",
      "output_path": "output.jpg",
      "model": "CodeFormer"
    }
  }
}
```

### Error Response
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Model 'InvalidModel' not found. Please download a valid model first.",
  "progress": 0,
  "pipelines_executed": ["DenoisePipeline"]
}
```

## Development and Testing

### Running Tests

Run the pipeline chaining tests to ensure functionality:

```bash
# Run specific pipeline chain tests
pytest tests/test_pipeline_chaining.py -v

# Run all tests including integration tests
pytest tests/ -k "pipeline" -v
```

### Example Usage in Code

```python
from src.api.routes.processing import process_pipeline_file
from fastapi import UploadFile, Form
import tempfile

# Simulate API request
async def example_pipeline_processing():
    # Create a mock file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(b'fake_image_data')
        tmp_path = tmp.name
    
    # Create pipeline configuration
    config = {
        "features": ["denoise", "upscale", "face-recovery"],
        "denoise_model": "NAFNet",
        "denoise_strength": 0.6,
        "upscale_model": "Real-ESRGAN",
        "upscale_scale": 4,
        "face_recovery_model": "CodeFormer",
        "face_fidelity_weight": 0.7,
        "device": "cuda"
    }
    
    # Note: This would normally be called through FastAPI
    # result = await process_pipeline_file(
    #     file=UploadFile(filename="input.jpg", file=open(tmp_path, "rb")),
    #     pipeline_config=config,
    #     device="cuda"
    # )
    
    return "Pipeline processing would execute here"

# Example from Python code
if __name__ == "__main__":
    from examples.pipeline_chaining_example import main
    main()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Model not found"
```
Error: Model 'MyModel' not found. Please download a valid model first.
```
**Solution**: Check model availability using the unified registry:
```python
from src.models.unified_registry import get_models_by_feature
models = get_models_by_feature("Upscale")
available = [m["name"] for m in models]
```

#### 2. Device Not Supported
```
Error: Device 'cuda' is not available
```
**Solution**: Check system capabilities or use a different device:
```python
from src.api.routes.processing import _resolve_device
device = _resolve_device("cuda", {"computeBackend": "auto"})
```

#### 3. Pipeline Order Issues
```
Error: Cannot chain features in this order
```
**Solution**: Reorder features according to dependencies:
```python
# Denoise should come before Upscale
denoise_model = get_denoise_model("model_name")
```

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid pipeline configuration |
| 401 | Model not downloaded |
| 403 | Device not available |
| 503 | Processing modules not available |

## Migration Guide

### From Old API to Pipeline Chaining

#### Before (Single Feature)
```python
# Old approach - single feature
result = await process_upscale_file(file, model_name="Real-ESRGAN", scale=4)
```

#### After (Multi-Feature Pipeline)
```python
# New approach - pipeline chaining
pipeline_config = {
    "features": ["denoise", "upscale", "face-recovery"],
    "denoise_model": "NAFNet",
    "denoise_strength": 0.6,
    "upscale_model": "Real-ESRGAN", 
    "upscale_scale": 4,
    "face_recovery_model": "CodeFormer",
    "face_fidelity_weight": 0.7,
    "device": "cuda"
}

result = await process_pipeline_file(file, pipeline_config=pipeline_config)
```

## Best Practices

### 1. Start Simple
Begin with a two-feature pipeline (e.g., "denoise, upscale") and add features gradually.

### 2. Validate Configuration
Always validate your pipeline configuration before processing:

```python
from src.schemas.processing import PipelineRequest

# This will raise validation errors for invalid configs
try:
    config = PipelineRequest(**your_config_dict)
    print("Configuration is valid!")
except Exception as e:
    print(f"Invalid configuration: {e}")
```

### 3. Monitor Progress
Enable progress tracking for long-running pipeline chains:

```python
config["enable_progress_tracking"] = True
```

### 4. Handle Failures Gracefully
Use `skip_failed_features` to continue processing even if some stages fail:

```python
config["skip_failed_features"] = True
config["max_retries"] = 3
```

### 5. Test Locally First
Use the example scripts to test pipeline configurations before running in production.

## Conclusion

Pipeline chaining transforms openVideoAI from a single-feature processor into a comprehensive enhancement platform. It enables:

✅ **Complex workflows** with multiple processing stages  
✅ **Reusable configurations** for consistent results  
✅ **Flexible feature ordering** based on your specific needs  
✅ **Better error handling** with skip/failure recovery options  
✅ **Progress tracking** across all pipeline stages  
✅ **Catalog-driven model discovery** for easy model management  

The pipeline chaining feature makes openVideoAI suitable for professional production workflows while maintaining backward compatibility with existing single-feature processing.

---

*Last Updated: October 2026*
*Version: 1.0*
*Status: Production Ready*