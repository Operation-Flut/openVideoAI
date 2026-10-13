"""
Example demonstrating pipeline chaining in openVideoAI

This example shows how to create complex processing workflows
using the new PipelineChain functionality.
"""

from src.processing import PipelineChain, UpscalePipeline, FaceRecoveryPipeline, DenoisePipeline
from src.models.unified_registry import get_models_by_feature
import tempfile
from pathlib import Path
def create_example_pipelines():
    """
    Create example pipelines with different configurations
    """
    print("Creating example pipelines...")
    
    # Get available models from the unified registry
    upscale_models = get_models_by_feature("Upscale")
    face_models = get_models_by_feature("Face Recovery")
    denoise_models = get_models_by_feature("Denoise")
    
    print(f"Available upscale models: {[m['name'] for m in upscale_models]}")
    print(f"Available face recovery models: {[m['name'] for m in face_models]}")
    print(f"Available denoise models: {[m['name'] for m in denoise_models]}")
    
    # Create pipelines with example configurations
    # Example 1: Basic pipeline (Denoise -> Upscale -> Face Recovery)
    denoise_pipeline = DenoisePipeline(
        model_name="NAFNet",
        device="cuda",  # or "cpu", "auto", etc.
        strength=0.6
    )
    
    upscale_pipeline = UpscalePipeline(
        model_name="Real-ESRGAN",
        device="cuda",
        scale=4
    )
    
    face_pipeline = FaceRecoveryPipeline(
        model_name="CodeFormer",
        device="cuda",
        fidelity_weight=0.7
    )
    
    return [denoise_pipeline, upscale_pipeline, face_pipeline]
def demonstrate_pipeline_chaining(input_file, output_file):
    """
    Demonstrate pipeline chaining with example configurations
    """
    print(f"\n=== Pipeline Chaining Example ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Create example pipeline chain
    pipelines = create_example_pipelines()
    
    # Create pipeline chain
    chain = PipelineChain(pipelines)
    
    print(f"\nCreated pipeline chain with {len(pipelines)} stages:")
    for i, pipeline in enumerate(pipelines):
        print(f"  {i+1}. {pipeline.__class__.__name__}")
        print(f"     Model: {pipeline.model_name}")
        print(f"     Device: {pipeline.device}")
    
    # In a real scenario, this would process the actual files
    print("\n=== Pipeline Chain Configuration ===")
    print("This demonstrates:")
    print("1. Sequential processing: Denoise → Upscale → Face Recovery")
    print("2. Device selection: CUDA acceleration for all stages")
    print("3. Model selection: Using catalog-driven model discovery")
    print("4. Strength tuning: Configurable parameters for each stage")
    
    # Show available features
    print("\n=== Available Processing Features ===")
    features = ["Denoise", "Upscale", "Face Recovery", "Sharpen", 
                "Frame Interpolation", "Slow Motion", "Match Color", 
                "Low Light", "Log Restoration"]
    
    for feature in features:
        models = get_models_by_feature(feature)
        print(f"{feature}: {len(models)} models available")
        if models:
            print(f"  Examples: {[m['name'] for m in models[:2]]}")
    
    print("\n=== Pipeline Chain Benefits ===")
    print("✓ Complex multi-stage workflows")
    print("✓ Reusable pipeline configurations")
    print("✓ Easy parameter management")
    print("✓ Progress tracking across all stages")
    print("✓ Error handling and recovery")
    print("✓ Catalog-driven model discovery")
    
    return chain
def show_example_pipeline_requests():
    """
    Show example API requests for pipeline chaining
    """
    print("\n=== Example Pipeline API Requests ===")
    
    # Example 1: Basic pipeline
    pipeline_request_1 = {
        "features": ["denoise", "upscale", "face-recovery"],
        "denoise_model": "NAFNet",
        "denoise_strength": 0.6,
        "upscale_model": "Real-ESRGAN", 
        "upscale_scale": 4,
        "face_recovery_model": "CodeFormer",
        "face_fidelity_weight": 0.7,
        "device": "cuda"
    }
    
    # Example 2: Video processing pipeline
    pipeline_request_2 = {
        "features": ["frame-interpolation", "slow-motion", "sharpen"],
        "frame_interp_model": "RIFE",
        "frame_interp_multiplier": 2,
        "slow_motion_model": "RIFE-SloMo",
        "slowdown_factor": 4.0,
        "sharpen_model": "NAFNet",
        "sharpen_strength": 1.2,
        "device": "cuda"
    }
    
    # Example 3: Color correction pipeline
    pipeline_request_3 = {
        "features": ["low-light", "match-color", "sharpen"],
        "low_light_model": "Zero-DCE",
        "low_light_strength": 0.8,
        "match_color_model": "Reinhard Transfer",
        "match_color_strength": 1.5,
        "match_color_preserve_luminance": True,
        "sharpen_model": "MPRNet",
        "sharpen_strength": 0.8,
        "device": "cuda"
    }
    
    print("\nRequest 1 - Image Enhancement Pipeline:")
    print(f"  Features: {pipeline_request_1['features']}")
    print(f"  Denoise: {pipeline_request_1['denoise_model']} (strength: {pipeline_request_1['denoise_strength']})")
    print(f"  Upscale: {pipeline_request_1['upscale_model']} (scale: {pipeline_request_1['upscale_scale']}x)")
    print(f"  Face Recovery: {pipeline_request_1['face_recovery_model']} (fidelity: {pipeline_request_1['face_fidelity_weight']})")
    
    print("\nRequest 2 - Video Enhancement Pipeline:")
    print(f"  Features: {pipeline_request_2['features']}")
    print(f"  Frame Interpolation: {pipeline_request_2['frame_interp_model']} (multiplier: {pipeline_request_2['frame_interp_multiplier']}x)")
    print(f"  Slow Motion: {pipeline_request_2['slow_motion_model']} (factor: {pipeline_request_2['slowdown_factor']}x)")
    print(f"  Sharpen: {pipeline_request_2['sharpen_model']} (strength: {pipeline_request_2['sharpen_strength']})")
    
    print("\nRequest 3 - Color Enhancement Pipeline:")
    print(f"  Features: {pipeline_request_3['features']}")
    print(f"  Low Light: {pipeline_request_3['low_light_model']} (strength: {pipeline_request_3['low_light_strength']})")
    print(f"  Match Color: {pipeline_request_3['match_color_model']} (strength: {pipeline_request_3['match_color_strength']})")
    print(f"  Sharpen: {pipeline_request_3['sharpen_model']} (strength: {pipeline_request_3['sharpen_strength']})")
if __name__ == "__main__":
    print("openVideoAI Pipeline Chaining Examples")
    print("=" * 50)
    
    # Create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        example_file = tmp_file.name
    
    try:
        # Create pipeline chain
        chain = demonstrate_pipeline_chaining(example_file, "enhanced_output.jpg")
        
        # Show API request examples
        show_example_pipeline_requests()
        
        print(f"\n=== Example Complete ===")
        print(f"Pipeline chaining demonstration finished successfully!")
        print(f"Example files: {example_file}")
        
    except Exception as e:
        print(f"Error in example: {e}")
    finally:
        # Clean up example file
        Path(example_file).unlink(missing_ok=True)

# Additional helper functions for advanced usage
def build_pipeline_from_configuration(config):
    """
    Helper function to build a pipeline from a configuration dictionary
    
    Args:
        config: Dictionary containing pipeline configuration
        
    Returns:
        PipelineChain instance
    """
    from src.processing import PipelineChain
    
    # Map feature names to pipeline classes
    pipeline_classes = {
        "upscale": UpscalePipeline,
        "face-recovery": FaceRecoveryPipeline,
        "sharpen": SharpenPipeline,
        "denoise": DenoisePipeline,
        "frame-interpolation": FrameInterpolationPipeline,
        "slow-motion": SlowMotionPipeline,
        "match-color": MatchColorPipeline,
        "low-light": LowLightEnhancementPipeline,
    }
    
    pipelines = []
    
    for feature in config.get("features", []):
        if feature in pipeline_classes:
            pipeline_class = pipeline_classes[feature]
            kwargs = {k: v for k, v in config.items() 
                     if k.startswith(f"{feature}_") or k in 
                     ["device", "output_path", "scale", "strength", 
                      "overlap", "multiplier", "fidelity_weight", 
                      "slowdown_factor", "preserve_luminance", 
                      "engine", "preset"]}
            
            # Clean up kwargs
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            pipeline = pipeline_class(**kwargs)
            pipelines.append(pipeline)
    
    return PipelineChain(pipelines) if pipelines else None
def validate_pipeline_configuration(config):
    """
    Validate pipeline configuration
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if "features" not in config:
        errors.append("Missing required field: 'features'")
    elif not config["features"]:
        errors.append("Features list cannot be empty")
    
    # Check feature validity
    valid_features = ["upscale", "face-recovery", "sharpen", "denoise",
                      "frame-interpolation", "slow-motion", "match-color",
                      "low-light", "log-restoration", "sdr-to-hdr",
                      "object-removal", "focus-fix", "overexposure-fix",
                      "flat-profile", "reencode"]
    
    for feature in config.get("features", []):
        if feature not in valid_features:
            errors.append(f"Invalid feature: '{feature}'")
    
    # Check device validity
    if "device" in config and config["device"] not in ["auto", "cpu", "cuda", "metal", "hip", "amd", "dml"]:
        errors.append(f"Invalid device: '{config['device']}'")
    
    # Check strength values if present
    for feature in config.get("features", []):
        strength_fields = {
            "sharpen": "sharpen_strength",
            "denoise": "denoise_strength", 
            "match-color": "match_color_strength",
            "low-light": "low_light_strength",
            "face-recovery": "face_fidelity_weight"
        }
        
        if feature in strength_fields:
            field = strength_fields[feature]
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"Invalid {field} type: must be numeric")
    
    return errors