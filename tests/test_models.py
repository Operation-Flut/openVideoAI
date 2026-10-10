#!/usr/bin/env python
"""
Test script to verify the model downloader and initialization logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (  # pyright: ignore[reportAttributeAccessIssue]
    initialize_models,
    get_model_status,
    get_all_models,
    get_downloaded_models,
    get_available_models,
    is_model_downloaded,
    get_model_by_name,
    download_model
)

def test_initialization():
    """Test model initialization"""
    print("=" * 60)
    print("Testing Model Initialization")
    print("=" * 60)
    
    def progress_cb(msg, progress):
        print(f"  [{progress*100:.0f}%] {msg}")
    
    result = initialize_models(progress_cb)
    print(f"\nInitialization Result:")
    print(f"  Total models: {result['total_models']}")
    print(f"  Downloaded: {result['downloaded']}")
    print(f"  Available: {result['available']}")
    
    return result

def test_model_catalog():
    """Test model catalog loading"""
    print("\n" + "=" * 60)
    print("Testing Model Catalog")
    print("=" * 60)
    
    models = get_all_models()
    print(f"Total models in catalog: {len(models)}")
    
    for model in models:
        status = "[OK]" if is_model_downloaded(model.name) else "[..]"
        print(f"  {status} {model.name} ({model.size}, {model.weight})")
        print(f"      {model.description_short}")
        print(f"      Features: {', '.join(f['name'] for f in model.features)}")
        print(f"      Download URL: {model.download_url}")
        print()

def test_status():
    """Test status API"""
    print("=" * 60)
    print("Testing Status API")
    print("=" * 60)
    
    status = get_model_status()
    print(f"Total: {status['total']}")
    print(f"Downloaded: {status['downloaded']}")
    print(f"Available: {status['available']}")
    
    for m in status['models']:
        dl_status = "[OK]" if m['downloaded'] else "[..]"
        print(f"  {dl_status} {m['name']} - {m['size']} - {m['weight']}")

def test_download_simulation():
    """Test download logic (without actually downloading)"""
    print("\n" + "=" * 60)
    print("Testing Download Logic (Simulation)")
    print("=" * 60)
    
    available = get_available_models()
    if available:
        model = available[0]
        print(f"First available model: {model.name}")
        print(f"  Download URL: {model.download_url}")
        print(f"  Filename: {model.filename}")
        print(f"  Local path: {model.local_path}")
    else:
        print("All models already downloaded!")

if __name__ == "__main__":
    print("openVideoAI Model System Test\n")
    
    try:
        test_initialization()
        test_model_catalog()
        test_status()
        test_download_simulation()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)