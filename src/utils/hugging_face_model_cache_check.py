# model_management.py
"""
Utilities to check, download, and manage LiveKit turn detection models
"""

import os
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, cached_assets_path
from transformers import AutoTokenizer

class LiveKitModelManager:
    """Manage LiveKit turn detection models"""
    
    def __init__(self):
        self.HG_MODEL = "livekit/turn-detector"
        self.MODEL_REVISIONS = {
            "en": "v1.2.2-en",
            "multilingual": "v0.2.0-intl",
        }
        self.ONNX_FILENAME = "model_q8.onnx"
        
    def get_cache_directory(self):
        """Get the HuggingFace cache directory"""
        # Default HuggingFace cache location
        cache_dir = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        hub_dir = os.path.join(cache_dir, 'hub')
        return hub_dir
    
    def get_model_cache_path(self, model_type="multilingual"):
        """Get the specific cache path for a model"""
        cache_dir = self.get_cache_directory()
        
        # HuggingFace cache structure: models--{org}--{repo}/snapshots/{commit_hash}
        model_cache_name = f"models--{self.HG_MODEL.replace('/', '--')}"
        model_dir = os.path.join(cache_dir, model_cache_name)
        
        if os.path.exists(model_dir):
            # List all snapshots (commit hashes)
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                return model_dir, snapshots
        
        return model_dir, []
    
    def check_model_status(self, model_type="multilingual"):
        """Check if model is downloaded and get details"""
        print(f"Checking {model_type} model status...")
        print("-" * 50)
        
        revision = self.MODEL_REVISIONS[model_type]
        print(f"Model: {self.HG_MODEL}")
        print(f"Revision: {revision}")
        
        # Check cache directory
        cache_dir = self.get_cache_directory()
        print(f"Cache directory: {cache_dir}")
        
        model_dir, snapshots = self.get_model_cache_path(model_type)
        print(f"Model cache path: {model_dir}")
        
        if not snapshots:
            print("❌ Model not downloaded")
            return False
        
        print(f"✅ Found {len(snapshots)} snapshot(s):")
        
        # Check each snapshot
        for i, snapshot in enumerate(snapshots):
            snapshot_path = os.path.join(model_dir, "snapshots", snapshot)
            print(f"\n  Snapshot {i+1}: {snapshot}")
            print(f"  Path: {snapshot_path}")
            
            # Check what files are in this snapshot
            if os.path.exists(snapshot_path):
                files = os.listdir(snapshot_path)
                print(f"  Files: {len(files)} total")
                
                # Check for key files
                key_files = {
                    "tokenizer.json": "Tokenizer",
                    "config.json": "Model config",
                    "onnx/model_q8.onnx": "ONNX model"
                }
                
                for file_path, description in key_files.items():
                    full_path = os.path.join(snapshot_path, file_path)
                    if os.path.exists(full_path):
                        size = os.path.getsize(full_path)
                        size_mb = size / (1024 * 1024)
                        print(f"    ✅ {description}: {size_mb:.1f} MB")
                    else:
                        print(f"    ❌ {description}: Missing")
        
        return True
    
    def download_model(self, model_type="multilingual", force=False):
        """Download model if not present"""
        print(f"Downloading {model_type} model...")
        print("-" * 50)
        
        revision = self.MODEL_REVISIONS[model_type]
        
        try:
            # Download ONNX model
            print("📥 Downloading ONNX model...")
            onnx_path = hf_hub_download(
                self.HG_MODEL,
                self.ONNX_FILENAME,
                subfolder="onnx",
                revision=revision,
                local_files_only=False,
                force_download=force
            )
            print(f"✅ ONNX model: {onnx_path}")
            
            # Download tokenizer
            print("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.HG_MODEL,
                revision=revision,
                force_download=force
            )
            print("✅ Tokenizer downloaded")
            
            # Try to download language config (if available)
            try:
                print("📥 Downloading language config...")
                lang_path = hf_hub_download(
                    self.HG_MODEL,
                    "languages.json",
                    revision=revision,
                    local_files_only=False,
                    force_download=force
                )
                print(f"✅ Language config: {lang_path}")
            except:
                print("⚠️  Language config not available for this revision")
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def clean_cache(self, model_type=None):
        """Clean model cache"""
        print("🧹 Cleaning model cache...")
        
        if model_type:
            print(f"Cleaning {model_type} model only")
            # TODO: Implement selective cleaning
            print("Selective cleaning not implemented yet")
        else:
            print("Cleaning all cached models")
            cache_dir = self.get_cache_directory()
            model_dir, _ = self.get_model_cache_path()
            
            if os.path.exists(model_dir):
                import shutil
                try:
                    shutil.rmtree(model_dir)
                    print(f"✅ Removed: {model_dir}")
                except Exception as e:
                    print(f"❌ Failed to remove: {e}")
            else:
                print("No cache to clean")
    
    def list_all_cached_models(self):
        """List all cached HuggingFace models"""
        print("📂 All cached HuggingFace models:")
        print("-" * 50)
        
        cache_dir = self.get_cache_directory()
        
        if not os.path.exists(cache_dir):
            print("No cache directory found")
            return
        
        model_dirs = [d for d in os.listdir(cache_dir) if d.startswith('models--')]
        
        if not model_dirs:
            print("No models cached")
            return
        
        total_size = 0
        for model_dir in sorted(model_dirs):
            model_path = os.path.join(cache_dir, model_dir)
            
            # Convert cache name back to model name
            model_name = model_dir.replace('models--', '').replace('--', '/')
            
            # Calculate size
            size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    size += os.path.getsize(os.path.join(root, file))
            
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            
            print(f"  📦 {model_name}: {size_mb:.1f} MB")
        
        print(f"\nTotal cache size: {total_size:.1f} MB")

def check_model():
    """Main function to demonstrate model management"""
    manager = LiveKitModelManager()
    
    print("LiveKit Turn Detection Model Manager")
    print("=" * 60)
    
    # Check both models
    for model_type in ["multilingual", "en"]:
        print(f"\n{'='*20} {model_type.upper()} MODEL {'='*20}")
        
        # Check if downloaded
        is_downloaded = manager.check_model_status(model_type)
        
        if not is_downloaded:
            print(f"\n📥 {model_type} model not found. Download? (y/n): ", end="")
            response = input().lower().strip()
            
            if response == 'y':
                success = manager.download_model(model_type)
                if success:
                    print(f"✅ {model_type} model ready!")
                else:
                    print(f"❌ Failed to download {model_type} model")
    
    # Show all cached models
    print(f"\n{'='*60}")
    manager.list_all_cached_models()
    
    print(f"\n{'='*60}")
    print("Model management complete!")
    print("\nUseful commands:")
    print("  - Check status: manager.check_model_status('multilingual')")
    print("  - Download: manager.download_model('multilingual')")
    print("  - Force re-download: manager.download_model('multilingual', force=True)")
    print("  - Clean cache: manager.clean_cache()")

# if __name__ == "__main__":
#     main()