"""
Model Version Verification Script

This script verifies the scikit-learn and joblib versions used to save
the Loan_RandomForest_model.pkl file and checks compatibility with
current environment.
"""

import os
import pickle
import joblib
import sklearn
import warnings
import glob
import sys
from typing import Dict, Optional, Any, List


def find_model_file(base_filename: str = "Loan_RandomForest_model.pkl") -> Optional[str]:
    """
    Search for the model file in various possible locations.
    
    Args:
        base_filename (str): Base filename to search for
        
    Returns:
        Optional[str]: Path to the model file if found, None otherwise
    """
    # Possible directories where model might be stored
    search_paths = [
        ".",  # Current directory
        "model",
        "model/loanApproval",
        "models",
        "saved_models", 
        "artifacts",
        "output",
        "exports"
    ]
    
    # Possible filename patterns
    filename_patterns = [
        base_filename,
        "*.pkl",
        "*RandomForest*.pkl",
        "*loan*.pkl",
        "*model*.pkl"
    ]
    
    print("Searching for model file...")
    print("-" * 30)
    
    # Search in each directory
    for search_dir in search_paths:
        if os.path.exists(search_dir):
            print(f"Checking directory: {search_dir}")
            
            # Check for exact filename match first
            exact_path = os.path.join(search_dir, base_filename)
            if os.path.exists(exact_path):
                print(f"✓ Found exact match: {exact_path}")
                return exact_path
            
            # Search for pattern matches
            for pattern in filename_patterns:
                search_pattern = os.path.join(search_dir, pattern)
                matches = glob.glob(search_pattern)
                if matches:
                    print(f"✓ Found pattern match: {matches[0]}")
                    return matches[0]
            
            print(f"  No model files found in {search_dir}")
        else:
            print(f"  Directory does not exist: {search_dir}")
    
    return None


def list_all_pkl_files() -> List[str]:
    """
    List all .pkl files in the project directory for debugging.
    
    Returns:
        List[str]: List of all .pkl file paths found
    """
    pkl_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def verify_model_versions(model_path: str) -> Dict[str, Any]:
    """
    Verify the versions of libraries used to save a pickle model file.
    
    Args:
        model_path (str): Path to the saved model pickle file
        
    Returns:
        Dict[str, Any]: Dictionary containing version information and model details
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model cannot be loaded
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Verifying model: {model_path}")
    print("=" * 50)
    
    # File information
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size/1024:.1f} KB")
    
    # Current environment versions
    current_sklearn_version = sklearn.__version__
    current_joblib_version = joblib.__version__
    current_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    print(f"Current Python version: {current_python_version}")
    print(f"Current scikit-learn version: {current_sklearn_version}")
    print(f"Current joblib version: {current_joblib_version}")
    print("-" * 30)
    
    # Try multiple loading methods
    loading_attempts = []
    model = None
    successful_method = None
    
    # Method 1: joblib (default)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
        loading_attempts.append(("joblib (default)", "SUCCESS"))
        successful_method = "joblib (default)"
        print("✓ Model loaded successfully with joblib (default)")
        
    except Exception as e:
        loading_attempts.append(("joblib (default)", f"FAILED: {str(e)}"))
        print(f"✗ joblib (default) failed: {e}")
    
    # Method 2: pickle
    if model is None:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            loading_attempts.append(("pickle (rb)", "SUCCESS"))
            successful_method = "pickle (rb)"
            print("✓ Model loaded successfully with pickle")
            
        except Exception as e:
            loading_attempts.append(("pickle (rb)", f"FAILED: {str(e)}"))
            print(f"✗ pickle (rb) failed: {e}")
    
    # Method 3: joblib with mmap_mode=None
    if model is None:
        try:
            model = joblib.load(model_path, mmap_mode=None)
            loading_attempts.append(("joblib (mmap_mode=None)", "SUCCESS"))
            successful_method = "joblib (mmap_mode=None)"
            print("✓ Model loaded successfully with joblib (mmap_mode=None)")
            
        except Exception as e:
            loading_attempts.append(("joblib (mmap_mode=None)", f"FAILED: {str(e)}"))
            print(f"✗ joblib (mmap_mode=None) failed: {e}")
    
    if model is None:
        raise Exception("Cannot load model with any available method")
    
    # Extract model information
    model_info = {
        'loading_method': successful_method,
        'model_type': type(model).__name__,
        'current_sklearn_version': current_sklearn_version,
        'current_joblib_version': current_joblib_version,
        'current_python_version': current_python_version,
        'loading_attempts': loading_attempts,
        'file_size_kb': file_size/1024,
    }
    
    # Try to extract version information from model
    sklearn_version_used = getattr(model, '_sklearn_version', 'Unknown')
    
    print(f"Model type: {model_info['model_type']}")
    print(f"Scikit-learn version used for training: {sklearn_version_used}")
    
    model_info['sklearn_version_used'] = sklearn_version_used
    
    # Check version compatibility
    if sklearn_version_used != 'Unknown':
        if sklearn_version_used == current_sklearn_version:
            print("✓ Scikit-learn versions match - Full compatibility")
            compatibility_status = "Compatible"
        else:
            print(f"⚠ Version mismatch detected!")
            print(f"  Training version: {sklearn_version_used}")
            print(f"  Current version: {current_sklearn_version}")
            compatibility_status = "Version Mismatch"
    else:
        print("⚠ Cannot determine training version - Potential compatibility issues")
        compatibility_status = "Unknown"
    
    model_info['compatibility_status'] = compatibility_status
    
    # Additional model details if available
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
        model_info['n_features'] = model.n_features_in_
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names available: Yes ({len(model.feature_names_in_)} features)")
        model_info['has_feature_names'] = True
        model_info['feature_names'] = model.feature_names_in_.tolist()
    else:
        print("Feature names available: No")
        model_info['has_feature_names'] = False
    
    # Check if it's a Pipeline and extract components
    if hasattr(model, 'steps'):
        print(f"Pipeline detected with {len(model.steps)} steps:")
        for i, (step_name, step_obj) in enumerate(model.steps):
            print(f"  Step {i+1}: {step_name} -> {type(step_obj).__name__}")
        model_info['is_pipeline'] = True
        model_info['pipeline_steps'] = [(name, type(obj).__name__) for name, obj in model.steps]
    else:
        model_info['is_pipeline'] = False
    
    return model_info


def check_model_compatibility(model_path: str) -> bool:
    """
    Quick compatibility check for the model.
    
    Args:
        model_path (str): Path to the saved model pickle file
        
    Returns:
        bool: True if model is compatible, False otherwise
    """
    try:
        model_info = verify_model_versions(model_path)
        return model_info['compatibility_status'] == "Compatible"
    except Exception as e:
        print(f"Compatibility check failed: {e}")
        return False


if __name__ == "__main__":
    # Check if path provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find the model file automatically
        model_path = find_model_file()
    
    if not model_path:
        print("=" * 60)
        print("MODEL FILE NOT FOUND")
        print("=" * 60)
        
        # List all .pkl files for debugging
        pkl_files = list_all_pkl_files()
        if pkl_files:
            print("Found the following .pkl files in the project:")
            for i, file_path in enumerate(pkl_files, 1):
                print(f"{i}. {file_path}")
            
            print("\nYou can manually specify the model path by running:")
            print("python verify_model_versions.py <path_to_model>")
        else:
            print("No .pkl files found in the project directory.")
            print("Please ensure the model file exists and has been saved properly.")
        
        print("\nSearched in the following directories:")
        search_dirs = [".", "model", "model/loanApproval", "models", "saved_models", "artifacts", "output", "exports"]
        for directory in search_dirs:
            status = "✓ EXISTS" if os.path.exists(directory) else "✗ NOT FOUND"
            print(f"- {directory:<20} {status}")
        
        exit(1)
    
    try:
        # Verify model versions
        model_info = verify_model_versions(model_path)
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Model Path: {model_path}")
        print(f"Model Status: {model_info['compatibility_status']}")
        print(f"Loading Method: {model_info['loading_method']}")
        print(f"Model Type: {model_info['model_type']}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if model_info['compatibility_status'] == "Version Mismatch":
            print("- Consider retraining the model with current scikit-learn version")
            print("- Or install the specific scikit-learn version used for training")
            print("- Test model predictions thoroughly before deployment")
        elif model_info['compatibility_status'] == "Compatible":
            print("- Model is compatible with current environment")
            print("- Safe to use for predictions")
        else:
            print("- Test model thoroughly before use")
            print("- Consider retraining with current versions")
            
    except Exception as e:
        print(f"Error during verification: {e}")
        exit(1)
