"""
Utility functions for running code on both local and Kaggle environments
"""
import os
import sys


def detect_environment():
    """
    Detect if code is running on Kaggle or locally
    
    Returns:
        dict: Environment configuration with paths
    """
    is_kaggle = os.path.exists('/kaggle/input')
    
    if is_kaggle:
        # Kaggle environment
        config = {
            'is_kaggle': True,
            'base_dir': '/kaggle/working',
            'data_base_dir': '/kaggle/input',
            'output_dir': '/kaggle/working',
            'cache_dir': '/kaggle/working/cache',
            'ckpt_dir': '/kaggle/working/ckpts',
            'results_dir': '/kaggle/working/results'
        }
    else:
        # Local environment
        config = {
            'is_kaggle': False,
            'base_dir': os.getcwd(),
            'data_base_dir': 'train/datasets',
            'output_dir': '.',
            'cache_dir': 'train/datasets',  # Use dataset dir for cache locally
            'ckpt_dir': 'train/ckpts',
            'results_dir': 'train/results'
        }
    
    # Create necessary directories
    os.makedirs(config['cache_dir'], exist_ok=True)
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    return config


def get_dataset_path(dataset_name, env_config=None):
    """
    Get the correct dataset path based on environment
    
    Args:
        dataset_name: Name of dataset (e.g., 'ASSIST2009', 'Algebra2005')
        env_config: Environment configuration (if None, will auto-detect)
    
    Returns:
        str: Full path to dataset directory
    """
    if env_config is None:
        env_config = detect_environment()
    
    if env_config['is_kaggle']:
        # On Kaggle, look for dataset in input directory
        # Try common naming patterns
        possible_paths = [
            f"/kaggle/input/kt-datasets/{dataset_name}",
            f"/kaggle/input/knowledge-tracing-datasets/{dataset_name}",
            f"/kaggle/input/{dataset_name.lower()}",
            f"/kaggle/input/datasets/{dataset_name}",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found, list available datasets
        print("Dataset not found. Available datasets:")
        if os.path.exists('/kaggle/input'):
            for item in os.listdir('/kaggle/input'):
                print(f"  - /kaggle/input/{item}")
        
        raise FileNotFoundError(f"Dataset {dataset_name} not found in Kaggle input directory")
    else:
        # Local environment
        return os.path.join('train/datasets', dataset_name)


def get_cache_path(dataset_name, filename, env_config=None):
    """
    Get path for cached/preprocessed files
    
    Args:
        dataset_name: Name of dataset
        filename: Name of cache file (e.g., 'q_seqs.pkl')
        env_config: Environment configuration
    
    Returns:
        str: Full path to cache file
    """
    if env_config is None:
        env_config = detect_environment()
    
    cache_dir = os.path.join(env_config['cache_dir'], dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, filename)


def setup_kaggle_paths():
    """
    Setup Python paths for Kaggle environment
    Adds necessary directories to sys.path
    """
    if os.path.exists('/kaggle'):
        paths_to_add = [
            '/kaggle/working',
            '/kaggle/working/train',
            '/kaggle/working/train/models',
            '/kaggle/working/data_loaders',
        ]
        
        for path in paths_to_add:
            if path not in sys.path and os.path.exists(path):
                sys.path.insert(0, path)


def print_environment_info():
    """Print information about current environment"""
    config = detect_environment()
    
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Running on: {'Kaggle' if config['is_kaggle'] else 'Local'}")
    print(f"Base directory: {config['base_dir']}")
    print(f"Data directory: {config['data_base_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Cache directory: {config['cache_dir']}")
    print(f"Checkpoint directory: {config['ckpt_dir']}")
    print(f"Results directory: {config['results_dir']}")
    
    if config['is_kaggle']:
        print("\nAvailable Kaggle datasets:")
        if os.path.exists('/kaggle/input'):
            for item in os.listdir('/kaggle/input'):
                print(f"  - {item}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the utilities
    print_environment_info()
    
    # Test dataset path detection
    try:
        path = get_dataset_path('ASSIST2009')
        print(f"\nASSIST2009 dataset path: {path}")
        print(f"Exists: {os.path.exists(path)}")
    except Exception as e:
        print(f"\nError finding dataset: {e}")
