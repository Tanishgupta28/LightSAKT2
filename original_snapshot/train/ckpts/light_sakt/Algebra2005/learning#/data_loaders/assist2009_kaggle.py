"""
ASSIST2009 Dataset Loader - Kaggle Compatible Version
Works on both local and Kaggle environments
"""
import os
import pickle
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from models.utils import match_seq_len

# Import kaggle utilities
try:
    from kaggle_utils import detect_environment, get_dataset_path, get_cache_path
except ImportError:
    # Fallback if kaggle_utils not available
    def detect_environment():
        return {'is_kaggle': False, 'cache_dir': 'train/datasets'}
    
    def get_dataset_path(name, config=None):
        return f"train/datasets/{name}"
    
    def get_cache_path(dataset, filename, config=None):
        return os.path.join(f"train/datasets/{dataset}", filename)


class ASSIST2009(Dataset):
    """
    ASSIST2009 Dataset Loader
    
    Automatically detects environment (local vs Kaggle) and adjusts paths accordingly.
    Handles read-only dataset directories on Kaggle by caching preprocessed files
    in the working directory.
    """
    
    def __init__(self, seq_len, dataset_dir=None, env_config=None):
        """
        Initialize ASSIST2009 dataset
        
        Args:
            seq_len: Sequence length for padding/truncating
            dataset_dir: Optional custom dataset directory path
            env_config: Optional environment configuration (auto-detected if None)
        """
        super().__init__()
        
        # Detect environment
        if env_config is None:
            self.env_config = detect_environment()
        else:
            self.env_config = env_config
        
        # Set dataset directory
        if dataset_dir is None:
            self.dataset_dir = get_dataset_path('ASSIST2009', self.env_config)
        else:
            self.dataset_dir = dataset_dir
        
        # Set dataset file path
        self.dataset_path = os.path.join(self.dataset_dir, "skill_builder_data.csv")
        
        # Verify dataset exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}\n"
                f"Please ensure the dataset is uploaded to Kaggle or available locally."
            )
        
        print(f"Loading ASSIST2009 from: {self.dataset_dir}")
        
        # Load or preprocess data
        self._load_or_preprocess()
        
        # Set dataset properties
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        
        # Match sequence length if specified
        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(self.q_seqs, self.r_seqs, seq_len)
        
        self.len = len(self.q_seqs)
        
        print(f"Dataset loaded: {self.len} sequences, {self.num_q} skills, {self.num_u} users")
    
    def _load_or_preprocess(self):
        """Load preprocessed data or preprocess from raw CSV"""
        
        # Try to load from cache first
        cache_files = {
            'q_seqs': get_cache_path('ASSIST2009', 'q_seqs.pkl', self.env_config),
            'r_seqs': get_cache_path('ASSIST2009', 'r_seqs.pkl', self.env_config),
            'q_list': get_cache_path('ASSIST2009', 'q_list.pkl', self.env_config),
            'u_list': get_cache_path('ASSIST2009', 'u_list.pkl', self.env_config),
            'q2idx': get_cache_path('ASSIST2009', 'q2idx.pkl', self.env_config),
            'u2idx': get_cache_path('ASSIST2009', 'u2idx.pkl', self.env_config),
        }
        
        # Check if all cache files exist
        all_cached = all(os.path.exists(path) for path in cache_files.values())
        
        if all_cached:
            print("Loading preprocessed data from cache...")
            with open(cache_files['q_seqs'], 'rb') as f:
                self.q_seqs = pickle.load(f)
            with open(cache_files['r_seqs'], 'rb') as f:
                self.r_seqs = pickle.load(f)
            with open(cache_files['q_list'], 'rb') as f:
                self.q_list = pickle.load(f)
            with open(cache_files['u_list'], 'rb') as f:
                self.u_list = pickle.load(f)
            with open(cache_files['q2idx'], 'rb') as f:
                self.q2idx = pickle.load(f)
            with open(cache_files['u2idx'], 'rb') as f:
                self.u2idx = pickle.load(f)
        else:
            print("Preprocessing data from CSV...")
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx = \
                self._preprocess_and_cache(cache_files)
    
    def _preprocess_and_cache(self, cache_files):
        """Preprocess raw data and save to cache"""
        
        # Load and preprocess CSV
        print(f"Reading CSV from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path, encoding='latin1')
        df = df.dropna(subset=["skill_name"])
        df = df.drop_duplicates(subset=["order_id", "skill_name"])
        df = df.sort_values(by=["order_id"])
        
        # Get unique users and skills
        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)
        
        print(f"Found {len(u_list)} users and {len(q_list)} skills")
        
        # Create mappings
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        
        # Create sequences
        q_seqs = []
        r_seqs = []
        
        print("Creating sequences...")
        for u in u_list:
            df_u = df[df["user_id"] == u]
            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])
            r_seq = df_u["correct"].values
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        
        # Save to cache
        print(f"Saving preprocessed data to cache...")
        with open(cache_files['q_seqs'], 'wb') as f:
            pickle.dump(q_seqs, f)
        with open(cache_files['r_seqs'], 'wb') as f:
            pickle.dump(r_seqs, f)
        with open(cache_files['q_list'], 'wb') as f:
            pickle.dump(q_list, f)
        with open(cache_files['u_list'], 'wb') as f:
            pickle.dump(u_list, f)
        with open(cache_files['q2idx'], 'wb') as f:
            pickle.dump(q2idx, f)
        with open(cache_files['u2idx'], 'wb') as f:
            pickle.dump(u2idx, f)
        
        print("Preprocessing complete!")
        
        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx
    
    def __getitem__(self, index):
        """Get item by index"""
        return self.q_seqs[index], self.r_seqs[index]
    
    def __len__(self):
        """Get dataset length"""
        return self.len


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing ASSIST2009 Dataset Loader")
    print("=" * 60)
    
    try:
        dataset = ASSIST2009(seq_len=100)
        print(f"\nDataset successfully loaded!")
        print(f"Number of sequences: {len(dataset)}")
        print(f"Number of skills: {dataset.num_q}")
        print(f"Number of users: {dataset.num_u}")
        
        # Test getting an item
        q_seq, r_seq = dataset[0]
        print(f"\nSample sequence:")
        print(f"Question sequence shape: {q_seq.shape}")
        print(f"Response sequence shape: {r_seq.shape}")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
