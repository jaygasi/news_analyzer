"""
Optimized file utilities with improved error handling and type safety
"""
import os
import glob
import shutil
from pathlib import Path
from typing import Optional, List, Union, Any
import pandas as pd

from utils.log_utils import logd, logw, loge


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        loge(f"Failed to create directory {path}: {e}")
        raise


def create_output_directories(directories: Optional[List[Union[str, Path]]] = None) -> None:
    """
    Create multiple output directories with improved error handling.
    
    Args:
        directories: List of directories to create. If None, uses default config directories.
    """
    if directories is None:
        # Import here to avoid circular imports
        try:
            from config import CACHE_DIR, LOG_DIR, RESULTS_DIR
            directories = [CACHE_DIR, LOG_DIR, RESULTS_DIR]
        except ImportError:
            logw("Could not import default directories from config")
            return
    
    created_count = 0
    failed_count = 0
    
    for directory in directories:
        try:
            ensure_directory_exists(directory)
            created_count += 1
        except OSError:
            failed_count += 1
    
    if created_count > 0:
        logd(f"Created/verified {created_count} directories")
    if failed_count > 0:
        logw(f"Failed to create {failed_count} directories")


def load_csv(directory: Union[str, Path], 
             file_name: str,
             encoding: str = 'utf-8',
             **kwargs) -> Optional[pd.DataFrame]:
    """
    Load CSV file with improved error handling and validation.
    
    Args:
        directory: Directory containing the file
        file_name: Name of the CSV file
        encoding: File encoding
        **kwargs: Additional pandas read_csv arguments
        
    Returns:
        DataFrame or None if loading failed
    """
    if not file_name:
        loge("File name cannot be empty")
        return None
    
    path = Path(directory) / file_name
    
    if not path.exists():
        logd(f"CSV file not found: {path}")
        return None
    
    if not path.is_file():
        loge(f"Path is not a file: {path}")
        return None
    
    try:
        # Check file size before loading
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # 500MB limit
            logw(f"Large CSV file detected: {file_size_mb:.1f}MB")
        
        df = pd.read_csv(path, encoding=encoding, **kwargs)
        
        if df.empty:
            logw(f"Loaded empty CSV file: {path}")
        else:
            logd(f"Loaded CSV with {len(df)} rows from {path}")
        
        return df
        
    except pd.errors.EmptyDataError:
        logw(f"CSV file is empty: {path}")
        return None
    except pd.errors.ParserError as e:
        loge(f"CSV parsing error for {path}: {e}")
        return None
    except UnicodeDecodeError as e:
        loge(f"Encoding error for {path}: {e}")
        # Try with different encoding
        try:
            df = pd.read_csv(path, encoding='latin-1', **kwargs)
            logw(f"Loaded CSV with latin-1 encoding: {path}")
            return df
        except Exception:
            loge(f"Failed to load CSV with alternative encoding: {path}")
            return None
    except MemoryError:
        loge(f"Not enough memory to load CSV: {path}")
        return None
    except Exception as e:
        loge(f"Unexpected error loading CSV {path}: {e}")
        return None


def store_csv(directory: Union[str, Path], 
              file_name: str, 
              df: pd.DataFrame,
              encoding: str = 'utf-8',
              backup: bool = True,
              **kwargs) -> bool:
    """
    Store DataFrame as CSV with improved error handling and backup option.
    
    Args:
        directory: Directory to store the file
        file_name: Name of the CSV file
        df: DataFrame to store
        encoding: File encoding
        backup: Whether to create backup of existing file
        **kwargs: Additional pandas to_csv arguments
        
    Returns:
        True if successful, False otherwise
    """
    if df is None or df.empty:
        logw("Cannot store empty or None DataFrame")
        return False
    
    if not file_name:
        loge("File name cannot be empty")
        return False
    
    try:
        directory_path = ensure_directory_exists(directory)
        file_path = directory_path / file_name
        
        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
            try:
                shutil.copy2(file_path, backup_path)
                logd(f"Created backup: {backup_path}")
            except Exception as e:
                logw(f"Failed to create backup: {e}")
        
        # Store DataFrame
        df.to_csv(file_path, encoding=encoding, index=False, **kwargs)
        
        # Verify the file was created
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logd(f"Stored CSV with {len(df)} rows ({file_size_mb:.1f}MB): {file_path}")
            return True
        else:
            loge(f"CSV file was not created: {file_path}")
            return False
            
    except PermissionError as e:
        loge(f"Permission denied storing CSV {file_name}: {e}")
        return False
    except OSError as e:
        loge(f"OS error storing CSV {file_name}: {e}")
        return False
    except Exception as e:
        loge(f"Unexpected error storing CSV {file_name}: {e}")
        return False


def delete_file(directory: Union[str, Path], file_name: str) -> bool:
    """
    Delete file with improved error handling.
    
    Args:
        directory: Directory containing the file
        file_name: Name of the file to delete
        
    Returns:
        True if successful or file doesn't exist, False on error
    """
    if not file_name:
        loge("File name cannot be empty")
        return False
    
    path = Path(directory) / file_name
    
    if not path.exists():
        logd(f"File doesn't exist (already deleted?): {path}")
        return True
    
    try:
        path.unlink()
        logd(f"Deleted file: {path}")
        return True
    except PermissionError as e:
        loge(f"Permission denied deleting {path}: {e}")
        return False
    except OSError as e:
        loge(f"OS error deleting {path}: {e}")
        return False
    except Exception as e:
        loge(f"Unexpected error deleting {path}: {e}")
        return False


def delete_files_in_directory(dir_path: Union[str, Path], 
                             pattern: str = "*",
                             recursive: bool = False) -> int:
    """
    Delete files in directory with pattern matching and safety checks.
    
    Args:
        dir_path: Directory path
        pattern: File pattern to match (default: all files)
        recursive: Whether to delete recursively
        
    Returns:
        Number of files deleted
    """
    directory = Path(dir_path)
    
    if not directory.exists():
        logw(f"Directory doesn't exist: {directory}")
        return 0
    
    if not directory.is_dir():
        loge(f"Path is not a directory: {directory}")
        return 0
    
    # Safety check: prevent deletion of system directories
    dangerous_paths = [Path.home(), Path("/"), Path("C:\\"), Path("C:\\Windows")]
    if any(directory.samefile(dangerous) for dangerous in dangerous_paths if dangerous.exists()):
        loge(f"Refusing to delete files in dangerous directory: {directory}")
        return 0
    
    deleted_count = 0
    failed_count = 0
    
    try:
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    failed_count += 1
                    logw(f"Error deleting file {file_path}: {e}")
        
        if deleted_count > 0:
            logd(f"Deleted {deleted_count} files from {directory}")
        if failed_count > 0:
            logw(f"Failed to delete {failed_count} files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        loge(f"Error scanning directory {directory}: {e}")
        return 0


def get_os_variable(key: str, default: Any = None, required: bool = True) -> Any:
    """
    Get OS environment variable with improved error handling.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required variable is not found and no default provided
    """
    if not key:
        raise ValueError("Environment variable key cannot be empty")
    
    value = os.environ.get(key)
    
    if value is not None:
        # Strip whitespace from string values
        if isinstance(value, str):
            value = value.strip()
            if not value:  # Empty string after stripping
                value = None
    
    if value is None:
        if default is not None:
            logd(f"Using default value for {key}")
            return default
        elif required:
            loge(f"Required environment variable {key} not set")
            raise ValueError(f"Required environment variable {key} not set")
        else:
            logd(f"Optional environment variable {key} not set")
            return None
    
    return value


def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if file doesn't exist
    """
    path = Path(file_path)
    
    try:
        if path.exists() and path.is_file():
            return path.stat().st_size
        else:
            return None
    except OSError as e:
        loge(f"Error getting file size for {path}: {e}")
        return None


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Move file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        loge(f"Source file doesn't exist: {src_path}")
        return False
    
    try:
        # Ensure destination directory exists
        ensure_directory_exists(dst_path.parent)
        
        # Move the file
        shutil.move(str(src_path), str(dst_path))
        logd(f"Moved file: {src_path} -> {dst_path}")
        return True
        
    except Exception as e:
        loge(f"Error moving file {src_path} to {dst_path}: {e}")
        return False


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Copy file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        loge(f"Source file doesn't exist: {src_path}")
        return False
    
    try:
        # Ensure destination directory exists
        ensure_directory_exists(dst_path.parent)
        
        # Copy the file
        shutil.copy2(str(src_path), str(dst_path))
        logd(f"Copied file: {src_path} -> {dst_path}")
        return True
        
    except Exception as e:
        loge(f"Error copying file {src_path} to {dst_path}: {e}")
        return False