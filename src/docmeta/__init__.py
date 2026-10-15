"""docmeta: LLM-based metadata extraction from agreements."""
from .config import Settings, build_extractor
from .pipeline import MetadataExtractor

__version__ = "0.1.0"
__all__ = ["MetadataExtractor", "Settings", "build_extractor", "__version__"]
