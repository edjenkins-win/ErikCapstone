# Standard library imports
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Third-party imports
import faiss
import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Configure logger
logger = logging.getLogger(__name__)

class VectorStore:
    """Efficient vector storage and retrieval system for image features."""
    
    def __init__(self, dimension: int = 512, index_path: str = "vector_store.index"):
        """Initialize the vector store.
        
        Args:
            dimension: Dimension of the feature vectors
            index_path: Path to save/load the FAISS index
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix('.json')
        self.backup_dir = self.index_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
        # Initialize clustering
        self.clusters = None
        self.cluster_centers = None
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the index and metadata from disk."""
        try:
            if self.index_path.exists():
                logger.info(f"Loading index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                self._load_metadata()
                logger.info(f"Successfully loaded index with {len(self.metadata)} vectors")
            else:
                logger.info("No existing index found, creating new one")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Create a new index if loading fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and their metadata to the store.
        
        Args:
            vectors: Array of feature vectors
            metadata: List of metadata dictionaries
        """
        try:
            # Convert to float32 if necessary
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Add metadata
            self.metadata.extend(metadata)
            
            # Save index and metadata
            self._save()
            logger.info(f"Added {len(vectors)} vectors to the index")
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of metadata dictionaries for the k nearest neighbors
        """
        try:
            # Convert to float32 if necessary
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)
            
            # Reshape if necessary
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_vector, k)
            
            # Return results with metadata
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    results.append({
                        'metadata': self.metadata[idx],
                        'distance': float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def _save(self) -> None:
        """Save the index and metadata."""
        try:
            # Create backup before saving
            self._create_backup()
            
            # Save index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            
            logger.info(f"Successfully saved index and metadata")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def _load_metadata(self) -> None:
        """Load metadata from file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = []
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = []
    
    def _create_backup(self) -> None:
        """Create a backup of the current index and metadata."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"index_{timestamp}.index"
            metadata_backup_path = self.backup_dir / f"metadata_{timestamp}.json"
            
            if self.index_path.exists():
                shutil.copy2(self.index_path, backup_path)
            if self.metadata_path.exists():
                shutil.copy2(self.metadata_path, metadata_backup_path)
            
            logger.info(f"Created backup at {timestamp}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def clear(self) -> None:
        """Clear the vector store."""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            self._save()
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def get_size(self) -> int:
        """Get the current number of vectors in the index."""
        return len(self.metadata)
    
    def is_loaded(self) -> bool:
        """Check if the index is loaded and contains vectors."""
        return self.index.ntotal > 0 and len(self.metadata) > 0
    
    def __del__(self):
        """Cleanup when the object is deleted."""
        try:
            self._save()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def cluster_styles(self, n_clusters: int = 10) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster styles based on their feature vectors.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of style metadata
        """
        try:
            if not self.is_loaded():
                logger.warning("No vectors available for clustering")
                return {}
            
            # Get all vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Normalize vectors
            vectors = normalize(vectors, axis=1)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = kmeans.fit_predict(vectors)
            self.cluster_centers = kmeans.cluster_centers_
            
            # Group metadata by cluster
            clustered_styles = {}
            for i, cluster_id in enumerate(self.clusters):
                if cluster_id not in clustered_styles:
                    clustered_styles[cluster_id] = []
                clustered_styles[cluster_id].append(self.metadata[i])
            
            logger.info(f"Created {n_clusters} style clusters")
            return clustered_styles
        except Exception as e:
            logger.error(f"Error clustering styles: {e}")
            return {}
    
    def find_similar_content(self, content_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Find styles that match the content of an image.
        
        Args:
            content_vector: Feature vector of the content image
            k: Number of results to return
            
        Returns:
            List of matching styles with metadata
        """
        try:
            if not self.is_loaded():
                return []
            
            # Normalize content vector
            content_vector = normalize(content_vector.reshape(1, -1))
            
            # Search for similar styles
            results = self.search(content_vector, k)
            
            # Sort by content similarity
            results.sort(key=lambda x: x['distance'])
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    def blend_styles(self, style_vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Blend multiple style vectors together.
        
        Args:
            style_vectors: List of style feature vectors
            weights: Optional weights for each style
            
        Returns:
            Blended style vector
        """
        try:
            if not style_vectors:
                raise ValueError("No style vectors provided")
            
            # Normalize vectors
            style_vectors = [normalize(v.reshape(1, -1)) for v in style_vectors]
            
            # Use equal weights if none provided
            if weights is None:
                weights = [1.0 / len(style_vectors)] * len(style_vectors)
            
            # Ensure weights sum to 1
            weights = np.array(weights) / sum(weights)
            
            # Blend vectors
            blended = np.zeros_like(style_vectors[0])
            for v, w in zip(style_vectors, weights):
                blended += v * w
            
            return blended.reshape(-1)
        except Exception as e:
            logger.error(f"Error blending styles: {e}")
            raise
    
    def get_style_progression(self, start_style: np.ndarray, end_style: np.ndarray, 
                            steps: int = 5) -> List[np.ndarray]:
        """Generate a progression of style vectors between two styles.
        
        Args:
            start_style: Starting style vector
            end_style: Ending style vector
            steps: Number of intermediate steps
            
        Returns:
            List of style vectors representing the progression
        """
        try:
            # Normalize input vectors
            start_style = normalize(start_style.reshape(1, -1))
            end_style = normalize(end_style.reshape(1, -1))
            
            # Generate progression
            progression = []
            for t in np.linspace(0, 1, steps):
                interpolated = (1 - t) * start_style + t * end_style
                progression.append(interpolated.reshape(-1))
            
            return progression
        except Exception as e:
            logger.error(f"Error generating style progression: {e}")
            raise
    
    def get_style_statistics(self) -> Dict[str, Any]:
        """Get statistics about the style vectors.
        
        Returns:
            Dictionary containing style statistics
        """
        try:
            if not self.is_loaded():
                return {}
            
            # Get all vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Calculate statistics
            mean_vector = np.mean(vectors, axis=0)
            std_vector = np.std(vectors, axis=0)
            
            return {
                'num_styles': len(vectors),
                'mean_vector': mean_vector.tolist(),
                'std_vector': std_vector.tolist(),
                'vector_dimension': self.dimension
            }
        except Exception as e:
            logger.error(f"Error calculating style statistics: {e}")
            return {}
    
    def interpolate_styles(self, style_vectors: List[np.ndarray], weights: List[float], 
                         num_steps: int = 10) -> List[np.ndarray]:
        """Interpolate between multiple styles with intermediate steps.
        
        Args:
            style_vectors: List of style feature vectors
            weights: Weights for each style (will be normalized)
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated style vectors
        """
        try:
            if not style_vectors:
                raise ValueError("No style vectors provided")
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Normalize vectors
            style_vectors = [normalize(v.reshape(1, -1)) for v in style_vectors]
            
            # Generate interpolation steps
            interpolated_styles = []
            for t in np.linspace(0, 1, num_steps):
                # Calculate weighted combination
                blended = np.zeros_like(style_vectors[0])
                for v, w in zip(style_vectors, weights):
                    # Use smoothstep function for smoother transitions
                    smooth_t = t * t * (3 - 2 * t)
                    blended += v * w * smooth_t
                
                interpolated_styles.append(blended.reshape(-1))
            
            return interpolated_styles
        except Exception as e:
            logger.error(f"Error interpolating styles: {e}")
            raise
    
    def calculate_style_quality(self, content_vector: np.ndarray, style_vector: np.ndarray,
                              target_vector: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for style transfer.
        
        Args:
            content_vector: Feature vector of content image
            style_vector: Feature vector of style image
            target_vector: Feature vector of generated image
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            # Normalize vectors
            content_vector = normalize(content_vector.reshape(1, -1))
            style_vector = normalize(style_vector.reshape(1, -1))
            target_vector = normalize(target_vector.reshape(1, -1))
            
            # Calculate content preservation
            content_preservation = 1 - cosine(content_vector.reshape(-1), target_vector.reshape(-1))
            
            # Calculate style transfer
            style_transfer = 1 - cosine(style_vector.reshape(-1), target_vector.reshape(-1))
            
            # Calculate style-content balance
            balance = 1 - abs(content_preservation - style_transfer)
            
            # Calculate overall quality score
            quality_score = (content_preservation + style_transfer + balance) / 3
            
            return {
                'content_preservation': float(content_preservation),
                'style_transfer': float(style_transfer),
                'balance': float(balance),
                'quality_score': float(quality_score)
            }
        except Exception as e:
            logger.error(f"Error calculating style quality: {e}")
            return {}
    
    def detect_style_outliers(self, threshold: float = 2.0) -> Tuple[List[int], List[float]]:
        """Detect outlier styles in the collection.
        
        Args:
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Tuple of (outlier indices, outlier scores)
        """
        try:
            if not self.is_loaded():
                return [], []
            
            # Get all vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Calculate pairwise distances
            distances = np.zeros(len(vectors))
            for i, v1 in enumerate(vectors):
                # Calculate average distance to other vectors
                dists = [cosine(v1, v2) for j, v2 in enumerate(vectors) if i != j]
                distances[i] = np.mean(dists)
            
            # Calculate z-scores
            z_scores = zscore(distances)
            
            # Find outliers
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
            outlier_scores = z_scores[outlier_indices]
            
            return list(outlier_indices), list(outlier_scores)
        except Exception as e:
            logger.error(f"Error detecting style outliers: {e}")
            return [], []
    
    def get_style_diversity(self) -> Dict[str, float]:
        """Calculate diversity metrics for the style collection.
        
        Returns:
            Dictionary of diversity metrics
        """
        try:
            if not self.is_loaded():
                return {}
            
            # Get all vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Calculate pairwise distances
            distances = []
            for i, v1 in enumerate(vectors):
                for j, v2 in enumerate(vectors[i+1:], i+1):
                    distances.append(cosine(v1, v2))
            
            distances = np.array(distances)
            
            return {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'diversity_score': float(1 - np.mean(distances))  # Higher score means more diverse
            }
        except Exception as e:
            logger.error(f"Error calculating style diversity: {e}")
            return {} 