import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional

class FAISSVectorIndex:
    """
    FAISS Fast Local Vector Search & Matrix Multiplication Search Engine.
    Stores L2-normalized 512-d employee embeddings and maps vector positions to employee IDs.
    """

    def __init__(self, dimension: int = 512, index_dir: str = "data/embeddings"):
        self.dimension = dimension
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.index_file = os.path.join(index_dir, "faiss_index.bin")
        self.meta_file = os.path.join(index_dir, "metadata.json")

        # Inner Product Index for Cosine Similarity on L2-normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        
        # Mapping: vector_id -> {"employee_id": str, "name": str, "image_path": str}
        self.id_map: List[Dict[str, str]] = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> None:
        """
        Adds normalized embeddings and metadata to the FAISS index.
        """
        if len(embeddings) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self.index.add(embeddings)
        self.id_map.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, str]]]:
        """
        Searches top_k matching candidates for a 512-d query embedding.
        Returns list of (cosine_similarity_score, employee_metadata_dict).
        """
        if self.index.ntotal == 0:
            return []

        query = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for sim, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((float(sim), self.id_map[idx]))

        return results

    def direct_matrix_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, str]]]:
        """
        Benchmark alternative: Direct Normalized Matrix Multiplication (X @ q).
        Extremely fast for small/medium galleries (<10,000 employees).
        """
        if self.index.ntotal == 0:
            return []

        # Reconstruct matrix
        matrix = self.index.reconstruct_n(0, self.index.ntotal).reshape(-1, self.dimension)
        query = query_embedding.flatten()

        sims = matrix @ query
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(float(sims[idx]), self.id_map[idx]) for idx in top_indices]

    def save(self) -> None:
        """Saves FAISS index binary and metadata JSON to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.id_map, f, indent=2)

    def load(self) -> bool:
        """Loads FAISS index and metadata if exists."""
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
            return True
        return False

    def remove_employee(self, employee_id: str) -> bool:
        """
        Removes an employee by employee_id from the FAISS vector index and metadata map.
        Rebuilds IndexFlatIP with the remaining vectors and saves to disk.
        """
        if self.index.ntotal == 0 or len(self.id_map) == 0:
            return False

        clean_id = employee_id.strip()
        keep_indices = []
        new_id_map = []

        for idx, item in enumerate(self.id_map):
            if item.get("employee_id") != clean_id:
                keep_indices.append(idx)
                new_id_map.append(item)

        if len(keep_indices) == len(self.id_map):
            return False

        if len(keep_indices) > 0:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal).reshape(-1, self.dimension)
            remaining_vectors = all_vectors[keep_indices]
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(np.ascontiguousarray(remaining_vectors.astype(np.float32)))
            self.id_map = new_id_map
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_map = []

        self.save()
        return True

    def get_all_employees(self) -> List[Dict[str, str]]:
        """Returns the list of all registered employee metadata dictionaries."""
        return list(self.id_map)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

