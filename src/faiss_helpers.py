import faiss
import numpy as np

def create_ivf_index(embeddings, nlist=None):
    """Create and train an IVFFlat index. nlist defaults to sqrt(n)."""
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    nlist = nlist or int(np.sqrt(len(embeddings)))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)
    return index

def search_neighbors(index, query, k=5, nprobe=10):
    """Search for k nearest neighbors. Returns (distances, indices)."""
    query = np.array(query).astype('float32').reshape(1, -1) if query.ndim == 1 else query.astype('float32')
    index.nprobe = nprobe
    return index.search(query, k)

