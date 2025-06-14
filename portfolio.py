import pandas as pd
import faiss
import numpy as np
import pickle
import os
import uuid
from sentence_transformers import SentenceTransformer

class Portfolio:
    def __init__(self, file_path=r"C:\Users\DELL\OneDrive\Desktop\My Personal Projects\Gen AI project\project-genai-cold-email-generator\app\my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
        # Initialize FAISS components
        self.vector_dir = 'vectorstore'
        os.makedirs(self.vector_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.vector_dir, 'faiss_index.bin')
        self.metadata_path = os.path.join(self.vector_dir, 'metadata.pkl')
        
        # Load model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        
        # Initialize or load index and metadata
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_size)
            self.metadata = []

    def load_portfolio(self):
        if self.index.ntotal == 0:
            for _, row in self.data.iterrows():
                # Generate embedding
                text = row["Techstack"]
                embedding = self.model.encode([text])[0].astype(np.float32)
                
                # Add to index
                faiss.normalize_L2(np.array([embedding]))
                self.index.add(np.array([embedding]))
                
                # Store metadata
                self.metadata.append({"links": row["Links"]})
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def query_links(self, skills):
        if self.index.ntotal == 0:
            return []
            
        # Convert skills to embedding
        query_embedding = self.model.encode([skills])[0].astype(np.float32)
        faiss.normalize_L2(np.array([query_embedding]))
        
        # Query the index
        n_results = min(2, self.index.ntotal)
        D, I = self.index.search(np.array([query_embedding]), n_results)
        
        # Format results to match original return format
        results = [self.metadata[idx] for idx in I[0]]
        return results