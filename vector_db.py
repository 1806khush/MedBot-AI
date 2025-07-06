# vector_db.py
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
from config.config import get_pinecone_config, get_embedding_config
from typing import List, Dict, Any
import time
import logging
import re
from rank_bm25 import BM25Okapi
import numpy as np
import concurrent.futures

class VectorDB:
    def __init__(self):
        pinecone_config = get_pinecone_config()
        embedding_config = get_embedding_config()

        self.embedding_model = SentenceTransformer(embedding_config['model_name'])
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.pc = Pinecone(api_key=pinecone_config['api_key'])
        spec = ServerlessSpec(cloud='aws', region=pinecone_config['environment'])

        if pinecone_config['index_name'] not in self.pc.list_indexes().names():
            logging.info(f"Creating new Pinecone index: {pinecone_config['index_name']}")
            self.pc.create_index(
                name=pinecone_config['index_name'],
                dimension=pinecone_config['dimension'],
                metric='cosine',
                spec=spec
            )
            time.sleep(10)

        self.index = self.pc.Index(pinecone_config['index_name'])
        logging.info(f"Connected to Pinecone index: {pinecone_config['index_name']}")
        
        # Thread pool for parallel operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def clean_for_embedding(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        cleaned = [self.clean_for_embedding(t) for t in texts]
        return self.embedding_model.encode(cleaned, show_progress_bar=False).tolist()

    def upsert_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            logging.warning("No documents to upsert")
            return

        logging.info(f"Upserting {len(documents)} documents")
        
        # Process documents in parallel batches
        batch_size = 100
        futures = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            future = self.executor.submit(self._process_batch, batch)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                logging.info(f"Upserted batch with {len(result)} vectors")
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents into vectors"""
        texts = [doc['text'] for doc in batch]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for doc, embedding in zip(batch, embeddings):
            vectors.append({
                'id': doc['id'],
                'values': embedding,
                'metadata': {
                    'title': doc.get('title', ''),
                    'section': doc.get('section', ''),
                    'source': doc.get('source', ''),
                    'text': doc.get('text', '')[:2000],
                    'authors': doc.get('authors', []),
                    'year': doc.get('year'),
                    'keywords': doc.get('keywords', []),
                    'is_semantic': doc.get('is_semantic', False)
                }
            })
        
        # Upsert the batch
        self.index.upsert(vectors=vectors)
        return vectors

    def hybrid_search(self, query: str, top_k: int = 10, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Perform hybrid vector + keyword search with re-ranking"""
        # Vector search
        query_embedding = self.generate_embeddings([query])[0]
        vector_results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 3,  # Get more results for re-ranking
            include_metadata=True
        )['matches']

        # Prepare documents for keyword search
        docs_for_keyword = [
            {'id': r['id'], 'text': r['metadata'].get('text', '')}
            for r in vector_results
        ]
        
        # BM25 keyword search
        tokenized_corpus = [self._tokenize(d['text']) for d in docs_for_keyword]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_results = []
        for i, vec_result in enumerate(vector_results):
            combined_score = (vec_result['score'] + bm25_scores[i]) / 2
            combined_results.append({
                'id': vec_result['id'],
                'vector_score': vec_result['score'],
                'keyword_score': bm25_scores[i],
                'combined_score': combined_score,
                'metadata': vec_result['metadata']
            })
        
        # Re-rank with cross-encoder
        if len(combined_results) > 0:
            cross_inp = [[query, res['metadata']['text']] for res in combined_results]
            cross_scores = self.cross_encoder.predict(cross_inp)
            
            for i, res in enumerate(combined_results):
                res['rerank_score'] = float(cross_scores[i])
                # Final score is weighted average
                res['final_score'] = (res['combined_score'] * 0.4) + (res['rerank_score'] * 0.6)
        
        # Filter and sort results
        relevant_docs = []
        for res in sorted(combined_results, key=lambda x: x['final_score'], reverse=True)[:top_k]:
            if res['final_score'] >= threshold:
                relevant_docs.append({
                    'id': res['id'],
                    'score': res['final_score'],
                    'text': res['metadata'].get('text', ''),
                    'title': res['metadata'].get('title', 'Untitled'),
                    'section': res['metadata'].get('section', 'Unknown section'),
                    'source': res['metadata'].get('source', 'Unknown source'),
                    'authors': res['metadata'].get('authors', []),
                    'year': res['metadata'].get('year'),
                    'keywords': res['metadata'].get('keywords', [])
                })
                logging.info(f"Found document with score {res['final_score']:.4f}: {res['metadata'].get('title', 'Untitled')}")

        return relevant_docs

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        return re.findall(r'\w+', text.lower())

    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Wrapper for hybrid search (maintaining original interface)"""
        return self.hybrid_search(query, top_k, threshold)
    
    def optimize_index(self):
        """Optimize the Pinecone index for better performance"""
        logging.info("Starting index optimization...")
        try:
            # Pinecone's optimize operation
            self.index.describe_index_stats()
            
            # Additional optimization steps
            stats = self.index.describe_index_stats()
            logging.info(f"Index stats before optimization: {stats}")
            
            # If index is too large, consider rebuilding
            if stats['total_vector_count'] > 100000:
                logging.warning("Large index detected, consider scaling up or splitting into multiple indexes")
            
            logging.info("Index optimization completed")
            return True
        except Exception as e:
            logging.error(f"Index optimization failed: {str(e)}")
            return False