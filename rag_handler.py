# rag_handler.py
from llm_orchestrator import LLMOrchestrator
from typing import List, Dict
import logging

class RAGHandler:
    def __init__(self):
        self.llm_orchestrator = LLMOrchestrator()
        logging.info("RAGHandler initialized with enhanced LLM Orchestrator")
    
    def generate_response(self, query: str, context: List[Dict[str, any]]) -> str:
        """Generate response with metadata handling"""
        enhanced_context = []
        for doc in context:
            enhanced_doc = {
                **doc,
                'metadata': {
                    'authors': doc.get('authors', []),
                    'year': doc.get('year'),
                    'keywords': doc.get('keywords', [])
                }
            }
            enhanced_context.append(enhanced_doc)
        
        return self.llm_orchestrator.generate_response(query, enhanced_context)