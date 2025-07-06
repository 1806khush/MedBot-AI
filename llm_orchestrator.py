# llm_orchestrator.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config.config import get_openrouter_config, get_biomistral_config
import requests
import logging
from typing import List, Dict, Optional
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self):
        self.openrouter_config = get_openrouter_config()
        self.biomistral_config = get_biomistral_config()
        self.biomistral_model = None
        self.biomistral_tokenizer = None
        self.query_classifier = None
        self.cache = {}
        
        # Initialize BioMistral if configured
        if self.biomistral_config['model_name']:
            self._init_biomistral()
        
        # Initialize query classifier
        self._init_query_classifier()
    
    def _init_biomistral(self):
        try:
            logger.info("Loading BioMistral model...")
            self.biomistral_tokenizer = AutoTokenizer.from_pretrained(
                self.biomistral_config['model_name']
            )
            self.biomistral_model = AutoModelForCausalLM.from_pretrained(
                self.biomistral_config['model_name'],
                device_map=self.biomistral_config['device'],
                torch_dtype=torch.float16 if 'cuda' in self.biomistral_config['device'] else torch.float32
            )
            logger.info("BioMistral model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BioMistral model: {str(e)}")
            self.biomistral_model = None

    def _init_query_classifier(self):
        """Initialize a simple zero-shot classifier for query routing"""
        try:
            self.query_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.biomistral_config['device'] if self.biomistral_model else -1
            )
        except Exception as e:
            logger.warning(f"Couldn't load query classifier: {str(e)}")
            self.query_classifier = None

    def classify_query(self, query: str, context: str = "") -> Dict:
        """Classify query with more sophisticated approach"""
        if self.query_classifier:
            candidate_labels = [
                "factual retrieval",
                "comparative analysis",
                "hypothesis generation",
                "summary request",
                "technical explanation"
            ]
            
            try:
                result = self.query_classifier(
                    query + " " + context[:500],  # Add some context
                    candidate_labels,
                    multi_label=True
                )
                return {
                    'primary_type': result['labels'][0],
                    'scores': dict(zip(result['labels'], result['scores']))}
            except Exception as e:
                logger.warning(f"Query classification failed: {str(e)}")
        
        # Fallback to rule-based approach
        query_lower = query.lower()
        if any(term in query_lower for term in ['hypothes', 'suggest', 'idea', 'novel', 'innovative']):
            return {'primary_type': 'hypothesis generation', 'scores': {}}
        elif any(term in query_lower for term in ['summar', 'overview', 'review', 'key points']):
            return {'primary_type': 'summary request', 'scores': {}}
        elif any(term in query_lower for term in ['compare', 'contrast', 'relationship', 'mechanism']):
            return {'primary_type': 'comparative analysis', 'scores': {}}
        else:
            return {'primary_type': 'factual retrieval', 'scores': {}}

    def select_model(self, query_type: str, context_length: int) -> str:
        """Dynamically select the best model based on query and context"""
        # If context is very large, prefer local model to avoid API timeouts
        if context_length > 4000 and self.biomistral_model:
            return 'biomistral'
        
        # For creative tasks, prefer BioMistral if available
        if query_type in ['hypothesis generation', 'comparative analysis'] and self.biomistral_model:
            return 'biomistral'
        
        # For summaries and factual retrieval, prefer OpenRouter
        return 'openrouter'

    def generate_with_biomistral(self, prompt: str, max_length: int = 1000) -> str:
        """Generate response using local BioMistral model with improved settings"""
        if not self.biomistral_model:
            raise Exception("BioMistral model not initialized")
            
        inputs = self.biomistral_tokenizer(prompt, return_tensors="pt").to(self.biomistral_model.device)
        
        with torch.no_grad():
            outputs = self.biomistral_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
        return self.biomistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_with_openrouter(self, prompt: str) -> str:
        """Generate response using OpenRouter API with improved settings"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.openrouter_config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1500,
            "presence_penalty": 0.2
        }
        
        try:
            response = requests.post(
                f"{self.openrouter_config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"Failed to generate with OpenRouter: {str(e)}")

    def generate_response(self, query: str, context: List[Dict[str, any]]) -> str:
        """Generate response using dynamic model selection"""
        cache_key = f"{query}_{hash(frozenset([c['id'] for c in context]))}"
        
        if cache_key in self.cache:
            logger.info("Returning cached response")
            return self.cache[cache_key]
        
        # Prepare context string with metadata
        context_str = "\n\n".join([
            f"Title: {doc.get('title', 'N/A')}\n"
            f"Authors: {', '.join(doc.get('authors', ['Unknown']))}\n"
            f"Year: {doc.get('year', 'N/A')}\n"
            f"Keywords: {', '.join(doc.get('keywords', []))}\n"
            f"Source: {doc.get('source', 'N/A')}\n"
            f"Section: {doc.get('section', 'Unknown section')}\n"
            f"Relevance Score: {doc.get('score', 0):.2f}\n"
            f"Content: {doc.get('text', '')[:2000]}..."
            for doc in context
        ])
        
        # Classify query with advanced method
        query_info = self.classify_query(query, context_str)
        query_type = query_info['primary_type']
        
        # Select appropriate model
        selected_model = self.select_model(query_type, len(context_str))
        
        # Prepare appropriate prompt based on query type
        prompt = self._create_dynamic_prompt(query, query_type, context_str)
        
        try:
            if selected_model == 'biomistral' and self.biomistral_model:
                result = self.generate_with_biomistral(prompt)
            else:
                result = self.generate_with_openrouter(prompt)
        except Exception as e:
            logger.error(f"Generation failed with {selected_model}: {str(e)}")
            # Fallback to other model
            try:
                if selected_model == 'biomistral':
                    result = self.generate_with_openrouter(prompt)
                elif self.biomistral_model:
                    result = self.generate_with_biomistral(prompt)
                else:
                    result = self._fallback_response(query, context_str)
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {str(fallback_error)}")
                result = self._fallback_response(query, context_str)
        
        self.cache[cache_key] = result
        return result

    def _create_dynamic_prompt(self, query: str, query_type: str, context: str) -> str:
        """Create prompt based on query type and context"""
        prompt_templates = {
            'hypothesis generation': """As a senior biomedical researcher, generate innovative hypotheses based on these papers:

Query: {query}

Context:
{context}

Generate 3-5 testable hypotheses that:
1. Combine findings from multiple studies
2. Address gaps in current research
3. Are biologically plausible
4. Include potential experimental approaches

Format each hypothesis with:
- Hypothesis statement
- Supporting evidence
- Novel contribution
- Validation approach""",

            'summary request': """Summarize key findings from these papers relevant to the query:

Query: {query}

Context:
{context}

Provide a concise summary (3-5 bullet points) covering:
- Key findings across studies
- Consensus areas
- Points of disagreement
- Most recent developments
- Clinical implications if relevant""",

            'comparative analysis': """Compare and analyze these research papers:

Query: {query}

Context:
{context}

Provide a detailed analysis that:
1. Answers the query directly
2. Compares methodologies and findings
3. Identifies mechanisms or relationships
4. Notes limitations and gaps
5. Suggests future research directions""",

            'factual retrieval': """Answer the question based on these research papers:

Query: {query}

Context:
{context}

Provide a precise answer that:
1. Directly addresses the question
2. Cites relevant papers
3. Notes confidence level
4. Highlights conflicting evidence if any"""
        }

        default_template = """Answer this biomedical research question:

Query: {query}

Context:
{context}

Provide a thorough response that synthesizes information from the provided research papers."""

        template = prompt_templates.get(query_type, default_template)
        return template.format(query=query, context=context)

    def _fallback_response(self, query: str, context: str) -> str:
        """Improved fallback response"""
        return f"""I couldn't generate a complete analysis, but here are the most relevant passages:

Question: {query}

Top Relevant Passages:
{context[:3000]}...

For better results, you might:
1. Rephrase your question to be more specific
2. Upload more related research papers
3. Ask about a different aspect of this research"""