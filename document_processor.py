# document_processor.py
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple
import re
from datetime import datetime
import spacy
from collections import defaultdict
import concurrent.futures
import logging

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise ImportError("Please install spaCy and the English model: python -m spacy download en_core_web_sm")

class DocumentProcessor:
    def __init__(self):
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF file"""
        reader = PdfReader(pdf_path)
        text = ""
        metadata = {
            'authors': [],
            'year': None,
            'keywords': [],
            'sections': defaultdict(str)
        }
        
        # Extract text from all pages in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            page_texts = list(executor.map(lambda page: page.extract_text() or "", reader.pages))
            text = " ".join(page_texts)
        
        # Extract metadata from document info
        doc_info = reader.metadata
        if doc_info:
            if doc_info.author:
                metadata['authors'] = [a.strip() for a in doc_info.author.split(';') if a.strip()]
            if doc_info.get('/CreationDate', ''):
                try:
                    date_str = doc_info['/CreationDate'][2:6]  # Extract year from PDF date format
                    metadata['year'] = int(date_str)
                except:
                    pass
        
        return text, metadata

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text with improved processing"""
        # Remove special characters and excessive whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Basic ASCII only
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def extract_keywords(text: str, n_keywords: int = 10) -> List[str]:
        """Extract keywords using NLP"""
        doc = nlp(text)
        keywords = []
        
        # Extract noun phrases and named entities
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC']]
        
        # Combine and get most frequent terms
        all_terms = noun_phrases + entities
        term_freq = defaultdict(int)
        for term in all_terms:
            if len(term.split()) <= 3:  # Limit to 3-word phrases
                term_freq[term] += 1
        
        # Get top terms
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, count in sorted_terms[:n_keywords]]

    @staticmethod
    def semantic_chunking(text: str, max_chunk_size: int = 1500) -> List[Dict[str, any]]:
        """Chunk text based on semantic boundaries using NLP"""
        doc = nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text
            sent_length = len(sent_text)
            
            if current_length + sent_length > max_chunk_size and current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'is_semantic': True
                })
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sent_text)
            current_length += sent_length
        
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'is_semantic': True
            })
        
        return chunks

    def chunk_document(self, text: str) -> Dict[str, any]:
        """Enhanced document chunking with both section-based and semantic approaches"""
        # First extract traditional sections
        sections = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'methods': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': '',
            'full_text': text
        }
        
        section_patterns = {
            'abstract': r'abstract|summary',
            'introduction': r'introduction|background',
            'methods': r'methods|methodology|materials and methods',
            'results': r'results|findings',
            'discussion': r'discussion|analysis',
            'conclusion': r'conclusion|concluding remarks',
            'references': r'references|bibliography'
        }
        
        matches = []
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matches.append((match.start(), section))
        
        matches.sort()
        
        if matches:
            sections['title'] = text[:matches[0][0]].strip()
        
        for i in range(len(matches)):
            start_pos, section_name = matches[i]
            end_pos = matches[i+1][0] if i+1 < len(matches) else len(text)
            sections[section_name] = text[start_pos:end_pos].strip()
        
        if not matches and text:
            first_newline = text.find('\n')
            if first_newline > 0:
                sections['title'] = text[:first_newline].strip()
        
        # Add semantic chunks for each section in parallel
        futures = {}
        for section in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']:
            if sections[section]:
                future = self.executor.submit(self.semantic_chunking, sections[section])
                futures[section] = future
        
        # Wait for all semantic chunking to complete
        for section, future in futures.items():
            try:
                semantic_chunks = future.result()
                sections[f'{section}_chunks'] = semantic_chunks
            except Exception as e:
                logging.error(f"Error in semantic chunking for {section}: {str(e)}")
                sections[f'{section}_chunks'] = []
        
        return sections

    @staticmethod
    def combine_sections(sections: Dict[str, str], section_list: List[str]) -> str:
        """Combine specified sections with improved handling"""
        combined = []
        for sec in section_list:
            if sec in sections and sections[sec]:
                combined.append(sections[sec])
        
        # Fallback to semantic chunks if main section is empty
        if not combined:
            for sec in section_list:
                if f'{sec}_chunks' in sections:
                    combined.extend([chunk['text'] for chunk in sections[f'{sec}_chunks']])
        
        return " ".join(combined) if combined else ""