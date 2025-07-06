Biomedical Research Assistant - RAG System
Overview
The Biomedical Research Assistant is a sophisticated Retrieval-Augmented Generation (RAG) system designed to help researchers analyze and query scientific papers. It combines document processing, vector search, and large language models to provide intelligent answers to research questions with proper citations.

Key Features
- PDF Document Processing: Extracts text, metadata, and sections from research papers

- Vector Database: Stores document embeddings for semantic search using Pinecone

- Hybrid Search: Combines vector, keyword, and cross-encoder ranking

- LLM Orchestration: Dynamically selects between BioMistral (local) and OpenRouter models

- Query Classification: Routes queries to appropriate response strategies

- User Management: Secure login/signup with chat history tracking

System Architecture

The system consists of several key components:

- Frontend: HTML/CSS/JS interface with upload and chat functionality

- Backend: Flask server handling API requests

- Document Processing: Extracts and chunks PDF content

Vector Database: Pinecone for semantic search

- LLM Orchestrator: Manages local and API-based language models

- RAG Handler: Coordinates retrieval and generation

- Database: MySQL for user data and chat history

Workflow
1. Document Upload and Processing
- User uploads a PDF research paper

- System extracts text and metadata using PyPDF2

- Document is cleaned and split into semantic chunks

- Sections are identified (abstract, methods, results, etc.)

- Text chunks are embedded using Sentence Transformers

- Embeddings are stored in Pinecone with metadata

2. Query Processing
- User submits a research question

- System performs hybrid search (vector + keyword + reranking)

- Relevant document chunks are retrieved

- Query is classified by intent (factual, comparative, etc.)

- Appropriate model is selected (BioMistral or OpenRouter)

- Context-aware prompt is generated

- LLM generates response with citations

3. Response Generation
- Retrieved documents are formatted with metadata

- Query-specific prompt template is selected

- Response is generated with:

- Direct answer to question

- Supporting evidence from papers

- Source citations

- Suggested follow-up questions

Installation Steps
- Prerequisites
- Python 3.8+

- MySQL database

- Pinecone account (for vector DB)

- OpenRouter API key (or local LLM setup)

![image](https://github.com/user-attachments/assets/ab235ac1-d6dd-42f9-a240-84cba4fee1a9)
