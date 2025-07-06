from configparser import ConfigParser
import os
from pathlib import Path

def load_config():
    config = ConfigParser()
    config_path = Path(__file__).parent / 'config.ini'
    config.read(config_path)
    return config

def get_pinecone_config():
    config = load_config()
    return {
        'api_key': config['pinecone']['api_key'],
        'environment': config['pinecone']['environment'],
        'index_name': config['pinecone']['index_name'],
        'dimension': int(config['pinecone']['dimension'])
    }

def get_embedding_config():
    config = load_config()
    return {
        'model_name': config['embedding']['model_name']
    }

def get_openrouter_config():
    config = load_config()
    return {
        'api_key': config['openrouter']['api_key'],
        'base_url': config['openrouter']['base_url'],
        'model': config['openrouter']['model']
    }

def get_biomistral_config():
    config = load_config()
    return {
        'model_name': config['biomistral']['model_name'],
        'device': config['biomistral']['device']
    }

def get_database_config():
    config = load_config()
    return {
        'host': config['database']['host'],
        'user': config['database']['user'],
        'password': config['database']['password'],
        'database': config['database']['database'],
        'port': int(config['database']['port'])
    }

def get_flask_config():
    config = load_config()
    return {
        'secret_key': config['flask']['secret_key'],
        'session_timeout_minutes': int(config['flask']['session_timeout_minutes'])
    }