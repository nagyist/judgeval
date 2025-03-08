import os
from typing import Dict, Any, List
from elasticsearch import Elasticsearch, ApiError, NotFoundError
import json
from uuid import uuid4

# ============================================================================
# ELASTICSEARCH SETUP INSTRUCTIONS
# ============================================================================
# 
# Prerequisites:
# 1. Install Elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
#    - Docker: docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.1
#    - Run: docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.11.1
#
# 2. Python requirements:
#    - pip install elasticsearch
#
# 3. Environment variables (optional):
#    - ELASTICSEARCH_HOST: Hostname (default: localhost)
#    - ELASTICSEARCH_PORT: Port (default: 9200)
#    - ELASTICSEARCH_USER: Username for authentication (if needed)
#    - ELASTICSEARCH_PASSWORD: Password for authentication (if needed)
#    - ELASTICSEARCH_USE_SSL: Set to "true" to use HTTPS (default: false)
# ============================================================================

def get_elasticsearch_client():
    """
    Returns an Elasticsearch client configured with environment variables or defaults.
    Default connection is to http://localhost:9200
    """
    # Initialize Elasticsearch client - uses environment variables or defaults
    es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
    es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
    es_user = os.getenv("ELASTICSEARCH_USER", "")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD", "")
    es_use_ssl = os.getenv("ELASTICSEARCH_USE_SSL", "false").lower() == "true"

    # Create Elasticsearch client with authentication if credentials provided
    if es_user and es_password:
        client = Elasticsearch(
            [f"{'https' if es_use_ssl else 'http'}://{es_host}:{es_port}"],
            basic_auth=(es_user, es_password),
            verify_certs=False if es_use_ssl else None
        )
    else:
        client = Elasticsearch([f"{'https' if es_use_ssl else 'http'}://{es_host}:{es_port}"])
    
    return client

# Create default client instance
es_client = get_elasticsearch_client()

# Define Elasticsearch index configurations
ES_INDEXES = {
    "devices": {
        "description": "Information about network devices like access points, switches, routers",
        "mappings": {
            "properties": {
                "name": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "status": {"type": "keyword"},
                "device_type": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "status": ["connected", "disconnected", "provisioning"],
            "device_type": ["access_point", "switch", "router"]
        }
    },
    "clients": {
        "description": "Information about client devices connected to the network",
        "mappings": {
            "properties": {
                "username": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "connection_status": {"type": "keyword"},
                "connection_type": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "connection_status": ["connected", "disconnected", "idle"],
            "connection_type": ["wifi", "wired", "vpn"]
        }
    },
    "locations": {
        "description": "Information about physical locations and sites",
        "mappings": {
            "properties": {
                "name": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "status": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "status": ["active", "maintenance", "inactive"],
            "type": ["campus", "building", "floor"]
        }
    },
    "events": {
        "description": "Network event logs including alerts and status changes",
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "event_type": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "event_type": ["device_disconnected", "client_disconnected", "auth_failure"],
            "severity": ["critical", "error", "warning", "info"]
        }
    }
}

# Sample data to load into Elasticsearch
ES_SAMPLE_DATA = {
    "devices": [
        {"id": "dev-101", "name": "AP-101", "org_id": "org_123456", "status": "connected", "device_type": "access_point", "model": "AP43", "ip_address": "10.0.0.101", "mac": "aa:bb:cc:11:22:33", "site": "site-001"},
        {"id": "dev-102", "name": "AP-102", "org_id": "org_123456", "status": "disconnected", "device_type": "access_point", "model": "AP43", "ip_address": "10.0.0.102", "mac": "aa:bb:cc:11:22:34", "site": "site-001"},
        {"id": "dev-103", "name": "AP-103", "org_id": "org_123456", "status": "connected", "device_type": "access_point", "model": "AP41", "ip_address": "10.0.0.103", "mac": "aa:bb:cc:11:22:35", "site": "site-002"},
        {"id": "dev-201", "name": "SW-201", "org_id": "org_123456", "status": "connected", "device_type": "switch", "model": "EX4300", "ip_address": "10.0.0.201", "mac": "aa:bb:cc:22:33:44", "site": "site-001"},
        {"id": "dev-202", "name": "SW-202", "org_id": "org_123456", "status": "disconnected", "device_type": "switch", "model": "EX4300", "ip_address": "10.0.0.202", "mac": "aa:bb:cc:22:33:45", "site": "site-002"},
        {"id": "dev-301", "name": "RT-301", "org_id": "org_123456", "status": "connected", "device_type": "router", "model": "SRX340", "ip_address": "10.0.0.1", "mac": "aa:bb:cc:33:44:55", "site": "site-001"}
    ],
    "clients": [
        {"id": "client-001", "username": "john.doe", "org_id": "org_123456", "connection_status": "connected", "connection_type": "wifi", "device": "iPhone", "ip_address": "10.0.1.101", "mac": "bb:cc:dd:11:22:33"},
        {"id": "client-002", "username": "jane.smith", "org_id": "org_123456", "connection_status": "disconnected", "connection_type": "wifi", "device": "Android", "ip_address": "10.0.1.102", "mac": "bb:cc:dd:11:22:34"},
        {"id": "client-003", "username": "admin", "org_id": "org_123456", "connection_status": "connected", "connection_type": "wired", "device": "Laptop", "ip_address": "10.0.1.103", "mac": "bb:cc:dd:11:22:35"}
    ],
    "locations": [
        {"id": "site-001", "name": "Building A", "org_id": "org_123456", "type": "building", "status": "active", "address": "123 Main St", "city": "San Francisco", "state": "CA"},
        {"id": "site-002", "name": "Building B", "org_id": "org_123456", "type": "building", "status": "active", "address": "456 Market St", "city": "San Francisco", "state": "CA"},
        {"id": "site-003", "name": "Data Center", "org_id": "org_123456", "type": "building", "status": "active", "address": "789 Server Ave", "city": "San Jose", "state": "CA"}
    ],
    "events": [
        {
            "id": "evt-001", 
            "org_id": "org_123456", 
            "timestamp": "2023-06-14T08:15:22Z", 
            "event_type": "device_disconnected",
            "device_name": "AP-102",
            "device_type": "access_point",
            "device_id": "dev-102",
            "status": "disconnected",
            "severity": "warning"
        },
        {
            "id": "evt-002", 
            "org_id": "org_123456", 
            "timestamp": "2023-06-13T14:22:36Z", 
            "event_type": "device_disconnected",
            "device_name": "SW-202",
            "device_type": "switch", 
            "device_id": "dev-202",
            "status": "disconnected",
            "severity": "warning"
        },
        {
            "id": "evt-003", 
            "org_id": "org_123456", 
            "timestamp": "2023-06-14T15:45:22Z", 
            "event_type": "client_disconnected",
            "username": "jane.smith",
            "client_id": "client-002",
            "connection_type": "wifi",
            "status": "disconnected",
            "severity": "info"
        },
        {
            "id": "evt-004", 
            "org_id": "org_123456", 
            "timestamp": "2023-06-15T09:30:15Z", 
            "event_type": "device_disconnected",
            "device_name": "AP-104",
            "device_type": "access_point",
            "device_id": "dev-104",
            "status": "disconnected",
            "severity": "warning"
        }
    ]
}

def init_elasticsearch():
    """Initialize Elasticsearch connection and load mock data"""
    try:
        # Use the get_elasticsearch_client function to create a client
        es = get_elasticsearch_client()
        
        # Test the connection
        if es.ping():
            print(f"Connected to Elasticsearch at http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}")
        else:
            print("Failed to connect to Elasticsearch")
            return False
        
        # Delete existing indices to ensure clean state
        for index_name in ES_INDEXES.keys():
            try:
                es.indices.delete(index=index_name)
                print(f"Deleted existing index: {index_name}")
            except NotFoundError:
                # Index doesn't exist, which is fine
                pass
            except Exception as e:
                print(f"Warning: Failed to delete index {index_name}: {e}")
        
        # Create indices
        create_indices(es)
        
        # Load mock data
        load_mock_data(es)
        
        # Verify indices were created
        for index_name in ES_INDEXES.keys():
            if not es.indices.exists(index=index_name):
                print(f"WARNING: Index {index_name} was not created properly")
            else:
                print(f"Verified index exists: {index_name}")
        
        return es
    except Exception as e:
        print(f"Elasticsearch initialization error: {e}")
        return False

def create_indices(es):
    """Create required indices if they don't exist"""
    # Create indices based on ES_INDEXES configuration
    for index_name, index_config in ES_INDEXES.items():
        if not es.indices.exists(index=index_name):
            es.indices.create(
                index=index_name,
                body={"mappings": index_config["mappings"]}
            )
            print(f"Created {index_name} index")

def load_mock_data(es):
    """Load sample data into the indices"""
    # Load sample data from ES_SAMPLE_DATA
    for index_name, documents in ES_SAMPLE_DATA.items():
        if not documents:
            continue
            
        # Bulk load documents
        bulk_data = []
        for doc in documents:
            doc_id = doc.get("id", str(uuid4()))
            bulk_data.append({"index": {"_index": index_name, "_id": doc_id}})
            bulk_data.append(doc)
        
        if bulk_data:
            es.bulk(body=bulk_data)
            print(f"Loaded {len(documents)} documents into {index_name}")

# Utility functions for querying Elasticsearch

def search_index(index_name: str, query: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
    """
    Search an Elasticsearch index with the provided query
    
    Args:
        index_name: Name of the index to search
        query: Elasticsearch query DSL dictionary
        size: Maximum number of results to return
        
    Returns:
        List of documents matching the query
    """
    try:
        result = es_client.search(index=index_name, body={"query": query, "size": size})
        return [hit["_source"] for hit in result["hits"]["hits"]]
    except Exception as e:
        print(f"Error searching index {index_name}: {e}")
        return []

def index_document(index_name: str, document: Dict[str, Any], doc_id: str = None) -> bool:
    """
    Index a document in Elasticsearch
    
    Args:
        index_name: Name of the index
        document: Document to index
        doc_id: Optional document ID (Elasticsearch will generate one if not provided)
        
    Returns:
        True if indexing was successful, False otherwise
    """
    try:
        if doc_id:
            es_client.index(index=index_name, id=doc_id, document=document)
        else:
            es_client.index(index=index_name, document=document)
        return True
    except Exception as e:
        print(f"Error indexing document to {index_name}: {e}")
        return False

def delete_document(index_name: str, doc_id: str) -> bool:
    """
    Delete a document from Elasticsearch
    
    Args:
        index_name: Name of the index
        doc_id: Document ID to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        es_client.delete(index=index_name, id=doc_id)
        return True
    except NotFoundError:
        print(f"Document {doc_id} not found in index {index_name}")
        return False
    except Exception as e:
        print(f"Error deleting document {doc_id} from {index_name}: {e}")
        return False

def check_elasticsearch_connection() -> bool:
    """
    Checks if the Elasticsearch connection is working
    
    Returns:
        True if connected, False otherwise
    """
    try:
        return es_client.ping()
    except Exception:
        return False 