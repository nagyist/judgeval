#!/usr/bin/env python3
"""
Demo script for setting and verifying customer ID and tags in Judgment traces.

This script demonstrates how to set metadata (customer ID and tags) on traces
and verifies that the metadata is correctly set and saved.

Usage:
    python customer_id_and_tags.py [--api-url API_URL] [--project PROJECT_NAME]
"""

import os
import sys
import json
import random
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("metadata-demo")

try:
    from judgeval.common.tracer import Tracer
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install required packages: pip install judgeval python-dotenv")
    sys.exit(1)

# Initialize a global variable for the Tracer instance
judgment = None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test customer ID and tags in Judgment traces")
    parser.add_argument(
        "--api-url", 
        default=None,
        help="Judgment API URL (default: uses .env or http://localhost:8000)"
    )
    parser.add_argument(
        "--project", 
        default="metadata-test",
        help="Project name for the trace (default: metadata-test)"
    )
    return parser.parse_args()


def validate_environment() -> Tuple[Optional[str], Optional[str]]:
    """Validate that required environment variables are set."""
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("JUDGMENT_API_KEY")
    org_id = os.getenv("JUDGMENT_ORG_ID")
    
    if not api_key:
        logger.error("JUDGMENT_API_KEY environment variable is not set")
        return None, org_id
        
    if not org_id:
        logger.error("JUDGMENT_ORG_ID environment variable is not set")
        return api_key, None
        
    return api_key, org_id


def get_customer_id() -> str:
    """Generate a random customer ID for testing."""
    return f"customer-{random.randint(1000, 9999)}"


def get_test_tags() -> List[str]:
    """Generate a list of tags for testing."""
    all_tags = ["test", "example", "metadata", "demo", "development"]
    # Select 2-4 random tags
    num_tags = random.randint(2, min(4, len(all_tags)))
    return random.sample(all_tags, num_tags)


def save_trace_data(trace_data: Dict[str, Any], filename: str = "trace_metadata.json") -> None:
    """Save trace data to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)
        logger.info(f"Trace metadata saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save trace data to {filename}: {e}")


def test_metadata(tracer: Tracer, customer_id: str, tags: List[str]) -> str:
    """Test setting and verifying customer ID and tags."""
    try:
        logger.info(f"Setting customer ID: {customer_id}")
        logger.info(f"Setting tags: {tags}")
        
        # Create a trace context
        with tracer.trace(name="test_metadata") as trace:
            # Set both customer ID and tags at once
            tracer.set_metadata(
                customer_id=customer_id,
                tags=tags
            )
            
            # Verify the metadata was set correctly
            current_trace = tracer.get_current_trace()
            if current_trace is None:
                raise ValueError("Failed to get current trace")
                
            if current_trace.customer_id != customer_id:
                logger.warning(f"Customer ID mismatch: expected '{customer_id}', got '{current_trace.customer_id}'")
            else:
                logger.info(f"Customer ID verified: {current_trace.customer_id}")
                
            if current_trace.tags != tags:
                logger.warning(f"Tags mismatch: expected {tags}, got {current_trace.tags}")
            else:
                logger.info(f"Tags verified: {current_trace.tags}")
            
            # Print all metadata
            logger.info("Printing trace metadata:")
            tracer.print_metadata()
            
            # Do some work
            logger.info("Performing test operations...")
            result = f"Test completed with customer ID: {customer_id} and tags: {tags}"
            
            # Save trace data to a file for inspection
            trace_data = {
                "trace_id": current_trace.trace_id,
                "name": current_trace.name,
                "project_name": current_trace.project_name,
                "created_at": current_trace.start_time,
                "customer_id": current_trace.customer_id,
                "tags": current_trace.tags
            }
            save_trace_data(trace_data)
            
            return result
    except Exception as e:
        logger.error(f"Error in test_metadata: {e}")
        raise


def main() -> int:
    """Main function."""
    args = parse_arguments()
    
    # Validate environment
    api_key, org_id = validate_environment()
    if api_key is None or org_id is None:
        return 1
    
    # Set API URL
    api_url = args.api_url or os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
    os.environ["JUDGMENT_API_URL"] = api_url
    
    # Initialize Tracer
    try:
        tracer = Tracer(
            api_key=api_key,
            organization_id=org_id,
            project_name=args.project
        )
    except Exception as e:
        logger.error(f"Failed to initialize Tracer: {e}")
        return 1
    
    # Generate test data
    customer_id = get_customer_id()
    tags = get_test_tags()
    
    print(f"\n=== Metadata Test ===\n")
    print(f"API URL: {api_url}")
    print(f"Project: {args.project}")
    print(f"Customer ID: {customer_id}")
    print(f"Tags: {tags}\n")
    
    try:
        # Run the test function
        result = test_metadata(tracer, customer_id, tags)
        print(f"\nResult: {result}")
        print("\nCheck the Judgment dashboard to verify the metadata was saved.")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())