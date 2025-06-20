import json
import os
import sys
from typing import Any, Dict, Generator, List, Union
import requests

spec_file = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000/openapi.json'

if spec_file.startswith('http'):
    r = requests.get(spec_file)
    r.raise_for_status()
    SPEC = r.json()
else:
    with open(spec_file, 'r') as f:
        SPEC = json.load(f)

JUDGEVAL_PATHS: List[str] = [
    # Traces
    '/traces/save/', 
    '/traces/fetch/', 
    '/traces/upsert/',
    '/traces/batch_fetch/',
    '/traces/delete/',
    '/traces/fetch_by_project/',
    '/traces/save_span/',
    # Evaluations
    '/evaluate_trace/', 
    '/evaluate/',
    '/log_eval_results/',
    '/fetch_eval_results_by_project_sorted_limit/',
    # Datasets
    '/datasets/pull_for_judgeval/',
    '/datasets/push/',
    '/datasets/insert_examples/',
    '/datasets/delete/',
    '/datasets/delete_examples/',
    '/datasets/fetch_by_project/',
    # Scorers
    '/save_scorer/',
    '/fetch_scorers/',
]



def resolve_ref(ref: str) -> str:
    assert ref.startswith('#/components/schemas/'), 'Reference must start with #/components/schemas/'
    return ref.replace('#/components/schemas/', '')


def walk(obj: Any) -> Generator[Any, None, None]:
    yield obj
    if isinstance(obj, list):
        for item in obj:
            yield from walk(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from walk(value)


def get_referenced_schemas(obj: Any) -> Generator[str, None, None]:
    for value in walk(obj):
        if isinstance(value, dict) and '$ref' in value:
            ref = value['$ref']
            resolved = resolve_ref(ref)
            assert isinstance(ref, str), 'Reference must be a string'
            yield resolved


filtered_paths = {
    path: spec_data 
    for path, spec_data in SPEC['paths'].items() 
    if path in JUDGEVAL_PATHS
}


def filter_schemas() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    schemas_to_scan: Any = filtered_paths
    
    while True:
        to_commit: Dict[str, Any] = {}
        for schema_name in get_referenced_schemas(schemas_to_scan):
            if schema_name in result:
                continue
            
            assert schema_name in SPEC['components']['schemas'], f'Schema {schema_name} not found in components.schemas'
            to_commit[schema_name] = SPEC['components']['schemas'][schema_name]
        
        if not to_commit:
            break
            
        result.update(to_commit)
        schemas_to_scan = to_commit

    return result


spec = {
    'openapi': SPEC['openapi'],
    'info': SPEC['info'],
    'paths': filtered_paths,
    'components': {
        **SPEC['components'],
        'schemas': filter_schemas(),
    },
}

print(json.dumps(spec, indent=4))