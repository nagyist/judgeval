"""
Contains prompts and constants for the Elasticsearch query generation pipeline.
"""

# OpenAI model name
OPENAI_MODEL = "gpt-4-turbo-preview"

# System messages for different components
ENTITY_EXTRACTION_SYSTEM = "You are an entity extraction specialist for network management systems. Extract entities precisely in the requested format."
ENTITY_VALIDATION_SYSTEM = "You are an entity validation specialist for network management queries. Ensure entities are complete and accurate."
FIELD_EXTRACTION_SYSTEM = "You are a field extraction specialist for Elasticsearch queries. Your job is to identify which fields and values are relevant to a given query."
INDEX_SELECTION_SYSTEM = "You are an index selection specialist for Elasticsearch. Your job is to select the most appropriate index based on the query."
QUERY_GENERATOR_SYSTEM = "You are an Elasticsearch query generator. Your job is to convert natural language queries to Elasticsearch DSL using the exact field names provided in the schema."
RESPONSE_FORMATTER_SYSTEM = "You are a helpful assistant that translates database query results into natural language."

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """
Extract specific entities from this network management query. Entities can be:
- user: A specific user mentioned (e.g., "john.doe", "Jane Smith")
- device: A specific device mentioned (e.g., "AP-123", "Switch-45", "Router-7")
- device_type: Type of device mentioned (e.g., "access point", "switch", "router", "gateway")
- location: A specific location mentioned (e.g., "Building A", "3rd Floor", "New York Office")
- status: A status mentioned (e.g., "disconnected", "offline", "connected")
- timeframe: Any time reference (e.g., "last week", "yesterday", "since Monday")
- event_type: Type of event mentioned (e.g., "disconnection", "authentication failure")
- quantity: Any numerical quantity mentioned (e.g., "top 5", "more than 10")

Format your response as a JSON array with objects containing "type" (from the list above) and "name" (the actual entity text from the query) fields.
If no entities are found, return an empty array.

Query: {user_query}
"""

# Entity validation prompt
ENTITY_VALIDATION_PROMPT = """
Review and validate these entities extracted from the user query: "{user_query}"

Extracted entities:
{entities_json}

Are these entities valid and complete? Please correct or add any missing entities.
Valid entity types include: user, device, device_type, location, status, timeframe, event_type, quantity.

Return a JSON array with objects containing "type" and "name" fields.
"""

# Index selection prompt
INDEX_SELECTION_PROMPT = """
Select the most appropriate Elasticsearch index for this query: "{user_query}"

Available indices:
{index_descriptions}

Entities identified in the query:
{validated_entities_json}

Return ONLY the name of the index as a string, with no additional text or explanation.
Valid options are: {valid_indices}
"""

# Field values prompt
FIELD_VALUES_PROMPT = """
For this query: "{user_query}"

Determine which essential field values are explicitly or implicitly referenced.

Available fields and their possible values:
{field_values_json}

Return a JSON object with field names as keys and the appropriate values from the options provided.
If a field is not relevant to the query, do not include it in your response.
"""

# Query generation prompt
QUERY_GENERATION_PROMPT = """
Generate an Elasticsearch query based on the following user query and entities.

User query: "{user_query}"

Selected index: {selected_index}

Index schema fields: {index_fields_json}

Entity field mappings for this index:
{entity_mappings_json}

Validated entities with field mappings:
{formatted_entities_json}

Selected field values: {field_values_json}

IMPORTANT: Use EXACT field names from the index schema. For example:
- For users, use "{user_field}" NOT "user.name" 
- For organizations, use "{org_field}" NOT "organization.id"
- For devices, use "{device_field}" NOT "device.id"

Return ONLY the Elasticsearch query as a JSON object, with no additional text or explanation.

Use the following guidelines:
1. Use the "bool" query with "must", "should", or "filter" clauses as appropriate
2. For exact matches, use "term" queries with the EXACT field names from the schema
3. For text searches, use "match" queries
4. For filtering by date ranges, use "range" queries
5. For counting or aggregating, use "aggs"
6. For any org_id fields, use the placeholder "MASKED_ORG_ID"
7. For any user_id fields, use the placeholder "MASKED_USER_ID"
8. For any device_id fields, use the placeholder "MASKED_DEVICE_ID"
9. For any site_id fields, use the placeholder "MASKED_SITE_ID"
10. Use the provided field values where applicable
11. Limit results to 10 unless the query specifies a different limit
"""

# Response formatting prompt
RESPONSE_FORMATTING_PROMPT = """
Generate a natural language response for the following user query: "{user_query}"

Elasticsearch Index: {selected_index}

Search Results:
- Total hits: {total_hits}
- Results: {hits_json}
- Aggregations: {aggs_json}

Relevant Entities: {entities_text}

Instructions:
1. Respond in a natural language using complete sentences.
2. Be concise but provide all relevant information from the search results.
3. If the total hits is 0, explain that no results were found.
4. If there are aggregations, explain what they mean.
5. Make sure to refer to specific entities from the query in your response.
6. Ensure your response answers the original query.
"""

# Entity field mappings for each index
ENTITY_FIELD_MAPPINGS = {
    "devices": {
        "org": "org_id",
        "device": "id",
        "site": "site",
        "name": "name"
    },
    "clients": {
        "org": "org_id",
        "user": "username",
        "client": "id",
        "device": "device"
    },
    "locations": {
        "org": "org_id",
        "site": "id",
        "name": "name"
    },
    "events": {
        "org": "org_id",
        "device": "device_id",
        "user": "username",
        "client": "client_id",
        "event": "id"
    }
} 