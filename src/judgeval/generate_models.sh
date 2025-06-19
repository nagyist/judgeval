#!/bin/bash
set -e

# Config
OPENAPI_URL="http://localhost:8000/openapi.json"
TEMP_DIR=".gen_temp"
OUTPUT_DIR="generated_models"

# Step 1: Generate into a temp directory
openapi-python-client generate --url "$OPENAPI_URL" --output-path "$TEMP_DIR" --meta 'none'

# Step 2: Find the actual generated client folder
CLIENT_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d)

if [ -z "$CLIENT_DIR" ]; then
  echo "❌ Could not find generated client directory inside $TEMP_DIR"
  exit 1
fi

# Step 3: Copy only the models/ folder
cp -r "$TEMP_DIR/models" "$OUTPUT_DIR/"

# Step 4: (Optional) Clean up
rm -rf "$TEMP_DIR"

echo "✅ Models copied to $OUTPUT_DIR"