#!/bin/bash
# Cleanup script for coverage files and test artifacts

echo "Cleaning up coverage files and test artifacts..."

# Remove coverage files
rm -f .coverage .coverage.*
rm -f coverage.xml
rm -f test-results.xml

# Remove coverage HTML directory
rm -rf htmlcov/

# Ensure we can write to the current directory
touch test-results.xml 2>/dev/null && rm -f test-results.xml || echo "Warning: Cannot write to current directory"

echo "Cleanup completed."
