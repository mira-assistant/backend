#!/bin/bash
# Cleanup script for coverage files and test artifacts

echo "Cleaning up coverage files and test artifacts..."

# Remove coverage files
rm -f .coverage .coverage.*
rm -f coverage.xml
rm -f test-results.xml

# Remove coverage HTML directory
rm -rf htmlcov/

echo "Cleanup completed."
