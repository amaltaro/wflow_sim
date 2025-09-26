#!/bin/bash
# Test script for git-chglog configuration
# This script tests the changelog generation locally

set -e

echo "Testing git-chglog configuration..."

# Check if git-chglog is installed
if ! command -v git-chglog &> /dev/null; then
    echo "Installing git-chglog..."
    go install github.com/git-chglog/git-chglog/cmd/git-chglog@latest
fi

# Test the configuration
echo "Testing changelog generation..."
git-chglog --config .chglog/config.yml --output test-changelog.md

if [ -f "test-changelog.md" ]; then
    echo "✅ Changelog generated successfully!"
    echo "Preview of generated changelog:"
    echo "----------------------------------------"
    head -20 test-changelog.md
    echo "----------------------------------------"
    rm test-changelog.md
else
    echo "❌ Failed to generate changelog"
    exit 1
fi

echo "✅ git-chglog configuration test passed!"
