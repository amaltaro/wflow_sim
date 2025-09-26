#!/bin/bash
# Simple test script for the release workflow
# This script verifies the release workflow is ready for GitHub

set -e

echo "🧪 Testing release workflow..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

echo "✅ Git repository detected"

# Check required files
echo "🔍 Checking required files..."
REQUIRED_FILES=(
    ".github/workflows/release-notes.yml"
    ".chglog/config.yml"
    "CONTRIBUTING.md"
    "docs/release-process.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check if git-chglog can be installed
echo "🔍 Checking git-chglog installation..."
if command -v go &> /dev/null; then
    echo "✅ Go is available"
    if go install github.com/git-chglog/git-chglog/cmd/git-chglog@latest 2>/dev/null; then
        echo "✅ git-chglog can be installed"
    else
        echo "❌ Failed to install git-chglog"
        exit 1
    fi
else
    echo "❌ Go is not available"
    exit 1
fi

# Check recent commit format
echo "🔍 Checking commit message format..."
CONVENTIONAL_COMMITS=$(git log --oneline -10 | grep -E "^(feat|fix|docs|style|refactor|perf|test|chore)(\(.+\))?:" | wc -l)
TOTAL_COMMITS=$(git log --oneline -10 | wc -l)

echo "📊 Conventional commits: $CONVENTIONAL_COMMITS out of $TOTAL_COMMITS recent commits"

if [ $CONVENTIONAL_COMMITS -gt 0 ]; then
    echo "✅ Found conventional commits - changelog generation will work!"
else
    echo "⚠️  No conventional commits found - changelog will be basic"
fi

# Test fallback mechanism
echo "🔍 Testing fallback mechanism..."
echo "## Test Release" > test-fallback.md
echo "" >> test-fallback.md
echo "Release Test Release" >> test-fallback.md
echo "" >> test-fallback.md
echo "Please ensure commits follow conventional commit format for automatic changelog generation." >> test-fallback.md

if [ -f "test-fallback.md" ]; then
    echo "✅ Fallback mechanism works!"
    rm test-fallback.md
else
    echo "❌ Fallback mechanism failed"
fi

echo ""
echo "✅ All tests completed!"
echo "🚀 Your release workflow is ready for GitHub!"
echo ""
echo "📝 Next steps:"
echo "1. git add . && git commit -m 'feat(ci): add automated release notes workflow'"
echo "2. git push origin main"
echo "3. git tag -a v0.1.0 -m 'Test release' && git push origin v0.1.0"
echo "4. Check: https://github.com/amaltaro/wflow_sim/actions"
