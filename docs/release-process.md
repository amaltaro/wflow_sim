# Release Process

Automated release notes system for the Workflow Simulator project.

## Quick Start

1. **Make commits** following conventional format:
   ```bash
   git commit -m "feat(metrics): add resource efficiency calculation"
   git commit -m "fix(simulator): correct group execution timing"
   ```

2. **Create and push a tag**:
   ```bash
   git tag -a 1.0.0 -m "Release 1.0.0"
   git push origin 1.0.0
   ```

3. **Automatic release** - GitHub Actions will generate changelog and create release

## Commit Format

```
<type>(<scope>): <description>
```

### Types
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation
- `style` - Code style
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Test changes
- `chore` - Maintenance

### Scopes
- `metrics` - Workflow metrics calculation
- `simulator` - Core simulation engine
- `ci` - CI/CD pipeline
- `docs` - Documentation
- `examples` - Example scripts
- `tests` - Test files

## Testing

```bash
# Test locally
./scripts/test-changelog.sh

# Generate changelog manually
git-chglog --config .chglog/config.yml --output CHANGELOG.md
```

## Troubleshooting

- **Workflow doesn't trigger**: Ensure tag follows semantic versioning (`1.0.0` or `v1.0.0`)
- **Empty changelog**: Check commits follow conventional format
- **Permission errors**: Verify GitHub Actions permissions

## Configuration Files

- `.github/workflows/release-notes.yml` - GitHub Actions workflow
- `.chglog/config.yml` - git-chglog configuration
- `CONTRIBUTING.md` - Detailed commit guidelines
