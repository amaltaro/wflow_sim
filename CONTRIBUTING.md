# Contributing to Workflow Simulator

Thank you for your interest in contributing to the Workflow Simulator! This document provides guidelines and instructions for contributing to the project.

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for our commit messages. This helps us maintain a clear and consistent changelog and makes it easier to generate release notes automatically.

### Commit Message Format

Each commit message should follow this format:

```
<type>(<scope>): <description>
```

### Types

The following types are allowed:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries

### Scopes

The scope is optional and should be a noun describing the part of the codebase affected by the change. For example:

- `metrics`: Workflow metrics calculation
- `simulator`: Core simulation engine
- `ci`: CI/CD pipeline changes
- `docs`: Documentation changes
- `examples`: Example scripts and templates
- `tests`: Test files and test infrastructure

### Examples

```
feat(metrics): add resource efficiency calculation
fix(simulator): correct group execution timing
docs: update README with new features
style: format Python files according to PEP 8
test(metrics): add unit tests for workflow calculator
chore(ci): update GitHub Actions workflow
```

## Release Process

For detailed information about creating releases and automated release notes, see [Release Process Documentation](docs/release-process.md).

## Development Workflow

1. Create a new branch for your feature or fix
2. Make your changes following the commit message guidelines
3. Push your changes and create a pull request
4. Once approved, your changes will be merged into the main branch

## Code Style

### Python Files
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Include docstrings for functions and classes
- Write unit tests for new functionality
- Use type hints for function parameters and return values

### JSON Files
- Use consistent indentation (2 spaces)
- Follow the workflow template structure
- Validate JSON syntax

### Documentation
- Update relevant documentation when adding features
- Include usage examples in docstrings
- Keep README.md current and comprehensive

## Project-Specific Guidelines

### Workflow Simulator Specifics

When contributing to this project, consider:

- **Group-Based Execution**: Ensure changes align with the group execution model
- **Metrics Calculation**: Maintain consistency in metric calculation methods
- **Resource Constraints**: Consider HPC/HTC environment requirements
- **Performance**: Optimize for simulation speed while maintaining accuracy

### Testing Requirements

- Write unit tests for every new function
- Update tests when function logic changes
- Aim for high test coverage (>90%)
- Use descriptive test names that explain the scenario

### Documentation Standards

- Document every function with comprehensive docstrings
- Use type hints for all function parameters and return values
- Keep documentation current and concise
- Include usage examples in docstrings for complex functions

## Questions?

If you have any questions about contributing, please open an issue in the repository.
