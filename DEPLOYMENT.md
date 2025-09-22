# Deployment Guide

This guide covers deploying JudgeSync to PyPI.

## Prerequisites

1. **PyPI Account**: Register at [pypi.org](https://pypi.org/account/register/)
2. **Test PyPI Account**: Register at [test.pypi.org](https://test.pypi.org/account/register/)
3. **API Tokens**: Create API tokens for both accounts (recommended over passwords)

## Setup API Tokens

Create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## Release Workflow

### 1. Prepare Release

```bash
# Update version in pyproject.toml
# version = "0.1.1"  # Increment appropriately

# Run quality checks
make test
make lint
make format
```

### 2. Deploy to Test PyPI

```bash
# Build and upload to test environment
make upload-test

# Test installation from test PyPI
pip install -i https://test.pypi.org/simple/ judgesync==0.1.1

# Verify it works
python -c "from judgesync import AlignmentTracker; print('Success!')"
```

### 3. Deploy to Production PyPI

```bash
# Upload to production PyPI (will prompt for confirmation)
make upload
```

### 4. Verify Production Release

```bash
# Install from production PyPI
pip install judgesync==0.1.1

# Test basic functionality
python -c "from judgesync import AlignmentTracker; print('Production release works!')"
```

## Version Management

Follow [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.0 → 0.1.1): Bug fixes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

## Manual Commands

If you prefer manual control:

```bash
# Clean and build
make clean
make build

# Check package integrity
make check-dist

# Manual upload
python3 -m twine upload --repository testpypi dist/*  # Test PyPI
python3 -m twine upload dist/*                        # Production PyPI
```

## Troubleshooting

### Build Errors
- Run `make clean` to remove stale build artifacts
- Ensure all dependencies are installed: `make install`

### Upload Errors
- Check API tokens in `~/.pypirc`
- Verify version number is incremented
- Ensure package name isn't already taken

### Version Conflicts
- PyPI doesn't allow re-uploading the same version
- Increment version number and rebuild

## Security Notes

- Never commit API tokens to git
- Use API tokens instead of passwords
- Test on Test PyPI before production deployment
- Always increment version numbers for new releases
