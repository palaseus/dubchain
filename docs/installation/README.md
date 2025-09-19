# DubChain Installation Guide

This guide provides detailed installation instructions for DubChain on various platforms and environments.

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB free space minimum
- **Network**: Internet connection for dependencies

### Required Software

- **Python 3.10+**: [Download from python.org](https://www.python.org/downloads/)
- **pip**: Python package manager (included with Python 3.4+)
- **Git**: Version control system
- **Virtual Environment**: venv (included with Python 3.3+)

## Installation Methods

### Method 1: Development Installation (Recommended)

This method installs DubChain in development mode with all dependencies.

```bash
# Clone the repository
git clone https://github.com/dubchain/dubchain.git
cd dubchain

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import dubchain; print(dubchain.__version__)"
```

### Method 2: Production Installation

For production use or when you don't need development tools:

```bash
# Clone the repository
git clone https://github.com/dubchain/dubchain.git
cd dubchain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install production dependencies
pip install -e .

# Verify installation
python -c "import dubchain; print(dubchain.__version__)"
```

### Method 3: Docker Installation

For containerized deployment:

```bash
# Clone the repository
git clone https://github.com/dubchain/dubchain.git
cd dubchain

# Build Docker image
docker build -t dubchain .

# Run DubChain node
docker run -p 8080:8080 dubchain
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev git

# Install additional dependencies
sudo apt install build-essential libssl-dev libffi-dev

# Follow installation steps above
```

### Linux (CentOS/RHEL/Fedora)

```bash
# Install Python and development tools
sudo dnf install python3.10 python3.10-venv python3.10-devel git

# Install additional dependencies
sudo dnf install gcc openssl-devel libffi-devel

# Follow installation steps above
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.10 git

# Follow installation steps above
```

### Windows

1. **Install Python 3.10+**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Install Git**:
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Open Command Prompt or PowerShell**:
   ```cmd
   # Clone repository
   git clone https://github.com/dubchain/dubchain.git
   cd dubchain

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   venv\Scripts\activate

   # Install dependencies
   pip install -e ".[dev]"
   ```

## Dependency Management

### Core Dependencies

DubChain requires the following core dependencies:

- **cryptography**: Cryptographic primitives
- **pydantic**: Data validation and settings
- **fastapi**: Web API framework
- **uvicorn**: ASGI server
- **websockets**: WebSocket support
- **aiofiles**: Async file operations

### Development Dependencies

For development work, additional dependencies are included:

- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking support
- **hypothesis**: Property-based testing
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Optional Dependencies

Some features require optional dependencies:

- **jupyter**: Jupyter notebook support
- **ipython**: Enhanced Python shell
- **matplotlib**: Plotting and visualization
- **pandas**: Data analysis

## Configuration

### Environment Variables

Set the following environment variables for configuration:

```bash
# Database configuration
export DUBCHAIN_DB_URL="sqlite:///dubchain.db"
export DUBCHAIN_DB_POOL_SIZE=10

# Network configuration
export DUBCHAIN_NETWORK_PORT=8080
export DUBCHAIN_NETWORK_HOST="0.0.0.0"

# Consensus configuration
export DUBCHAIN_CONSENSUS_TYPE="proof_of_stake"
export DUBCHAIN_BLOCK_TIME=10

# Logging configuration
export DUBCHAIN_LOG_LEVEL="INFO"
export DUBCHAIN_LOG_FILE="dubchain.log"
```

### Configuration File

Create a `config.yaml` file in the project root:

```yaml
# Database configuration
database:
  url: "sqlite:///dubchain.db"
  pool_size: 10
  echo: false

# Network configuration
network:
  port: 8080
  host: "0.0.0.0"
  max_connections: 1000

# Consensus configuration
consensus:
  type: "proof_of_stake"
  block_time: 10
  difficulty_adjustment: true

# Logging configuration
logging:
  level: "INFO"
  file: "dubchain.log"
  max_size: "10MB"
  backup_count: 5
```

## Verification

### Basic Verification

```bash
# Check Python version
python --version

# Check DubChain installation
python -c "import dubchain; print(f'DubChain version: {dubchain.__version__}')"

# Check dependencies
python -c "import cryptography, pydantic, fastapi; print('Core dependencies OK')"
```

### Run Tests

```bash
# Run basic tests
pytest tests/unit/test_core_block.py -v

# Run all tests
pytest

# Run with coverage
pytest --cov=dubchain --cov-report=term-missing
```

### Run Examples

```bash
# Run basic blockchain demo
python examples/basic_blockchain_demo.py

# Run consensus demo
python examples/advanced_consensus_demo.py

# Run bridge demo
python examples/cross_chain_bridge_demo.py
```

## Troubleshooting

### Common Issues

#### Python Version Issues

**Problem**: Python version too old
**Solution**: Install Python 3.10 or higher

```bash
# Check Python version
python --version

# Install Python 3.10 (Ubuntu/Debian)
sudo apt install python3.10 python3.10-venv
```

#### Dependency Installation Issues

**Problem**: Failed to install cryptography
**Solution**: Install development headers

```bash
# Ubuntu/Debian
sudo apt install python3.10-dev libssl-dev libffi-dev

# CentOS/RHEL/Fedora
sudo dnf install python3.10-devel openssl-devel libffi-devel

# macOS
brew install openssl libffi
```

#### Virtual Environment Issues

**Problem**: Virtual environment not activating
**Solution**: Check activation script

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Verify activation
which python  # Should point to venv/bin/python
```

#### Permission Issues

**Problem**: Permission denied errors
**Solution**: Check file permissions

```bash
# Fix permissions
chmod +x venv/bin/activate
chmod -R 755 venv/
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the output
2. **Verify prerequisites**: Ensure all required software is installed
3. **Check documentation**: Review relevant documentation sections
4. **Search issues**: Check existing GitHub issues
5. **Create issue**: Create a new issue with detailed information

### Support Channels

- **GitHub Issues**: [Report bugs and request features](https://github.com/dubchain/dubchain/issues)
- **GitHub Discussions**: [Ask questions and discuss](https://github.com/dubchain/dubchain/discussions)
- **Email**: dev@dubchain.io
- **Documentation**: [Full documentation](docs/README.md)

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: [docs/quickstart/README.md](quickstart/README.md)
2. **Explore Examples**: Run the example scripts in `examples/`
3. **Read Documentation**: Browse the comprehensive documentation
4. **Join the Community**: Participate in discussions and contribute

## Uninstallation

To remove DubChain:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove repository
rm -rf dubchain
```

## Updating

To update DubChain:

```bash
# Navigate to project directory
cd dubchain

# Activate virtual environment
source venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install -e ".[dev]" --upgrade

# Run tests to verify
pytest
```
