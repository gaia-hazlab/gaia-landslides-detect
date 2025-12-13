# Contributing to Gaia Landslides Detection

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/gaia-landslides-detect.git
   cd gaia-landslides-detect
   ```
3. **Create a new branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit them
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
5. **Push to your fork** and submit a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

3. Run tests (when available):
   ```bash
   pytest
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

You can use `black` for automatic formatting:
```bash
black src/ train.py
```

## Pull Request Guidelines

- **Describe your changes** clearly in the PR description
- **Include tests** for new functionality when applicable
- **Update documentation** if you're changing functionality
- **Keep PRs focused** - one feature or fix per PR
- **Reference issues** if your PR addresses an existing issue

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (Python version, OS, etc.)
- Relevant code snippets or error messages

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
