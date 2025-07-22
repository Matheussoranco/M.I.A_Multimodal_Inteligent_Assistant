# Contributing to M.I.A

Thank you for your interest in contributing to M.I.A (Multimodal Intelligent Assistant)! 

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/M.I.A-The-successor-of-pseudoJarvis.git
   cd M.I.A-The-successor-of-pseudoJarvis
   ```

2. **Set up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

## 🛠️ Development Guidelines

### Code Style
- Use `black` for code formatting: `black src/ tests/`
- Use `isort` for import sorting: `isort src/ tests/`
- Follow PEP 8 guidelines
- Add type hints for all functions
- Write docstrings for all public functions and classes

### Testing
- Write tests for all new features
- Maintain test coverage above 80%
- Use meaningful test names that describe what is being tested
- Mock external dependencies appropriately

### Commit Messages
Follow the conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(llm): add support for Anthropic Claude models

- Add Claude provider to LLM manager
- Update configuration schema
- Add tests for Claude integration
```

## 🔧 Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python -m pytest tests/ -v
   black --check src/ tests/
   isort --check-only src/ tests/
   ```

4. **Submit Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Include screenshots for UI changes
   - Request review from maintainers

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and OS
- M.I.A version
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

Before submitting feature requests:
- Check existing issues for duplicates
- Provide clear use cases
- Consider implementation complexity
- Be open to alternative solutions

## 📝 Documentation

Help improve our documentation:
- Fix typos and grammar
- Add examples and tutorials
- Improve API documentation
- Translate to other languages

## 🏗️ Architecture Overview

M.I.A follows a modular architecture:

```
src/mia/
├── core/              # Core cognitive architecture
├── llm/               # LLM integration layer
├── audio/             # Audio processing
├── multimodal/        # Vision and multimodal processing
├── memory/            # Memory systems
├── tools/             # Action execution
├── security/          # Security management
└── utils/             # Utilities
```

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different perspectives and experiences

## 📞 Getting Help

- 💬 Join our [Discord](link-to-discord)
- 📧 Email: matheussoranco@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis/issues)
- 📖 Documentation: [Read the Docs](link-to-docs)

## 🙏 Recognition

Contributors will be recognized in:
- CHANGELOG.md
- README.md contributors section
- Release notes

Thank you for making M.I.A better! 🎉
