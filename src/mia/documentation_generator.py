"""
Documentation Generator - Automatic documentation generation for M.I.A
"""
import ast
import os
import inspect
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class DocFunction:
    """Documentation for a function."""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, str]]
    returns: str
    raises: List[str]
    examples: List[str]
    
@dataclass
class DocClass:
    """Documentation for a class."""
    name: str
    docstring: str
    methods: List[DocFunction]
    attributes: List[Dict[str, str]]
    inheritance: List[str]
    
@dataclass
class DocModule:
    """Documentation for a module."""
    name: str
    path: str
    docstring: str
    classes: List[DocClass]
    functions: List[DocFunction]
    imports: List[str]
    constants: List[Dict[str, str]]

class DocumentationGenerator:
    """Generates comprehensive documentation for M.I.A codebase."""
    
    def __init__(self, source_dir: str = "src/mia"):
        self.source_dir = Path(source_dir)
        self.docs_dir = Path("docs/generated")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_docs(self) -> None:
        """Generate documentation for all modules."""
        logger.info("Starting documentation generation...")
        
        # Find all Python files
        python_files = list(self.source_dir.rglob("*.py"))
        modules = []
        
        for py_file in python_files:
            if py_file.name != "__init__.py":
                try:
                    module_doc = self._analyze_module(py_file)
                    modules.append(module_doc)
                except Exception as e:
                    logger.error(f"Error analyzing {py_file}: {e}")
                    
        # Generate documentation files
        self._generate_module_docs(modules)
        self._generate_api_reference(modules)
        self._generate_usage_examples(modules)
        self._generate_index(modules)
        
        logger.info(f"Documentation generated in {self.docs_dir}")
        
    def _analyze_module(self, file_path: Path) -> DocModule:
        """Analyze a Python module and extract documentation."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            raise
            
        module_name = str(file_path.relative_to(self.source_dir)).replace('/', '.').replace('.py', '')
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree) or ""
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
                    
        # Extract classes and functions
        classes = []
        functions = []
        constants = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(self._analyze_class(node))
            elif isinstance(node, ast.FunctionDef):
                functions.append(self._analyze_function(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value) if hasattr(ast, 'unparse') else 'N/A'
                        })
                        
        return DocModule(
            name=module_name,
            path=str(file_path),
            docstring=module_docstring,
            classes=classes,
            functions=functions,
            imports=imports,
            constants=constants
        )
        
    def _analyze_class(self, node: ast.ClassDef) -> DocClass:
        """Analyze a class and extract documentation."""
        class_docstring = ast.get_docstring(node) or ""
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._analyze_function(item))
                
        # Extract attributes (from __init__ if present)
        attributes = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                attributes.extend(self._extract_attributes(item))
                
        # Extract inheritance
        inheritance = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
            elif isinstance(base, ast.Attribute):
                inheritance.append(ast.unparse(base) if hasattr(ast, 'unparse') else 'N/A')
                
        return DocClass(
            name=node.name,
            docstring=class_docstring,
            methods=methods,
            attributes=attributes,
            inheritance=inheritance
        )
        
    def _analyze_function(self, node: ast.FunctionDef) -> DocFunction:
        """Analyze a function and extract documentation."""
        function_docstring = ast.get_docstring(node) or ""
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, 'unparse') else 'Any',
                'description': ''
            }
            parameters.append(param_info)
            
        # Extract default values
        defaults = node.args.defaults
        if defaults:
            default_offset = len(parameters) - len(defaults)
            for i, default in enumerate(defaults):
                param_index = default_offset + i
                if param_index < len(parameters):
                    parameters[param_index]['default'] = ast.unparse(default) if hasattr(ast, 'unparse') else 'N/A'
                    
        # Extract return type
        returns = ""
        if node.returns:
            returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'Any'
            
        # Parse docstring for additional info
        docstring_info = self._parse_docstring(function_docstring)
        
        # Update parameter descriptions
        for param in parameters:
            if param['name'] in docstring_info.get('parameters', {}):
                param['description'] = docstring_info['parameters'][param['name']]
                
        return DocFunction(
            name=node.name,
            signature=self._generate_signature(node),
            docstring=function_docstring,
            parameters=parameters,
            returns=docstring_info.get('returns', returns),
            raises=docstring_info.get('raises', []),
            examples=docstring_info.get('examples', [])
        )
        
    def _extract_attributes(self, init_node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Extract class attributes from __init__ method."""
        attributes = []
        
        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            attributes.append({
                                'name': target.attr,
                                'type': 'Any',
                                'description': ''
                            })
                            
        return attributes
        
    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring to extract structured information."""
        info = {
            'parameters': {},
            'returns': '',
            'raises': [],
            'examples': []
        }
        
        if not docstring:
            return info
            
        lines = docstring.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line.startswith(':param '):
                match = re.match(r':param\s+(\w+):\s*(.*)', line)
                if match:
                    param_name, description = match.groups()
                    info['parameters'][param_name] = description
                    
            elif line.startswith(':return:') or line.startswith(':returns:'):
                info['returns'] = line.split(':', 2)[-1].strip()
                
            elif line.startswith(':raises:') or line.startswith(':raise:'):
                info['raises'].append(line.split(':', 2)[-1].strip())
                
            elif line.startswith('Example:') or line.startswith('Examples:'):
                current_section = 'examples'
                
            elif current_section == 'examples' and line:
                info['examples'].append(line)
                
        return info
        
    def _generate_signature(self, node: ast.FunctionDef) -> str:
        """Generate function signature string."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'Any'}"
            args.append(arg_str)
            
        # Default values
        defaults = node.args.defaults
        if defaults:
            default_offset = len(args) - len(defaults)
            for i, default in enumerate(defaults):
                arg_index = default_offset + i
                if arg_index < len(args):
                    default_val = ast.unparse(default) if hasattr(ast, 'unparse') else 'N/A'
                    args[arg_index] += f" = {default_val}"
                    
        signature = f"{node.name}({', '.join(args)})"
        
        if node.returns:
            signature += f" -> {ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'Any'}"
            
        return signature
        
    def _generate_module_docs(self, modules: List[DocModule]) -> None:
        """Generate documentation files for each module."""
        for module in modules:
            doc_file = self.docs_dir / f"{module.name.replace('.', '_')}.md"
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(f"# {module.name}\n\n")
                
                if module.docstring:
                    f.write(f"{module.docstring}\n\n")
                    
                # Classes
                if module.classes:
                    f.write("## Classes\n\n")
                    for cls in module.classes:
                        f.write(f"### {cls.name}\n\n")
                        if cls.docstring:
                            f.write(f"{cls.docstring}\n\n")
                            
                        if cls.inheritance:
                            f.write(f"**Inherits from:** {', '.join(cls.inheritance)}\n\n")
                            
                        # Methods
                        if cls.methods:
                            f.write("#### Methods\n\n")
                            for method in cls.methods:
                                f.write(f"##### {method.signature}\n\n")
                                if method.docstring:
                                    f.write(f"{method.docstring}\n\n")
                                    
                                if method.parameters:
                                    f.write("**Parameters:**\n\n")
                                    for param in method.parameters:
                                        f.write(f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n")
                                    f.write("\n")
                                    
                                if method.returns:
                                    f.write(f"**Returns:** {method.returns}\n\n")
                                    
                # Functions
                if module.functions:
                    f.write("## Functions\n\n")
                    for func in module.functions:
                        f.write(f"### {func.signature}\n\n")
                        if func.docstring:
                            f.write(f"{func.docstring}\n\n")
                            
                        if func.parameters:
                            f.write("**Parameters:**\n\n")
                            for param in func.parameters:
                                f.write(f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n")
                            f.write("\n")
                            
                        if func.returns:
                            f.write(f"**Returns:** {func.returns}\n\n")
                            
                # Constants
                if module.constants:
                    f.write("## Constants\n\n")
                    for const in module.constants:
                        f.write(f"- `{const['name']}`: {const['value']}\n")
                    f.write("\n")
                    
    def _generate_api_reference(self, modules: List[DocModule]) -> None:
        """Generate API reference documentation."""
        api_file = self.docs_dir / "api_reference.md"
        
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write("# M.I.A API Reference\n\n")
            f.write("This document provides a comprehensive API reference for M.I.A components.\n\n")
            
            # Group modules by package
            packages = {}
            for module in modules:
                package = module.name.split('.')[0] if '.' in module.name else 'core'
                if package not in packages:
                    packages[package] = []
                packages[package].append(module)
                
            for package, package_modules in packages.items():
                f.write(f"## {package.title()} Package\n\n")
                
                for module in package_modules:
                    f.write(f"### {module.name}\n\n")
                    
                    # Quick overview
                    class_count = len(module.classes)
                    function_count = len(module.functions)
                    f.write(f"**Classes:** {class_count} | **Functions:** {function_count}\n\n")
                    
                    if module.docstring:
                        f.write(f"{module.docstring}\n\n")
                        
                    # Link to detailed documentation
                    doc_link = f"{module.name.replace('.', '_')}.md"
                    f.write(f"[Detailed Documentation]({doc_link})\n\n")
                    
    def _generate_usage_examples(self, modules: List[DocModule]) -> None:
        """Generate usage examples documentation."""
        examples_file = self.docs_dir / "usage_examples.md"
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("# M.I.A Usage Examples\n\n")
            f.write("This document provides practical examples of using M.I.A components.\n\n")
            
            # Find examples in docstrings
            for module in modules:
                module_examples = []
                
                # Function examples
                for func in module.functions:
                    if func.examples:
                        module_examples.extend(func.examples)
                        
                # Class method examples
                for cls in module.classes:
                    for method in cls.methods:
                        if method.examples:
                            module_examples.extend(method.examples)
                            
                if module_examples:
                    f.write(f"## {module.name}\n\n")
                    for example in module_examples:
                        f.write(f"```python\n{example}\n```\n\n")
                        
    def _generate_index(self, modules: List[DocModule]) -> None:
        """Generate documentation index."""
        index_file = self.docs_dir / "index.md"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# M.I.A Documentation Index\n\n")
            f.write("Welcome to the M.I.A (Multimodal Intelligent Assistant) documentation.\n\n")
            
            f.write("## Quick Navigation\n\n")
            f.write("- [API Reference](api_reference.md)\n")
            f.write("- [Usage Examples](usage_examples.md)\n\n")
            
            f.write("## Modules\n\n")
            
            # Sort modules by name
            sorted_modules = sorted(modules, key=lambda m: m.name)
            
            for module in sorted_modules:
                doc_link = f"{module.name.replace('.', '_')}.md"
                f.write(f"- [{module.name}]({doc_link})")
                
                if module.docstring:
                    # Extract first line of docstring
                    first_line = module.docstring.split('\n')[0]
                    f.write(f" - {first_line}")
                    
                f.write("\n")
                
            f.write("\n")
            
            # Statistics
            total_classes = sum(len(module.classes) for module in modules)
            total_functions = sum(len(module.functions) for module in modules)
            
            f.write(f"## Statistics\n\n")
            f.write(f"- **Modules:** {len(modules)}\n")
            f.write(f"- **Classes:** {total_classes}\n")
            f.write(f"- **Functions:** {total_functions}\n")
            
def generate_documentation():
    """Generate comprehensive documentation for M.I.A."""
    generator = DocumentationGenerator()
    generator.generate_all_docs()
    print("Documentation generation completed!")

if __name__ == "__main__":
    generate_documentation()
