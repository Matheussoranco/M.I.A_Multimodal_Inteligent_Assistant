"""
Code Quality Manager - Automated code quality checks and improvements
"""
import ast
import os
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    file_path: str
    line_number: int
    column: int
    issue_type: str
    severity: str
    message: str
    rule_id: str
    suggestion: Optional[str] = None

@dataclass
class QualityReport:
    """Code quality report."""
    total_files: int
    issues_count: int
    issues: List[CodeIssue]
    quality_score: float
    suggestions: List[str]

class CodeQualityManager:
    """Manages code quality checks and improvements."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Quality rules configuration
        self.rules = {
            'max_line_length': 88,
            'max_function_length': 50,
            'max_class_length': 500,
            'max_complexity': 10,
            'min_docstring_length': 10,
            'required_type_hints': True,
            'max_arguments': 8,
            'max_nested_depth': 4
        }
        
    def run_quality_checks(self) -> QualityReport:
        """Run comprehensive code quality checks."""
        logger.info("Starting code quality analysis...")
        
        python_files = list(self.source_dir.rglob("*.py"))
        total_files = len(python_files)
        all_issues = []
        
        for py_file in python_files:
            try:
                file_issues = self._analyze_file(py_file)
                all_issues.extend(file_issues)
            except Exception as e:
                logger.error(f"Error analyzing {py_file}: {e}")
                
        # Calculate quality score
        quality_score = self._calculate_quality_score(total_files, all_issues)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(all_issues)
        
        report = QualityReport(
            total_files=total_files,
            issues_count=len(all_issues),
            issues=all_issues,
            quality_score=quality_score,
            suggestions=suggestions
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Quality analysis completed. Score: {quality_score:.2f}")
        return report
        
    def _analyze_file(self, file_path: Path) -> List[CodeIssue]:
        """Analyze a single file for quality issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Check various quality aspects
            issues.extend(self._check_line_length(file_path, content))
            issues.extend(self._check_docstrings(file_path, tree))
            issues.extend(self._check_complexity(file_path, tree))
            issues.extend(self._check_function_length(file_path, tree, content))
            issues.extend(self._check_type_hints(file_path, tree))
            issues.extend(self._check_naming_conventions(file_path, tree))
            issues.extend(self._check_imports(file_path, tree))
            issues.extend(self._check_code_style(file_path, content))
            
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=e.lineno or 1,
                column=e.offset or 1,
                issue_type="syntax",
                severity="error",
                message=f"Syntax error: {e.msg}",
                rule_id="E001"
            ))
            
        return issues
        
    def _check_line_length(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Check line length violations."""
        issues = []
        max_length = self.rules['max_line_length']
        
        for line_no, line in enumerate(content.split('\n'), 1):
            if len(line) > max_length:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=max_length + 1,
                    issue_type="style",
                    severity="warning",
                    message=f"Line too long ({len(line)} > {max_length} characters)",
                    rule_id="E501",
                    suggestion="Break long lines or use line continuation"
                ))
                
        return issues
        
    def _check_docstrings(self, file_path: Path, tree: ast.AST) -> List[CodeIssue]:
        """Check docstring presence and quality."""
        issues = []
        min_length = self.rules['min_docstring_length']
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                
                if not docstring:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="documentation",
                        severity="warning",
                        message=f"Missing docstring for {node.__class__.__name__.lower()} '{node.name}'",
                        rule_id="D100",
                        suggestion="Add descriptive docstring"
                    ))
                elif len(docstring) < min_length:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="documentation",
                        severity="info",
                        message=f"Short docstring for {node.__class__.__name__.lower()} '{node.name}'",
                        rule_id="D101",
                        suggestion="Expand docstring with more details"
                    ))
                    
        return issues
        
    def _check_complexity(self, file_path: Path, tree: ast.AST) -> List[CodeIssue]:
        """Check cyclomatic complexity."""
        issues = []
        max_complexity = self.rules['max_complexity']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                
                if complexity > max_complexity:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="complexity",
                        severity="warning",
                        message=f"High complexity in function '{node.name}' (complexity: {complexity})",
                        rule_id="C901",
                        suggestion="Consider breaking into smaller functions"
                    ))
                    
        return issues
        
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
        
    def _check_function_length(self, file_path: Path, tree: ast.AST, content: str) -> List[CodeIssue]:
        """Check function length."""
        issues = []
        max_length = self.rules['max_function_length']
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate function length
                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if node.end_lineno else len(lines)
                function_length = end_line - start_line + 1
                
                if function_length > max_length:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="design",
                        severity="warning",
                        message=f"Long function '{node.name}' ({function_length} lines)",
                        rule_id="D200",
                        suggestion="Consider breaking into smaller functions"
                    ))
                    
        return issues
        
    def _check_type_hints(self, file_path: Path, tree: ast.AST) -> List[CodeIssue]:
        """Check type hints usage."""
        issues = []
        
        if not self.rules['required_type_hints']:
            return issues
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function parameters
                for arg in node.args.args:
                    if not arg.annotation and arg.arg != 'self':
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type="typing",
                            severity="info",
                            message=f"Missing type hint for parameter '{arg.arg}' in function '{node.name}'",
                            rule_id="T100",
                            suggestion="Add type annotation"
                        ))
                        
                # Check return type
                if not node.returns and node.name not in ['__init__', '__del__']:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="typing",
                        severity="info",
                        message=f"Missing return type hint for function '{node.name}'",
                        rule_id="T101",
                        suggestion="Add return type annotation"
                    ))
                    
        return issues
        
    def _check_naming_conventions(self, file_path: Path, tree: ast.AST) -> List[CodeIssue]:
        """Check naming conventions."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check snake_case for functions
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('__'):
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="naming",
                        severity="info",
                        message=f"Function name '{node.name}' should be snake_case",
                        rule_id="N801",
                        suggestion="Use snake_case naming"
                    ))
                    
            elif isinstance(node, ast.ClassDef):
                # Check PascalCase for classes
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="naming",
                        severity="info",
                        message=f"Class name '{node.name}' should be PascalCase",
                        rule_id="N802",
                        suggestion="Use PascalCase naming"
                    ))
                    
        return issues
        
    def _check_imports(self, file_path: Path, tree: ast.AST) -> List[CodeIssue]:
        """Check import style and organization."""
        issues = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append((node.lineno, node))
                
        # Check import order (should be: standard library, third-party, local)
        prev_type = None
        for line_no, node in imports:
            import_type = self._get_import_type(node)
            
            if prev_type and import_type < prev_type:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=node.col_offset,
                    issue_type="import",
                    severity="info",
                    message="Imports not properly ordered",
                    rule_id="I100",
                    suggestion="Order imports: standard library, third-party, local"
                ))
                
            prev_type = import_type
            
        return issues
        
    def _get_import_type(self, node: ast.AST) -> int:
        """Get import type (0: standard, 1: third-party, 2: local)."""
        if isinstance(node, ast.Import):
            module_name = node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
        else:
            return 1
            
        # Standard library modules (simplified check)
        stdlib_modules = {
            'os', 'sys', 'time', 'json', 'logging', 'threading', 'subprocess',
            'pathlib', 'typing', 'dataclasses', 'collections', 'functools',
            'itertools', 'asyncio', 'multiprocessing', 'urllib', 'http',
            'datetime', 'random', 'math', 'sqlite3', 'pickle', 'hashlib'
        }
        
        first_part = module_name.split('.')[0]
        
        if first_part in stdlib_modules:
            return 0  # Standard library
        elif module_name.startswith('.') or module_name.startswith('src.'):
            return 2  # Local
        else:
            return 1  # Third-party
            
    def _check_code_style(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Check general code style issues."""
        issues = []
        
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            # Check trailing whitespace
            if line.rstrip() != line:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=len(line.rstrip()) + 1,
                    issue_type="style",
                    severity="info",
                    message="Trailing whitespace",
                    rule_id="W291",
                    suggestion="Remove trailing whitespace"
                ))
                
            # Check tabs vs spaces
            if '\t' in line:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=line.find('\t') + 1,
                    issue_type="style",
                    severity="info",
                    message="Tab character found, use spaces",
                    rule_id="W191",
                    suggestion="Use 4 spaces instead of tabs"
                ))
                
        return issues
        
    def _calculate_quality_score(self, total_files: int, issues: List[CodeIssue]) -> float:
        """Calculate overall quality score."""
        if not issues:
            return 100.0
            
        # Weight different severity levels
        severity_weights = {
            'error': 10,
            'warning': 5,
            'info': 1
        }
        
        weighted_issues = sum(severity_weights.get(issue.severity, 1) for issue in issues)
        
        # Calculate score (higher is better)
        max_possible_score = total_files * 100
        deduction = min(weighted_issues * 2, max_possible_score * 0.8)
        
        score = max(0, max_possible_score - deduction) / max_possible_score * 100
        return round(score, 2)
        
    def _generate_suggestions(self, issues: List[CodeIssue]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Count issue types
        issue_types = {}
        for issue in issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
            
        # Generate suggestions based on most common issues
        sorted_types = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
        
        for issue_type, count in sorted_types[:5]:  # Top 5 issues
            if issue_type == "style":
                suggestions.append(f"Fix {count} style issues by running automated formatter")
            elif issue_type == "documentation":
                suggestions.append(f"Add documentation for {count} missing docstrings")
            elif issue_type == "complexity":
                suggestions.append(f"Reduce complexity in {count} functions")
            elif issue_type == "typing":
                suggestions.append(f"Add type hints for {count} functions/parameters")
            elif issue_type == "naming":
                suggestions.append(f"Fix {count} naming convention violations")
                
        return suggestions
        
    def _save_report(self, report: QualityReport) -> None:
        """Save quality report to file."""
        report_file = self.reports_dir / "quality_report.json"
        
        # Convert to serializable format
        report_data = {
            'total_files': report.total_files,
            'issues_count': report.issues_count,
            'quality_score': report.quality_score,
            'suggestions': report.suggestions,
            'issues': []
        }
        
        for issue in report.issues:
            report_data['issues'].append({
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'column': issue.column,
                'issue_type': issue.issue_type,
                'severity': issue.severity,
                'message': issue.message,
                'rule_id': issue.rule_id,
                'suggestion': issue.suggestion
            })
            
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Quality report saved to {report_file}")
        
    def run_automated_fixes(self) -> None:
        """Run automated code fixes."""
        logger.info("Running automated code fixes...")
        
        # Run black formatter
        try:
            subprocess.run([
                'python', '-m', 'black', 
                '--line-length', str(self.rules['max_line_length']),
                str(self.source_dir)
            ], check=True)
            logger.info("Black formatter applied successfully")
        except subprocess.CalledProcessError:
            logger.warning("Black formatter not available or failed")
        except FileNotFoundError:
            logger.warning("Black formatter not installed")
            
        # Run isort for import sorting
        try:
            subprocess.run([
                'python', '-m', 'isort',
                str(self.source_dir)
            ], check=True)
            logger.info("Import sorting applied successfully")
        except subprocess.CalledProcessError:
            logger.warning("isort not available or failed")
        except FileNotFoundError:
            logger.warning("isort not installed")
            
        logger.info("Automated fixes completed")
        
    def generate_quality_report_html(self, report: QualityReport) -> None:
        """Generate HTML quality report."""
        html_file = self.reports_dir / "quality_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>M.I.A Code Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .issue.warning {{ border-left-color: orange; }}
                .issue.error {{ border-left-color: red; }}
                .issue.info {{ border-left-color: blue; }}
                .suggestions {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>M.I.A Code Quality Report</h1>
                <div class="score">Quality Score: {report.quality_score:.1f}/100</div>
                <p>Total Files: {report.total_files} | Issues: {report.issues_count}</p>
            </div>
            
            <div class="suggestions">
                <h2>Improvement Suggestions</h2>
                <ul>
                    {''.join(f'<li>{suggestion}</li>' for suggestion in report.suggestions)}
                </ul>
            </div>
            
            <h2>Issues Details</h2>
        """
        
        # Group issues by file
        issues_by_file = {}
        for issue in report.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
            
        for file_path, file_issues in issues_by_file.items():
            html_content += f"<h3>{file_path}</h3>"
            
            for issue in file_issues:
                html_content += f"""
                <div class="issue {issue.severity}">
                    <strong>Line {issue.line_number}:</strong> {issue.message}
                    <br><small>Type: {issue.issue_type} | Severity: {issue.severity} | Rule: {issue.rule_id}</small>
                    {f'<br><em>Suggestion: {issue.suggestion}</em>' if issue.suggestion else ''}
                </div>
                """
                
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated: {html_file}")

def run_quality_check():
    """Run comprehensive code quality check."""
    manager = CodeQualityManager()
    report = manager.run_quality_checks()
    manager.generate_quality_report_html(report)
    
    print(f"Quality Score: {report.quality_score:.1f}/100")
    print(f"Total Issues: {report.issues_count}")
    print(f"Files Analyzed: {report.total_files}")
    
    if report.suggestions:
        print("\nTop Suggestions:")
        for suggestion in report.suggestions[:3]:
            print(f"  - {suggestion}")
            
    return report

if __name__ == "__main__":
    run_quality_check()
