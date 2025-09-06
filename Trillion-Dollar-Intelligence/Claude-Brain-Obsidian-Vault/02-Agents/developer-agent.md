# Developer Agent - Code Implementation Specialization

## Agent Configuration

```yaml
role: developer
instructions: |
  You are a Senior Full-Stack Developer with 12+ years of experience across multiple languages and frameworks.
  Your role is to implement robust, production-ready code from technical specifications.
  
  Core Responsibilities:
  - Generate clean, maintainable, and well-documented code
  - Implement features following architecture specifications and best practices
  - Handle multiple programming languages and frameworks efficiently
  - Optimize code for performance, security, and maintainability
  - Create comprehensive documentation and code comments
  - Follow test-driven development practices
  
  When implementing code:
  1. Follow existing code conventions and architectural patterns
  2. Implement proper error handling and logging
  3. Write self-documenting code with clear variable names
  4. Include comprehensive inline documentation
  5. Consider performance and scalability implications
  6. Implement security best practices from the ground up
  7. Create modular, reusable components

tools:
  - code_generator
  - syntax_validator
  - performance_analyzer
  - security_scanner
  - documentation_generator
  - dependency_manager
  - git_integration

handoff_targets:
  - qa_engineer
  - architect
  - devops

guardrails:
  code_quality_check: true
  security_scan: true
  performance_validation: true
  documentation_required: true
  test_coverage_minimum: 80

model_config:
  temperature: 0.1
  max_tokens: 8000
  reasoning_mode: systematic
```

## Implementation

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import ast
import re

class ProgrammingLanguage(Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CSHARP = "csharp"

class CodeQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"

@dataclass
class CodeModule:
    """Individual code module or component"""
    name: str
    language: ProgrammingLanguage
    code: str
    documentation: str
    tests: str
    dependencies: List[str]
    entry_points: List[str]
    quality_score: float

@dataclass
class ImplementationResult:
    """Complete implementation output"""
    modules: List[CodeModule]
    documentation: str
    deployment_instructions: str
    configuration_files: Dict[str, str]
    database_migrations: List[str]
    api_documentation: str
    quality_report: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class DeveloperAgent(Agent):
    """Developer specialized agent for code implementation"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.code_templates = self._load_code_templates()
        self.best_practices = self._load_best_practices()
        self.security_patterns = self._load_security_patterns()
        
    def process_task(self, task: str, context: Dict[str, Any]) -> ImplementationResult:
        """
        Process technical specifications into production-ready code
        """
        
        # Extract technical specification from architect handoff
        tech_spec = context.get("technical_specification", {})
        components = tech_spec.get("system_components", [])
        technology_stack = tech_spec.get("technology_stack", {})
        
        # Step 1: Plan implementation approach
        implementation_plan = self._create_implementation_plan(components, technology_stack)
        
        # Step 2: Generate code modules
        modules = self._generate_code_modules(implementation_plan, tech_spec)
        
        # Step 3: Create configuration files
        config_files = self._generate_configuration_files(tech_spec)
        
        # Step 4: Generate database migrations
        migrations = self._generate_database_migrations(tech_spec.get("database_design", {}))
        
        # Step 5: Create API documentation
        api_docs = self._generate_api_documentation(tech_spec.get("api_specifications", {}))
        
        # Step 6: Generate comprehensive documentation
        documentation = self._generate_project_documentation(implementation_plan, modules)
        
        # Step 7: Create deployment instructions
        deployment_instructions = self._generate_deployment_instructions(tech_spec)
        
        # Step 8: Analyze code quality and performance
        quality_report = self._analyze_code_quality(modules)
        performance_metrics = self._analyze_performance(modules, tech_spec)
        
        return ImplementationResult(
            modules=modules,
            documentation=documentation,
            deployment_instructions=deployment_instructions,
            configuration_files=config_files,
            database_migrations=migrations,
            api_documentation=api_docs,
            quality_report=quality_report,
            performance_metrics=performance_metrics
        )
    
    def _create_implementation_plan(self, components: List[Dict], technology_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed implementation plan from architecture"""
        
        planning_prompt = f"""
        Create implementation plan for these system components:
        
        Components: {json.dumps(components, indent=2)}
        Technology Stack: {json.dumps(technology_stack, indent=2)}
        
        Plan should include:
        1. Implementation order and dependencies
        2. Code structure and organization
        3. Module boundaries and interfaces
        4. Shared libraries and utilities
        5. Configuration and environment setup
        6. Database schema implementation
        7. API endpoint implementation
        8. Testing strategy and approach
        9. Security implementation points
        10. Performance optimization opportunities
        
        Focus on production-ready, maintainable code architecture.
        """
        
        result = self._execute_with_reasoning(planning_prompt)
        return json.loads(result)
    
    def _generate_code_modules(self, plan: Dict[str, Any], tech_spec: Dict[str, Any]) -> List[CodeModule]:
        """Generate individual code modules based on implementation plan"""
        
        modules = []
        
        for component in plan.get("components", []):
            module = self._implement_component(component, tech_spec, plan)
            modules.append(module)
        
        # Generate shared utilities and libraries
        shared_modules = self._generate_shared_modules(plan, tech_spec)
        modules.extend(shared_modules)
        
        return modules
    
    def _implement_component(self, component: Dict[str, Any], tech_spec: Dict[str, Any], plan: Dict[str, Any]) -> CodeModule:
        """Implement individual system component"""
        
        component_name = component.get("name", "")
        language = self._determine_language(component, tech_spec)
        
        implementation_prompt = f"""
        Implement this system component:
        
        Component: {json.dumps(component, indent=2)}
        Technical Specification: {json.dumps(tech_spec, indent=2)}
        Implementation Plan: {json.dumps(plan, indent=2)}
        Language: {language.value}
        
        Generate production-ready code including:
        1. Main implementation with proper structure
        2. Error handling and logging
        3. Input validation and sanitization
        4. Security best practices implementation
        5. Performance optimizations
        6. Comprehensive inline documentation
        7. Type hints and interfaces (where applicable)
        8. Configuration management
        9. Dependency injection and IoC
        10. Monitoring and observability hooks
        
        Follow {language.value} best practices and conventions.
        """
        
        code = self._execute_with_reasoning(implementation_prompt)
        
        # Generate tests for the component
        tests = self._generate_component_tests(component, code, language)
        
        # Generate documentation
        documentation = self._generate_component_documentation(component, code, language)
        
        # Analyze dependencies
        dependencies = self._extract_dependencies(code, language)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(code, language)
        
        return CodeModule(
            name=component_name,
            language=language,
            code=code,
            documentation=documentation,
            tests=tests,
            dependencies=dependencies,
            entry_points=self._extract_entry_points(code, language),
            quality_score=quality_score
        )
    
    def _generate_component_tests(self, component: Dict[str, Any], code: str, language: ProgrammingLanguage) -> str:
        """Generate comprehensive tests for component"""
        
        test_prompt = f"""
        Generate comprehensive test suite for this component:
        
        Component: {json.dumps(component, indent=2)}
        Implementation Code: {code[:2000]}...  # Truncated for prompt
        Language: {language.value}
        
        Create tests including:
        1. Unit tests for all public methods/functions
        2. Integration tests for component interactions
        3. Edge case and error condition tests
        4. Performance and load tests
        5. Security tests for input validation
        6. Mock tests for external dependencies
        7. Property-based tests (where applicable)
        
        Target minimum 90% code coverage.
        Use appropriate testing framework for {language.value}.
        """
        
        return self._execute_with_reasoning(test_prompt)
    
    def _generate_component_documentation(self, component: Dict[str, Any], code: str, language: ProgrammingLanguage) -> str:
        """Generate comprehensive documentation for component"""
        
        doc_prompt = f"""
        Generate comprehensive documentation for this component:
        
        Component: {json.dumps(component, indent=2)}
        Implementation: {code[:1500]}...  # Truncated for prompt
        Language: {language.value}
        
        Documentation should include:
        1. Component overview and purpose
        2. API reference with examples
        3. Configuration options and environment variables
        4. Dependency requirements and setup
        5. Usage examples and code samples
        6. Error handling and troubleshooting
        7. Performance considerations
        8. Security considerations
        9. Deployment and operational notes
        10. Contributing guidelines
        
        Use clear, developer-friendly language with practical examples.
        """
        
        return self._execute_with_reasoning(doc_prompt)
    
    def _generate_configuration_files(self, tech_spec: Dict[str, Any]) -> Dict[str, str]:
        """Generate configuration files for the project"""
        
        config_prompt = f"""
        Generate configuration files for this project:
        
        Technical Specification: {json.dumps(tech_spec, indent=2)}
        
        Generate configuration for:
        1. Package management (package.json, requirements.txt, Cargo.toml, etc.)
        2. Build configuration (webpack, Dockerfile, Makefile)
        3. Environment configuration (.env templates, config files)
        4. Database configuration (connection strings, migrations)
        5. CI/CD pipeline configuration (.github/workflows, .gitlab-ci.yml)
        6. Linting and formatting (.eslintrc, .prettierrc, pyproject.toml)
        7. Testing configuration (jest.config.js, pytest.ini)
        8. Deployment configuration (docker-compose.yml, k8s manifests)
        9. Monitoring configuration (prometheus, grafana)
        10. Security configuration (CORS, CSP, rate limiting)
        
        Return as dictionary mapping filename to file content.
        """
        
        result = self._execute_with_reasoning(config_prompt)
        return json.loads(result)
    
    def _generate_database_migrations(self, database_design: Dict[str, Any]) -> List[str]:
        """Generate database migration scripts"""
        
        if not database_design:
            return []
        
        migration_prompt = f"""
        Generate database migration scripts for this design:
        
        Database Design: {json.dumps(database_design, indent=2)}
        
        Create migrations for:
        1. Initial schema creation (tables, columns, types)
        2. Index creation for performance optimization
        3. Foreign key constraints and relationships
        4. Initial data population (seed data)
        5. Stored procedures and functions (if applicable)
        6. Views and materialized views
        7. Triggers and event handlers
        8. User roles and permissions
        
        Generate production-ready SQL migrations with proper error handling.
        Include rollback scripts for each migration.
        """
        
        result = self._execute_with_reasoning(migration_prompt)
        return json.loads(result)
    
    def _generate_api_documentation(self, api_specs: Dict[str, Any]) -> str:
        """Generate comprehensive API documentation"""
        
        if not api_specs:
            return ""
        
        api_doc_prompt = f"""
        Generate comprehensive API documentation:
        
        API Specifications: {json.dumps(api_specs, indent=2)}
        
        Create documentation including:
        1. API overview and authentication
        2. Endpoint documentation with examples
        3. Request/response schemas and examples
        4. Error codes and handling
        5. Rate limiting and pagination
        6. SDK examples in multiple languages
        7. Postman/Insomnia collection
        8. Interactive API explorer (OpenAPI/Swagger)
        9. WebSocket documentation (if applicable)
        10. Webhook documentation (if applicable)
        
        Use OpenAPI 3.0 specification format.
        Include practical examples and use cases.
        """
        
        return self._execute_with_reasoning(api_doc_prompt)
    
    def _analyze_code_quality(self, modules: List[CodeModule]) -> Dict[str, Any]:
        """Analyze code quality across all modules"""
        
        quality_metrics = {
            "overall_score": 0.0,
            "module_scores": {},
            "issues": [],
            "recommendations": [],
            "complexity_analysis": {},
            "security_analysis": {},
            "performance_analysis": {}
        }
        
        total_score = 0.0
        
        for module in modules:
            # Analyze individual module
            module_analysis = self._analyze_module_quality(module)
            quality_metrics["module_scores"][module.name] = module_analysis
            total_score += module.quality_score
            
            # Extract issues and recommendations
            quality_metrics["issues"].extend(module_analysis.get("issues", []))
            quality_metrics["recommendations"].extend(module_analysis.get("recommendations", []))
        
        quality_metrics["overall_score"] = total_score / len(modules) if modules else 0.0
        
        return quality_metrics
    
    def _analyze_module_quality(self, module: CodeModule) -> Dict[str, Any]:
        """Analyze quality of individual code module"""
        
        analysis_prompt = f"""
        Analyze code quality for this module:
        
        Module: {module.name}
        Language: {module.language.value}
        Code: {module.code[:3000]}...  # Truncated for analysis
        
        Analyze:
        1. Code complexity and maintainability
        2. Security vulnerabilities and best practices
        3. Performance bottlenecks and optimization opportunities
        4. Code style and convention adherence
        5. Error handling and edge case coverage
        6. Documentation quality and completeness
        7. Test coverage and quality
        8. Dependency management and security
        
        Provide specific issues found and actionable recommendations.
        Rate overall quality from 0.0 to 1.0.
        """
        
        result = self._execute_with_reasoning(analysis_prompt)
        return json.loads(result)
    
    def _determine_language(self, component: Dict[str, Any], tech_spec: Dict[str, Any]) -> ProgrammingLanguage:
        """Determine programming language for component"""
        
        # Extract technology choices from specification
        backend_tech = tech_spec.get("technology_stack", {}).get("backend", [])
        frontend_tech = tech_spec.get("technology_stack", {}).get("frontend", [])
        
        component_tech = component.get("technology_choice", "")
        
        # Simple language detection logic
        if "python" in component_tech.lower() or "python" in str(backend_tech).lower():
            return ProgrammingLanguage.PYTHON
        elif "typescript" in component_tech.lower() or "typescript" in str(frontend_tech).lower():
            return ProgrammingLanguage.TYPESCRIPT
        elif "javascript" in component_tech.lower() or "node.js" in component_tech.lower():
            return ProgrammingLanguage.JAVASCRIPT
        elif "rust" in component_tech.lower():
            return ProgrammingLanguage.RUST
        elif "go" in component_tech.lower():
            return ProgrammingLanguage.GO
        else:
            return ProgrammingLanguage.PYTHON  # Default fallback
    
    def request_handoff(self, target: AgentRole, implementation: ImplementationResult) -> HandoffRequest:
        """Request handoff to QA or DevOps with implementation"""
        
        handoff_context = {
            "implementation": {
                "modules": [{"name": m.name, "language": m.language.value, "quality_score": m.quality_score} for m in implementation.modules],
                "quality_report": implementation.quality_report,
                "performance_metrics": implementation.performance_metrics,
                "configuration_files": list(implementation.configuration_files.keys()),
                "api_documentation": implementation.api_documentation != "",
                "deployment_ready": bool(implementation.deployment_instructions)
            },
            "code_artifacts": {
                "total_modules": len(implementation.modules),
                "languages_used": list(set(m.language.value for m in implementation.modules)),
                "test_coverage": self._calculate_overall_test_coverage(implementation.modules),
                "documentation_coverage": self._calculate_documentation_coverage(implementation.modules)
            }
        }
        
        if target == AgentRole.QA_ENGINEER:
            task_description = "Test and validate implementation quality, security, and performance"
        elif target == AgentRole.DEVOPS:
            task_description = "Deploy and configure production infrastructure"
        else:
            task_description = f"Continue workflow with implementation artifacts"
            
        return HandoffRequest(
            target_agent=target,
            context=handoff_context,
            task_description=task_description,
            priority=1
        )

# Utility functions for development

def load_code_templates() -> Dict[str, str]:
    """Load code templates for different languages and patterns"""
    return {
        "python_class": """
class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, {init_params}):
        {init_body}
    
    def {method_name}(self, {method_params}) -> {return_type}:
        \"\"\"
        {method_description}
        \"\"\"
        {method_body}
        """,
        "typescript_interface": """
interface {interface_name} {
  {properties}
}

export class {class_name} implements {interface_name} {
  {implementation}
}
        """,
        "rest_api_endpoint": """
@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    try:
        {implementation}
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        """
    }

def load_best_practices() -> Dict[str, List[str]]:
    """Load best practices for different languages"""
    return {
        "python": [
            "Use type hints for all function parameters and return values",
            "Follow PEP 8 style guidelines",
            "Use docstrings for all public functions and classes",
            "Implement proper error handling with specific exceptions",
            "Use context managers for resource management",
            "Write unit tests for all public methods"
        ],
        "typescript": [
            "Use strict type checking",
            "Implement proper error boundaries",
            "Use async/await for asynchronous operations",
            "Follow consistent naming conventions",
            "Implement proper null checking",
            "Use interfaces for object type definitions"
        ]
    }

def load_security_patterns() -> Dict[str, List[str]]:
    """Load security patterns and best practices"""
    return {
        "input_validation": [
            "Validate all user inputs against expected formats",
            "Sanitize inputs to prevent injection attacks",
            "Use parameterized queries for database operations",
            "Implement proper authentication and authorization"
        ],
        "data_protection": [
            "Encrypt sensitive data at rest and in transit",
            "Use secure random number generation",
            "Implement proper session management",
            "Follow principle of least privilege"
        ]
    }
```

## Example Output

```json
{
  "modules": [
    {
      "name": "UserService",
      "language": "python",
      "code": "# Production-ready User Service implementation...",
      "documentation": "# User Service Documentation\n\nComprehensive user management service...",
      "tests": "# User Service Test Suite\n\nimport pytest...",
      "dependencies": ["fastapi", "sqlalchemy", "pydantic", "bcrypt"],
      "entry_points": ["/api/users", "/api/auth"],
      "quality_score": 0.92
    }
  ],
  "quality_report": {
    "overall_score": 0.89,
    "security_score": 0.95,
    "performance_score": 0.88,
    "maintainability_score": 0.85
  },
  "configuration_files": {
    "requirements.txt": "fastapi==0.104.1\nsqlalchemy==2.0.23...",
    "Dockerfile": "FROM python:3.11-slim\n...",
    "docker-compose.yml": "version: '3.8'\nservices:..."
  }
}
```

The Developer Agent produces production-ready code with comprehensive testing, documentation, and deployment configurations, ensuring high-quality implementation that can be immediately deployed.

---
*Status: Developer Agent Implementation - Ready for Integration*
*Next: QA Engineer Agent Implementation*