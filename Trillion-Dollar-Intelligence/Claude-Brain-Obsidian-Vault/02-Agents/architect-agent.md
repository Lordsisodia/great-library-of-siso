# Architect Agent - Technical Design Specialization

## Agent Configuration

```yaml
role: architect
instructions: |
  You are a Senior Software Architect with 15+ years of experience in system design.
  Your role is to translate product requirements into robust technical architectures.
  
  Core Responsibilities:
  - Design scalable system architectures from product requirements
  - Create detailed technical specifications and API designs
  - Select optimal technology stacks and architectural patterns
  - Plan database schemas and data flow diagrams
  - Ensure non-functional requirements (performance, security, scalability)
  - Design integration patterns and service boundaries
  
  When creating technical designs:
  1. Apply first principles thinking to system design
  2. Consider scalability, maintainability, and performance
  3. Choose proven architectural patterns and technologies
  4. Design for testability and monitoring
  5. Plan for deployment and operational requirements
  6. Consider security and compliance from the ground up

tools:
  - system_design_generator
  - technology_evaluator
  - database_designer
  - api_designer
  - security_analyzer
  - performance_modeler

handoff_targets:
  - developer
  - qa_engineer
  - product_manager

guardrails:
  architectural_consistency: true
  technology_validation: true
  scalability_check: true
  security_review: true

model_config:
  temperature: 0.2
  max_tokens: 6000
  reasoning_mode: systematic
```

## Implementation

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ArchitecturalPattern(Enum):
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"

@dataclass
class TechnologyStack:
    """Technology stack recommendation"""
    frontend: List[str]
    backend: List[str]
    database: List[str]
    infrastructure: List[str]
    monitoring: List[str]
    security: List[str]

@dataclass
class SystemComponent:
    """Individual system component design"""
    name: str
    purpose: str
    responsibilities: List[str]
    interfaces: List[str]
    dependencies: List[str]
    technology_choice: str
    scalability_considerations: str

@dataclass
class TechnicalSpecification:
    """Comprehensive technical architecture output"""
    system_overview: str
    architectural_pattern: ArchitecturalPattern
    technology_stack: TechnologyStack
    system_components: List[SystemComponent]
    api_specifications: Dict[str, Any]
    database_design: Dict[str, Any]
    data_flow: Dict[str, Any]
    security_architecture: Dict[str, Any]
    deployment_strategy: Dict[str, Any]
    monitoring_strategy: Dict[str, Any]
    performance_requirements: Dict[str, Any]

class ArchitectAgent(Agent):
    """Software Architect specialized agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.design_patterns = self._load_design_patterns()
        self.technology_matrix = self._load_technology_matrix()
        self.architectural_principles = self._load_architectural_principles()
        
    def process_task(self, task: str, context: Dict[str, Any]) -> TechnicalSpecification:
        """
        Process product requirements into technical architecture
        """
        
        # Extract product analysis from PM handoff
        product_analysis = context.get("product_analysis", {})
        user_stories = product_analysis.get("user_stories", [])
        
        # Step 1: Analyze technical requirements
        tech_requirements = self._analyze_technical_requirements(user_stories)
        
        # Step 2: Select architectural pattern
        architectural_pattern = self._select_architectural_pattern(tech_requirements)
        
        # Step 3: Choose technology stack
        technology_stack = self._choose_technology_stack(tech_requirements, architectural_pattern)
        
        # Step 4: Design system components
        system_components = self._design_system_components(user_stories, architectural_pattern)
        
        # Step 5: Create API specifications
        api_specifications = self._design_apis(system_components, user_stories)
        
        # Step 6: Design database schema
        database_design = self._design_database(user_stories, system_components)
        
        # Step 7: Plan data flow
        data_flow = self._design_data_flow(system_components, api_specifications)
        
        # Step 8: Design security architecture
        security_architecture = self._design_security(tech_requirements, system_components)
        
        # Step 9: Plan deployment strategy
        deployment_strategy = self._plan_deployment(architectural_pattern, technology_stack)
        
        # Step 10: Design monitoring strategy
        monitoring_strategy = self._design_monitoring(system_components, tech_requirements)
        
        # Step 11: Define performance requirements
        performance_requirements = self._define_performance_requirements(tech_requirements)
        
        return TechnicalSpecification(
            system_overview=self._generate_system_overview(tech_requirements, architectural_pattern),
            architectural_pattern=architectural_pattern,
            technology_stack=technology_stack,
            system_components=system_components,
            api_specifications=api_specifications,
            database_design=database_design,
            data_flow=data_flow,
            security_architecture=security_architecture,
            deployment_strategy=deployment_strategy,
            monitoring_strategy=monitoring_strategy,
            performance_requirements=performance_requirements
        )
    
    def _analyze_technical_requirements(self, user_stories: List[Dict]) -> Dict[str, Any]:
        """Extract technical requirements from user stories"""
        
        analysis_prompt = f"""
        Analyze these user stories to extract technical requirements:
        
        User Stories: {json.dumps(user_stories, indent=2)}
        
        Extract:
        1. Functional requirements (what the system must do)
        2. Non-functional requirements (performance, scalability, security)
        3. Data requirements (what data needs to be stored/processed)
        4. Integration requirements (external systems, APIs)
        5. User experience requirements (response times, availability)
        6. Compliance requirements (regulations, standards)
        7. Operational requirements (monitoring, logging, backup)
        
        Apply first principles thinking to identify core technical challenges.
        """
        
        result = self._execute_with_reasoning(analysis_prompt)
        return json.loads(result)
    
    def _select_architectural_pattern(self, tech_requirements: Dict[str, Any]) -> ArchitecturalPattern:
        """Select optimal architectural pattern based on requirements"""
        
        selection_prompt = f"""
        Select the optimal architectural pattern for these requirements:
        
        Requirements: {json.dumps(tech_requirements, indent=2)}
        
        Available patterns:
        1. Microservices - For complex, scalable, distributed systems
        2. Monolith - For simple to moderate complexity, faster initial development
        3. Serverless - For event-driven, highly scalable, cost-effective solutions
        4. Event-driven - For real-time, asynchronous, loosely coupled systems
        5. Layered - For traditional enterprise applications with clear separation
        6. Hexagonal - For domain-driven design with external adapters
        
        Consider:
        - System complexity and scale requirements
        - Team size and experience
        - Performance and latency requirements
        - Deployment and operational complexity
        - Cost constraints and resource limitations
        
        Recommend ONE pattern with detailed reasoning.
        """
        
        result = self._execute_with_reasoning(selection_prompt)
        pattern_name = json.loads(result)["recommended_pattern"]
        return ArchitecturalPattern(pattern_name.lower())
    
    def _choose_technology_stack(self, tech_requirements: Dict[str, Any], pattern: ArchitecturalPattern) -> TechnologyStack:
        """Choose optimal technology stack for the architecture"""
        
        stack_prompt = f"""
        Choose technology stack for these requirements and architectural pattern:
        
        Requirements: {json.dumps(tech_requirements, indent=2)}
        Architectural Pattern: {pattern.value}
        
        Select technologies for:
        1. Frontend (frameworks, libraries, tools)
        2. Backend (languages, frameworks, runtime)
        3. Database (primary, cache, search, analytics)
        4. Infrastructure (cloud, containers, orchestration)
        5. Monitoring (logging, metrics, alerting, tracing)
        6. Security (authentication, authorization, encryption)
        
        Consider:
        - Team expertise and learning curve
        - Performance and scalability requirements
        - Ecosystem maturity and community support
        - Cost and licensing implications
        - Operational complexity and maintenance
        
        Provide specific technology recommendations with reasoning.
        """
        
        result = self._execute_with_reasoning(stack_prompt)
        stack_data = json.loads(result)
        
        return TechnologyStack(
            frontend=stack_data["frontend"],
            backend=stack_data["backend"],
            database=stack_data["database"],
            infrastructure=stack_data["infrastructure"],
            monitoring=stack_data["monitoring"],
            security=stack_data["security"]
        )
    
    def _design_system_components(self, user_stories: List[Dict], pattern: ArchitecturalPattern) -> List[SystemComponent]:
        """Design individual system components based on user stories"""
        
        components = []
        
        # Group user stories by domain/functionality
        story_groups = self._group_stories_by_domain(user_stories)
        
        for domain, stories in story_groups.items():
            component = self._design_component(domain, stories, pattern)
            components.append(component)
        
        # Add cross-cutting components
        cross_cutting = self._design_cross_cutting_components(pattern)
        components.extend(cross_cutting)
        
        return components
    
    def _design_component(self, domain: str, stories: List[Dict], pattern: ArchitecturalPattern) -> SystemComponent:
        """Design individual component for a domain"""
        
        component_prompt = f"""
        Design a system component for this domain:
        
        Domain: {domain}
        User Stories: {json.dumps(stories, indent=2)}
        Architectural Pattern: {pattern.value}
        
        Define:
        1. Component name and purpose
        2. Core responsibilities
        3. Public interfaces (APIs, events)
        4. Dependencies on other components
        5. Technology implementation choice
        6. Scalability considerations
        
        Follow {pattern.value} architectural principles.
        """
        
        result = self._execute_with_reasoning(component_prompt)
        component_data = json.loads(result)
        
        return SystemComponent(
            name=component_data["name"],
            purpose=component_data["purpose"],
            responsibilities=component_data["responsibilities"],
            interfaces=component_data["interfaces"],
            dependencies=component_data["dependencies"],
            technology_choice=component_data["technology_choice"],
            scalability_considerations=component_data["scalability_considerations"]
        )
    
    def _design_apis(self, components: List[SystemComponent], user_stories: List[Dict]) -> Dict[str, Any]:
        """Design API specifications for system components"""
        
        api_prompt = f"""
        Design API specifications for these system components:
        
        Components: {[{"name": c.name, "responsibilities": c.responsibilities} for c in components]}
        User Stories: {json.dumps(user_stories, indent=2)}
        
        For each API, provide:
        1. OpenAPI 3.0 specification
        2. Endpoint definitions with methods and paths
        3. Request/response schemas
        4. Authentication and authorization requirements
        5. Rate limiting and throttling policies
        6. Error handling and status codes
        7. Versioning strategy
        
        Ensure APIs are RESTful, consistent, and developer-friendly.
        """
        
        result = self._execute_with_reasoning(api_prompt)
        return json.loads(result)
    
    def _design_database(self, user_stories: List[Dict], components: List[SystemComponent]) -> Dict[str, Any]:
        """Design database schema and data architecture"""
        
        database_prompt = f"""
        Design database architecture for this system:
        
        User Stories: {json.dumps(user_stories, indent=2)}
        Components: {[{"name": c.name, "responsibilities": c.responsibilities} for c in components]}
        
        Design:
        1. Logical data model (entities, relationships, attributes)
        2. Physical database schema (tables, indexes, constraints)
        3. Data partitioning and sharding strategy
        4. Backup and recovery procedures
        5. Data migration and versioning strategy
        6. Performance optimization (indexes, queries)
        7. Data governance and compliance
        
        Consider ACID properties, scalability, and performance requirements.
        """
        
        result = self._execute_with_reasoning(database_prompt)
        return json.loads(result)
    
    def _design_security(self, tech_requirements: Dict[str, Any], components: List[SystemComponent]) -> Dict[str, Any]:
        """Design comprehensive security architecture"""
        
        security_prompt = f"""
        Design security architecture for this system:
        
        Requirements: {json.dumps(tech_requirements, indent=2)}
        Components: {[c.name for c in components]}
        
        Design security for:
        1. Authentication and identity management
        2. Authorization and access control
        3. Data encryption (in transit and at rest)
        4. Network security and firewalls
        5. API security and rate limiting
        6. Vulnerability management and patching
        7. Security monitoring and incident response
        8. Compliance requirements (GDPR, SOX, etc.)
        
        Follow security best practices and zero-trust principles.
        """
        
        result = self._execute_with_reasoning(security_prompt)
        return json.loads(result)
    
    def _plan_deployment(self, pattern: ArchitecturalPattern, stack: TechnologyStack) -> Dict[str, Any]:
        """Plan deployment strategy and infrastructure"""
        
        deployment_prompt = f"""
        Plan deployment strategy for this architecture:
        
        Architectural Pattern: {pattern.value}
        Technology Stack: {stack.__dict__}
        
        Plan:
        1. Deployment environments (dev, staging, prod)
        2. Infrastructure as Code (Terraform, CloudFormation)
        3. Container orchestration (Kubernetes, Docker)
        4. CI/CD pipeline and automation
        5. Blue-green and canary deployment strategies
        6. Rollback and disaster recovery procedures
        7. Scaling and auto-scaling policies
        8. Cost optimization strategies
        
        Ensure high availability and fault tolerance.
        """
        
        result = self._execute_with_reasoning(deployment_prompt)
        return json.loads(result)
    
    def request_handoff(self, target: AgentRole, specification: TechnicalSpecification) -> HandoffRequest:
        """Request handoff to developer with technical architecture"""
        
        handoff_context = {
            "technical_specification": specification.__dict__,
            "implementation_priorities": self._prioritize_implementation(specification),
            "technology_stack": specification.technology_stack.__dict__,
            "api_contracts": specification.api_specifications,
            "database_schema": specification.database_design,
            "security_requirements": specification.security_architecture
        }
        
        if target == AgentRole.DEVELOPER:
            task_description = "Implement system components following technical architecture and specifications"
        elif target == AgentRole.QA_ENGINEER:
            task_description = "Create comprehensive testing strategy based on technical architecture"
        else:
            task_description = f"Continue workflow with technical architecture"
            
        return HandoffRequest(
            target_agent=target,
            context=handoff_context,
            task_description=task_description,
            priority=1
        )
    
    def _prioritize_implementation(self, specification: TechnicalSpecification) -> List[str]:
        """Prioritize implementation order for development team"""
        
        prioritization_prompt = f"""
        Prioritize implementation order for these system components:
        
        Components: {[c.name for c in specification.system_components]}
        Dependencies: {[(c.name, c.dependencies) for c in specification.system_components]}
        
        Create implementation phases considering:
        1. Component dependencies
        2. Risk and complexity
        3. Business value delivery
        4. Team capacity and skills
        5. Testing and integration requirements
        
        Return ordered list of implementation phases.
        """
        
        result = self._execute_with_reasoning(prioritization_prompt)
        return json.loads(result)

# Utility functions for architectural design

def load_design_patterns() -> Dict[str, Any]:
    """Load common design patterns and their applications"""
    return {
        "creational": ["Singleton", "Factory", "Builder", "Prototype"],
        "structural": ["Adapter", "Facade", "Decorator", "Proxy"],
        "behavioral": ["Observer", "Strategy", "Command", "State"],
        "architectural": ["MVC", "MVP", "MVVM", "Clean Architecture"]
    }

def load_technology_matrix() -> Dict[str, Any]:
    """Load technology evaluation matrix"""
    return {
        "languages": {
            "python": {"strengths": ["AI/ML", "rapid development"], "weaknesses": ["performance"]},
            "typescript": {"strengths": ["type safety", "ecosystem"], "weaknesses": ["runtime overhead"]},
            "rust": {"strengths": ["performance", "safety"], "weaknesses": ["learning curve"]},
            "go": {"strengths": ["simplicity", "concurrency"], "weaknesses": ["limited ecosystem"]}
        },
        "databases": {
            "postgresql": {"strengths": ["ACID", "extensions"], "weaknesses": ["scaling complexity"]},
            "mongodb": {"strengths": ["flexibility", "scaling"], "weaknesses": ["consistency"]},
            "redis": {"strengths": ["performance", "caching"], "weaknesses": ["memory usage"]}
        }
    }

def load_architectural_principles() -> List[str]:
    """Load architectural principles for design guidance"""
    return [
        "Single Responsibility Principle",
        "Open/Closed Principle", 
        "Liskov Substitution Principle",
        "Interface Segregation Principle",
        "Dependency Inversion Principle",
        "Don't Repeat Yourself (DRY)",
        "Keep It Simple, Stupid (KISS)",
        "You Aren't Gonna Need It (YAGNI)",
        "Separation of Concerns",
        "Principle of Least Surprise"
    ]
```

## Example Output

```json
{
  "system_overview": "Microservices architecture with event-driven communication, designed for scalability and maintainability",
  "architectural_pattern": "microservices",
  "technology_stack": {
    "frontend": ["React", "TypeScript", "Tailwind CSS"],
    "backend": ["Node.js", "Express", "TypeScript"],
    "database": ["PostgreSQL", "Redis", "MongoDB"],
    "infrastructure": ["Docker", "Kubernetes", "AWS"],
    "monitoring": ["Prometheus", "Grafana", "ELK Stack"],
    "security": ["JWT", "OAuth 2.0", "TLS 1.3"]
  },
  "system_components": [
    {
      "name": "UserService",
      "purpose": "Manage user authentication and profiles",
      "responsibilities": ["User registration", "Authentication", "Profile management"],
      "interfaces": ["/api/users", "/api/auth"],
      "dependencies": ["DatabaseService", "NotificationService"],
      "technology_choice": "Node.js with PostgreSQL",
      "scalability_considerations": "Horizontal scaling with load balancing"
    }
  ],
  "api_specifications": {
    "UserService": {
      "openapi": "3.0.0",
      "paths": {
        "/api/users": {
          "get": {"summary": "Get user list", "responses": {"200": {"description": "Success"}}}
        }
      }
    }
  }
}
```

The Architect Agent ensures that product requirements are translated into robust, scalable technical designs that development teams can implement effectively.

---
*Status: Architect Agent Implementation - Ready for Integration*
*Next: Developer Agent Implementation*