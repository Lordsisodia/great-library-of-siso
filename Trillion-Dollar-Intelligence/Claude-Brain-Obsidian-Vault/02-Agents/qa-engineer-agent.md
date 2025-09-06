# QA Engineer Agent - Testing and Quality Assurance Specialization

## Agent Configuration

```yaml
role: qa_engineer
instructions: |
  You are a Senior QA Engineer with 10+ years of experience in comprehensive testing strategies.
  Your role is to ensure the highest quality standards for all deliverables through systematic testing.
  
  Core Responsibilities:
  - Design and execute comprehensive test strategies and plans
  - Perform code review and quality analysis with security focus
  - Create automated test suites covering all testing pyramids
  - Validate functionality against requirements and acceptance criteria
  - Conduct performance, security, and accessibility testing
  - Generate detailed quality reports and recommendations
  
  When testing and validating:
  1. Create comprehensive test plans covering all user scenarios
  2. Implement automated testing at unit, integration, and E2E levels
  3. Perform security testing and vulnerability assessments
  4. Conduct performance and load testing
  5. Validate accessibility and usability standards
  6. Review code quality, maintainability, and best practices
  7. Generate actionable quality reports with specific recommendations

tools:
  - test_generator
  - security_scanner
  - performance_tester
  - accessibility_checker
  - code_quality_analyzer
  - vulnerability_scanner
  - load_testing_tools

handoff_targets:
  - developer
  - devops
  - product_manager

guardrails:
  test_coverage_minimum: 90
  security_compliance: true
  performance_standards: true
  accessibility_compliance: true

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

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    USABILITY = "usability"
    API = "api"

class TestPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    UNACCEPTABLE = "unacceptable"

@dataclass
class TestCase:
    """Individual test case specification"""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    preconditions: List[str]
    test_steps: List[str]
    expected_results: List[str]
    test_data: Dict[str, Any]
    automation_script: str
    tags: List[str]

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    test_coverage: Dict[str, float]
    security_assessment: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    accessibility_score: float
    code_quality_metrics: Dict[str, Any]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]

@dataclass
class TestSuite:
    """Complete test suite with all test types"""
    test_cases: List[TestCase]
    automation_framework: str
    test_data_sets: Dict[str, Any]
    environment_requirements: Dict[str, Any]
    execution_instructions: str
    ci_cd_integration: str

@dataclass
class QAValidationResult:
    """Complete QA validation output"""
    test_suite: TestSuite
    quality_report: QualityReport
    security_audit: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    deployment_approval: bool
    remediation_plan: List[str]

class QAEngineerAgent(Agent):
    """QA Engineer specialized agent for testing and quality assurance"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.test_frameworks = self._load_test_frameworks()
        self.quality_standards = self._load_quality_standards()
        self.security_checklist = self._load_security_checklist()
        
    def process_task(self, task: str, context: Dict[str, Any]) -> QAValidationResult:
        """
        Process implementation artifacts through comprehensive QA validation
        """
        
        # Extract implementation details from developer handoff
        implementation = context.get("implementation", {})
        modules = implementation.get("modules", [])
        
        # Step 1: Create comprehensive test strategy
        test_strategy = self._create_test_strategy(implementation, context)
        
        # Step 2: Generate test suite
        test_suite = self._generate_test_suite(test_strategy, implementation)
        
        # Step 3: Perform code quality analysis
        code_quality = self._analyze_code_quality(modules)
        
        # Step 4: Conduct security audit
        security_audit = self._conduct_security_audit(implementation)
        
        # Step 5: Perform performance analysis
        performance_analysis = self._analyze_performance(implementation)
        
        # Step 6: Validate accessibility compliance
        accessibility_score = self._validate_accessibility(implementation)
        
        # Step 7: Generate comprehensive quality report
        quality_report = self._generate_quality_report(
            code_quality, security_audit, performance_analysis, accessibility_score
        )
        
        # Step 8: Determine deployment approval
        deployment_approval = self._evaluate_deployment_readiness(quality_report)
        
        # Step 9: Create remediation plan if needed
        remediation_plan = self._create_remediation_plan(quality_report) if not deployment_approval else []
        
        return QAValidationResult(
            test_suite=test_suite,
            quality_report=quality_report,
            security_audit=security_audit,
            performance_analysis=performance_analysis,
            deployment_approval=deployment_approval,
            remediation_plan=remediation_plan
        )
    
    def _create_test_strategy(self, implementation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive test strategy based on implementation"""
        
        strategy_prompt = f"""
        Create comprehensive test strategy for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        Context: {json.dumps(context, indent=2)}
        
        Define strategy for:
        1. Test pyramid approach (unit, integration, e2e ratios)
        2. Risk-based testing priorities
        3. Test environment requirements
        4. Test data management strategy
        5. Automation framework selection
        6. Performance testing approach
        7. Security testing methodology
        8. Accessibility testing requirements
        9. CI/CD integration strategy
        10. Test execution timeline and phases
        
        Focus on comprehensive coverage with efficient execution.
        """
        
        result = self._execute_with_reasoning(strategy_prompt)
        return json.loads(result)
    
    def _generate_test_suite(self, strategy: Dict[str, Any], implementation: Dict[str, Any]) -> TestSuite:
        """Generate comprehensive test suite based on strategy"""
        
        test_cases = []
        
        # Generate unit tests
        unit_tests = self._generate_unit_tests(implementation)
        test_cases.extend(unit_tests)
        
        # Generate integration tests
        integration_tests = self._generate_integration_tests(implementation)
        test_cases.extend(integration_tests)
        
        # Generate end-to-end tests
        e2e_tests = self._generate_e2e_tests(implementation)
        test_cases.extend(e2e_tests)
        
        # Generate performance tests
        performance_tests = self._generate_performance_tests(implementation)
        test_cases.extend(performance_tests)
        
        # Generate security tests
        security_tests = self._generate_security_tests(implementation)
        test_cases.extend(security_tests)
        
        # Generate API tests
        api_tests = self._generate_api_tests(implementation)
        test_cases.extend(api_tests)
        
        # Create automation framework
        automation_framework = self._design_automation_framework(strategy, implementation)
        
        # Create test data sets
        test_data_sets = self._create_test_data_sets(test_cases)
        
        # Define environment requirements
        environment_requirements = self._define_test_environments(strategy)
        
        # Create execution instructions
        execution_instructions = self._create_execution_instructions(test_cases, strategy)
        
        # Design CI/CD integration
        ci_cd_integration = self._design_ci_cd_integration(strategy, test_cases)
        
        return TestSuite(
            test_cases=test_cases,
            automation_framework=automation_framework,
            test_data_sets=test_data_sets,
            environment_requirements=environment_requirements,
            execution_instructions=execution_instructions,
            ci_cd_integration=ci_cd_integration
        )
    
    def _generate_unit_tests(self, implementation: Dict[str, Any]) -> List[TestCase]:
        """Generate comprehensive unit test cases"""
        
        unit_test_prompt = f"""
        Generate comprehensive unit test cases for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Create unit tests for:
        1. All public methods and functions
        2. Edge cases and boundary conditions
        3. Error handling and exception scenarios
        4. Input validation and sanitization
        5. Business logic validation
        6. State management and data mutations
        7. Mocking external dependencies
        8. Property-based testing scenarios
        
        Generate test cases with:
        - Clear test descriptions and objectives
        - Comprehensive test steps
        - Expected results and assertions
        - Test data and mock configurations
        - Automation scripts for execution
        
        Target 95%+ code coverage for critical components.
        """
        
        result = self._execute_with_reasoning(unit_test_prompt)
        return self._parse_test_cases(result, TestType.UNIT)
    
    def _generate_integration_tests(self, implementation: Dict[str, Any]) -> List[TestCase]:
        """Generate integration test cases"""
        
        integration_test_prompt = f"""
        Generate integration test cases for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Create integration tests for:
        1. Component interactions and interfaces
        2. Database integration and transactions
        3. External API integrations
        4. Message queue and event handling
        5. File system and storage operations
        6. Authentication and authorization flows
        7. Cross-service communication
        8. Data flow and transformation
        
        Focus on testing component boundaries and data contracts.
        Include both happy path and failure scenarios.
        """
        
        result = self._execute_with_reasoning(integration_test_prompt)
        return self._parse_test_cases(result, TestType.INTEGRATION)
    
    def _generate_e2e_tests(self, implementation: Dict[str, Any]) -> List[TestCase]:
        """Generate end-to-end test cases"""
        
        e2e_test_prompt = f"""
        Generate end-to-end test cases for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Create E2E tests for:
        1. Complete user journeys and workflows
        2. Critical business processes
        3. User interface interactions
        4. Cross-browser and device compatibility
        5. Performance under realistic load
        6. Error recovery and resilience
        7. Data consistency across the system
        8. Security and access control
        
        Use realistic user scenarios and data.
        Include both positive and negative test paths.
        """
        
        result = self._execute_with_reasoning(e2e_test_prompt)
        return self._parse_test_cases(result, TestType.END_TO_END)
    
    def _generate_security_tests(self, implementation: Dict[str, Any]) -> List[TestCase]:
        """Generate comprehensive security test cases"""
        
        security_test_prompt = f"""
        Generate security test cases for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Create security tests for:
        1. Input validation and injection attacks (SQL, XSS, CSRF)
        2. Authentication and authorization bypass attempts
        3. Session management and token security
        4. Data encryption and secure transmission
        5. Access control and privilege escalation
        6. API security and rate limiting
        7. File upload and download security
        8. Configuration and environment security
        
        Include OWASP Top 10 vulnerability testing.
        Test both automated scanning and manual penetration testing scenarios.
        """
        
        result = self._execute_with_reasoning(security_test_prompt)
        return self._parse_test_cases(result, TestType.SECURITY)
    
    def _conduct_security_audit(self, implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive security audit"""
        
        security_prompt = f"""
        Conduct comprehensive security audit for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Audit areas:
        1. Code security analysis (static analysis)
        2. Dependency vulnerability scanning
        3. Configuration security review
        4. Authentication and authorization mechanisms
        5. Data protection and encryption
        6. Network security and communication
        7. Input validation and output encoding
        8. Error handling and information disclosure
        9. Logging and monitoring security
        10. Compliance with security standards (OWASP, NIST)
        
        Provide:
        - Vulnerability assessment with severity ratings
        - Security score (0-100)
        - Specific remediation recommendations
        - Compliance gap analysis
        """
        
        result = self._execute_with_reasoning(security_prompt)
        return json.loads(result)
    
    def _analyze_performance(self, implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance characteristics and bottlenecks"""
        
        performance_prompt = f"""
        Analyze performance characteristics for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Analyze:
        1. Response time and latency characteristics
        2. Throughput and scalability limits
        3. Resource utilization (CPU, memory, I/O)
        4. Database query performance
        5. Network communication efficiency
        6. Caching strategy effectiveness
        7. Load balancing and distribution
        8. Bottleneck identification and resolution
        
        Provide:
        - Performance benchmarks and targets
        - Load testing scenarios and results
        - Optimization recommendations
        - Scalability planning
        """
        
        result = self._execute_with_reasoning(performance_prompt)
        return json.loads(result)
    
    def _validate_accessibility(self, implementation: Dict[str, Any]) -> float:
        """Validate accessibility compliance"""
        
        accessibility_prompt = f"""
        Validate accessibility compliance for this implementation:
        
        Implementation: {json.dumps(implementation, indent=2)}
        
        Check compliance with:
        1. WCAG 2.1 AA standards
        2. Section 508 requirements
        3. ADA compliance guidelines
        4. Keyboard navigation support
        5. Screen reader compatibility
        6. Color contrast and visual design
        7. Alternative text and descriptions
        8. Focus management and indicators
        
        Provide accessibility score (0-100) and specific recommendations.
        """
        
        result = self._execute_with_reasoning(accessibility_prompt)
        return json.loads(result).get("accessibility_score", 0.0)
    
    def _generate_quality_report(self, code_quality: Dict[str, Any], security_audit: Dict[str, Any], 
                               performance_analysis: Dict[str, Any], accessibility_score: float) -> QualityReport:
        """Generate comprehensive quality assessment report"""
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(
            code_quality, security_audit, performance_analysis, accessibility_score
        )
        
        # Compile test coverage metrics
        test_coverage = {
            "unit_tests": code_quality.get("test_coverage", {}).get("unit", 0.0),
            "integration_tests": code_quality.get("test_coverage", {}).get("integration", 0.0),
            "e2e_tests": code_quality.get("test_coverage", {}).get("e2e", 0.0),
            "overall": code_quality.get("test_coverage", {}).get("overall", 0.0)
        }
        
        # Compile issues from all analysis
        issues = []
        issues.extend(code_quality.get("issues", []))
        issues.extend(security_audit.get("vulnerabilities", []))
        issues.extend(performance_analysis.get("bottlenecks", []))
        
        # Compile recommendations
        recommendations = []
        recommendations.extend(code_quality.get("recommendations", []))
        recommendations.extend(security_audit.get("recommendations", []))
        recommendations.extend(performance_analysis.get("optimizations", []))
        
        # Assess risks
        risk_assessment = self._assess_quality_risks(
            code_quality, security_audit, performance_analysis, accessibility_score
        )
        
        return QualityReport(
            overall_score=overall_score,
            test_coverage=test_coverage,
            security_assessment=security_audit,
            performance_metrics=performance_analysis,
            accessibility_score=accessibility_score,
            code_quality_metrics=code_quality,
            issues=issues,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
    
    def _evaluate_deployment_readiness(self, quality_report: QualityReport) -> bool:
        """Evaluate if implementation is ready for deployment"""
        
        evaluation_prompt = f"""
        Evaluate deployment readiness based on quality report:
        
        Quality Report: {json.dumps(quality_report.__dict__, indent=2, default=str)}
        
        Evaluate against criteria:
        1. Overall quality score >= 0.85
        2. Test coverage >= 90%
        3. No critical security vulnerabilities
        4. Performance meets requirements
        5. Accessibility score >= 0.85
        6. No blocking issues or risks
        
        Return boolean decision with detailed reasoning.
        """
        
        result = self._execute_with_reasoning(evaluation_prompt)
        return json.loads(result).get("deployment_approved", False)
    
    def request_handoff(self, target: AgentRole, validation_result: QAValidationResult) -> HandoffRequest:
        """Request handoff with QA validation results"""
        
        handoff_context = {
            "qa_validation": {
                "deployment_approved": validation_result.deployment_approval,
                "overall_quality_score": validation_result.quality_report.overall_score,
                "test_coverage": validation_result.quality_report.test_coverage,
                "security_score": validation_result.security_audit.get("security_score", 0),
                "performance_score": validation_result.performance_analysis.get("performance_score", 0),
                "accessibility_score": validation_result.quality_report.accessibility_score,
                "critical_issues": len([i for i in validation_result.quality_report.issues if i.get("severity") == "critical"]),
                "total_test_cases": len(validation_result.test_suite.test_cases)
            },
            "recommendations": validation_result.quality_report.recommendations,
            "remediation_plan": validation_result.remediation_plan
        }
        
        if target == AgentRole.DEVOPS and validation_result.deployment_approval:
            task_description = "Deploy validated implementation to production environment"
        elif target == AgentRole.DEVELOPER and not validation_result.deployment_approval:
            task_description = "Address quality issues and implement remediation plan"
        elif target == AgentRole.PRODUCT_MANAGER:
            task_description = "Review quality assessment and approve next steps"
        else:
            task_description = f"Continue workflow with QA validation results"
            
        return HandoffRequest(
            target_agent=target,
            context=handoff_context,
            task_description=task_description,
            priority=1 if validation_result.deployment_approval else 2
        )

# Utility functions for QA operations

def load_test_frameworks() -> Dict[str, str]:
    """Load test framework recommendations by language"""
    return {
        "python": "pytest + pytest-asyncio + pytest-cov",
        "typescript": "Jest + Testing Library + Cypress",
        "javascript": "Jest + Mocha + Puppeteer",
        "rust": "cargo test + proptest",
        "go": "testing + testify + ginkgo"
    }

def load_quality_standards() -> Dict[str, float]:
    """Load quality standards and thresholds"""
    return {
        "minimum_test_coverage": 0.90,
        "minimum_code_quality": 0.85,
        "minimum_security_score": 0.90,
        "minimum_performance_score": 0.80,
        "minimum_accessibility_score": 0.85,
        "minimum_overall_score": 0.85
    }

def load_security_checklist() -> List[str]:
    """Load comprehensive security testing checklist"""
    return [
        "Input validation and sanitization",
        "SQL injection prevention",
        "XSS and CSRF protection",
        "Authentication and authorization",
        "Session management security",
        "Data encryption and protection",
        "Secure communication (HTTPS/TLS)",
        "Access control and permissions",
        "Error handling and information disclosure",
        "Dependency vulnerability scanning"
    ]
```

## Example Output

```json
{
  "test_suite": {
    "test_cases": [
      {
        "id": "TC001",
        "name": "User Registration - Valid Input",
        "description": "Test successful user registration with valid data",
        "test_type": "integration",
        "priority": "critical",
        "preconditions": ["Database is accessible", "API server is running"],
        "test_steps": ["Submit valid registration data", "Verify user creation", "Check email confirmation"],
        "expected_results": ["User created successfully", "Confirmation email sent", "User can login"],
        "automation_script": "test_user_registration_valid()",
        "tags": ["registration", "authentication", "smoke"]
      }
    ],
    "automation_framework": "pytest + requests + selenium",
    "test_data_sets": {"valid_users": [...], "invalid_inputs": [...]},
    "execution_instructions": "Run with: pytest tests/ --cov=src --cov-report=html"
  },
  "quality_report": {
    "overall_score": 0.89,
    "test_coverage": {"unit": 0.95, "integration": 0.88, "e2e": 0.75, "overall": 0.92},
    "security_assessment": {"security_score": 0.91, "vulnerabilities": []},
    "performance_metrics": {"response_time": "< 200ms", "throughput": "> 1000 rps"},
    "accessibility_score": 0.87,
    "deployment_approval": true
  }
}
```

The QA Engineer Agent ensures comprehensive quality validation through systematic testing, security auditing, and performance analysis, providing confidence for production deployment.

---
*Status: QA Engineer Agent Implementation - Complete*
*Phase 1 Multi-Agent Architecture - Ready for Testing*