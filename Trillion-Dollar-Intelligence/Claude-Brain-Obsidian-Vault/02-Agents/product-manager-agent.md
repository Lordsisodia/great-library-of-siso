# Product Manager Agent - Role Specialization Implementation

## Agent Configuration

```yaml
role: product_manager
instructions: |
  You are an expert Product Manager with 10+ years of experience in software development.
  Your role is to analyze requirements, create user stories, and provide strategic product guidance.
  
  Core Responsibilities:
  - Parse natural language requirements into structured specifications
  - Generate comprehensive user stories with acceptance criteria
  - Perform competitive analysis and market research
  - Create product roadmaps and feature prioritization
  - Ensure alignment between business goals and technical implementation
  
  When analyzing requirements:
  1. Break down complex requirements into atomic user stories
  2. Identify edge cases and potential issues
  3. Consider user experience and business impact
  4. Provide clear acceptance criteria
  5. Suggest metrics for success measurement

tools:
  - web_search
  - competitor_analysis
  - user_story_generator
  - requirements_parser
  - market_research

handoff_targets:
  - architect
  - qa_engineer
  - orchestrator

guardrails:
  output_validation: structured_format
  quality_threshold: 0.85
  completeness_check: true

model_config:
  temperature: 0.3
  max_tokens: 4000
  reasoning_mode: systematic
```

## Implementation

```python
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class UserStory:
    """Structured user story representation"""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: str  # High, Medium, Low
    estimated_effort: str  # XS, S, M, L, XL
    dependencies: List[str]
    business_value: str

@dataclass
class RequirementAnalysis:
    """Comprehensive requirement analysis output"""
    user_stories: List[UserStory]
    competitive_analysis: Dict[str, Any]
    market_research: Dict[str, Any]
    success_metrics: List[str]
    risk_assessment: Dict[str, Any]
    product_roadmap: Dict[str, Any]

class ProductManagerAgent(Agent):
    """Product Manager specialized agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.user_story_template = self._load_user_story_template()
        self.analysis_framework = self._load_analysis_framework()
        
    def process_task(self, task: str, context: Dict[str, Any]) -> RequirementAnalysis:
        """
        Process requirements with Product Manager expertise
        """
        
        # Step 1: Parse and understand the requirement
        parsed_requirement = self._parse_requirement(task)
        
        # Step 2: Generate user stories
        user_stories = self._generate_user_stories(parsed_requirement)
        
        # Step 3: Perform competitive analysis
        competitive_analysis = self._analyze_competitors(parsed_requirement)
        
        # Step 4: Conduct market research
        market_research = self._conduct_market_research(parsed_requirement)
        
        # Step 5: Define success metrics
        success_metrics = self._define_success_metrics(parsed_requirement)
        
        # Step 6: Assess risks
        risk_assessment = self._assess_risks(parsed_requirement, user_stories)
        
        # Step 7: Create product roadmap
        product_roadmap = self._create_roadmap(user_stories)
        
        return RequirementAnalysis(
            user_stories=user_stories,
            competitive_analysis=competitive_analysis,
            market_research=market_research,
            success_metrics=success_metrics,
            risk_assessment=risk_assessment,
            product_roadmap=product_roadmap
        )
    
    def _parse_requirement(self, requirement: str) -> Dict[str, Any]:
        """Parse natural language requirement into structured format"""
        
        parsing_prompt = f"""
        Analyze this requirement and extract key information:
        
        Requirement: {requirement}
        
        Extract:
        1. Core functionality needed
        2. Target users/personas
        3. Business objectives
        4. Constraints and limitations
        5. Success criteria (if mentioned)
        6. Technical considerations (if any)
        
        Return structured JSON format.
        """
        
        # Use reasoning framework for systematic analysis
        result = self._execute_with_reasoning(parsing_prompt)
        return json.loads(result)
    
    def _generate_user_stories(self, parsed_requirement: Dict[str, Any]) -> List[UserStory]:
        """Generate comprehensive user stories from parsed requirements"""
        
        user_stories = []
        core_functionality = parsed_requirement.get("core_functionality", [])
        
        for functionality in core_functionality:
            # Generate main user story
            main_story = self._create_user_story(functionality, "main")
            user_stories.append(main_story)
            
            # Generate edge case stories
            edge_cases = self._identify_edge_cases(functionality)
            for edge_case in edge_cases:
                edge_story = self._create_user_story(edge_case, "edge_case")
                user_stories.append(edge_story)
        
        # Generate cross-cutting concern stories
        cross_cutting = self._identify_cross_cutting_concerns(parsed_requirement)
        for concern in cross_cutting:
            concern_story = self._create_user_story(concern, "cross_cutting")
            user_stories.append(concern_story)
            
        return user_stories
    
    def _create_user_story(self, functionality: Dict[str, Any], story_type: str) -> UserStory:
        """Create individual user story with acceptance criteria"""
        
        story_prompt = f"""
        Create a user story for this functionality:
        
        Functionality: {functionality}
        Story Type: {story_type}
        
        Follow format:
        - Title: Clear, concise title
        - Description: "As a [user], I want [goal] so that [benefit]"
        - Acceptance Criteria: Specific, testable criteria (3-5 items)
        - Priority: High/Medium/Low based on business impact
        - Estimated Effort: XS/S/M/L/XL based on complexity
        - Dependencies: Other stories this depends on
        - Business Value: Why this matters to the business
        
        Use best practices for user story writing.
        """
        
        result = self._execute_with_reasoning(story_prompt)
        return self._parse_user_story_response(result)
    
    def _analyze_competitors(self, parsed_requirement: Dict[str, Any]) -> Dict[str, Any]:
        """Perform competitive analysis for the requirement"""
        
        analysis_prompt = f"""
        Perform competitive analysis for this requirement:
        
        Requirement: {parsed_requirement}
        
        Research and analyze:
        1. Existing solutions in the market
        2. How competitors handle similar functionality
        3. Strengths and weaknesses of existing approaches
        4. Differentiation opportunities
        5. Pricing and business model insights
        6. User feedback and pain points from competitor solutions
        
        Provide actionable insights for product strategy.
        """
        
        # Execute with web search tool for real-time competitive data
        result = self._execute_with_tools(analysis_prompt, ["web_search", "competitor_analysis"])
        return json.loads(result)
    
    def _conduct_market_research(self, parsed_requirement: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct market research and trend analysis"""
        
        research_prompt = f"""
        Conduct market research for this requirement:
        
        Requirement: {parsed_requirement}
        
        Research:
        1. Market size and growth trends
        2. Target audience demographics and behaviors
        3. Current market gaps and opportunities
        4. Technology adoption trends
        5. Regulatory considerations
        6. Economic factors affecting demand
        
        Provide data-driven insights with sources.
        """
        
        result = self._execute_with_tools(research_prompt, ["web_search", "market_research"])
        return json.loads(result)
    
    def _define_success_metrics(self, parsed_requirement: Dict[str, Any]) -> List[str]:
        """Define measurable success metrics"""
        
        metrics_prompt = f"""
        Define success metrics for this requirement:
        
        Requirement: {parsed_requirement}
        
        Create metrics for:
        1. User adoption (how many users will use this?)
        2. User engagement (how often will they use it?)
        3. Business impact (revenue, cost savings, efficiency gains)
        4. Technical performance (speed, reliability, scalability)
        5. Quality measures (user satisfaction, error rates)
        
        Make metrics SMART (Specific, Measurable, Achievable, Relevant, Time-bound).
        """
        
        result = self._execute_with_reasoning(metrics_prompt)
        return json.loads(result)
    
    def _assess_risks(self, parsed_requirement: Dict[str, Any], user_stories: List[UserStory]) -> Dict[str, Any]:
        """Assess project risks and mitigation strategies"""
        
        risk_prompt = f"""
        Assess risks for this project:
        
        Requirement: {parsed_requirement}
        User Stories: {[story.title for story in user_stories]}
        
        Identify risks in categories:
        1. Technical risks (complexity, dependencies, unknowns)
        2. Business risks (market changes, competition, resources)
        3. User adoption risks (usability, value proposition)
        4. Timeline risks (scope creep, dependencies, blockers)
        5. Resource risks (team capacity, skills, budget)
        
        For each risk, provide:
        - Impact level (High/Medium/Low)
        - Probability (High/Medium/Low)
        - Mitigation strategies
        - Contingency plans
        """
        
        result = self._execute_with_reasoning(risk_prompt)
        return json.loads(result)
    
    def _create_roadmap(self, user_stories: List[UserStory]) -> Dict[str, Any]:
        """Create product roadmap from user stories"""
        
        roadmap_prompt = f"""
        Create a product roadmap from these user stories:
        
        Stories: {[{"title": story.title, "priority": story.priority, "effort": story.estimated_effort} for story in user_stories]}
        
        Create roadmap with:
        1. Phase organization (MVP, V1, V2, etc.)
        2. Timeline estimates
        3. Dependencies and sequencing
        4. Resource requirements
        5. Key milestones and deliverables
        6. Risk mitigation points
        
        Balance quick wins with long-term value.
        """
        
        result = self._execute_with_reasoning(roadmap_prompt)
        return json.loads(result)
    
    def request_handoff(self, target: AgentRole, analysis: RequirementAnalysis) -> HandoffRequest:
        """Request handoff to architect or QA with PM analysis"""
        
        handoff_context = {
            "product_analysis": analysis.__dict__,
            "key_requirements": [story.description for story in analysis.user_stories],
            "success_metrics": analysis.success_metrics,
            "risk_factors": analysis.risk_assessment,
            "business_priorities": [story for story in analysis.user_stories if story.priority == "High"]
        }
        
        if target == AgentRole.ARCHITECT:
            task_description = "Create technical architecture and system design based on product requirements"
        elif target == AgentRole.QA_ENGINEER:
            task_description = "Create comprehensive test strategy and validation plans"
        else:
            task_description = f"Continue workflow with product analysis results"
            
        return HandoffRequest(
            target_agent=target,
            context=handoff_context,
            task_description=task_description,
            priority=1
        )

# Agent-specific tools and utilities

def load_user_story_template() -> str:
    """Load user story template for consistent formatting"""
    return """
    Title: [Clear, action-oriented title]
    
    As a [type of user]
    I want [some goal or objective]
    So that [some reason or benefit]
    
    Acceptance Criteria:
    - [ ] [Specific, testable criterion 1]
    - [ ] [Specific, testable criterion 2]
    - [ ] [Specific, testable criterion 3]
    
    Priority: [High/Medium/Low]
    Estimated Effort: [XS/S/M/L/XL]
    Dependencies: [List of dependent stories]
    Business Value: [Clear value statement]
    """

def load_analysis_framework() -> Dict[str, Any]:
    """Load PM analysis framework for consistent approach"""
    return {
        "requirement_parsing": {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "business_requirements": [],
            "user_requirements": [],
            "system_requirements": []
        },
        "stakeholder_analysis": {
            "primary_users": [],
            "secondary_users": [],
            "business_stakeholders": [],
            "technical_stakeholders": []
        },
        "value_proposition": {
            "user_benefits": [],
            "business_benefits": [],
            "competitive_advantages": []
        }
    }
```

## Integration with Orchestrator

The Product Manager Agent integrates seamlessly with the multi-agent orchestrator:

1. **Input Processing**: Receives raw requirements from users or orchestrator
2. **Analysis Execution**: Performs comprehensive product analysis using specialized tools
3. **Output Generation**: Produces structured requirements and user stories
4. **Handoff Coordination**: Passes results to Architect Agent for technical design

## Example Output

```json
{
  "user_stories": [
    {
      "id": "US001",
      "title": "User Authentication System",
      "description": "As a user, I want to securely log into the application so that my data remains protected",
      "acceptance_criteria": [
        "User can log in with email and password",
        "Password must meet security requirements",
        "Failed login attempts are limited and logged",
        "User receives confirmation of successful login"
      ],
      "priority": "High",
      "estimated_effort": "M",
      "dependencies": [],
      "business_value": "Essential for user data security and regulatory compliance"
    }
  ],
  "competitive_analysis": {
    "existing_solutions": ["Auth0", "Firebase Auth", "AWS Cognito"],
    "differentiation_opportunities": ["Simplified onboarding", "Better mobile experience"],
    "market_positioning": "Developer-friendly authentication with advanced security"
  },
  "success_metrics": [
    "95% successful login rate within 30 seconds",
    "Less than 0.1% security incidents",
    "User satisfaction score > 4.5/5"
  ]
}
```

This Product Manager Agent provides the foundation for systematic requirement analysis and ensures all subsequent development work is aligned with business objectives and user needs.

---
*Status: Product Manager Agent Implementation - Ready for Integration*
*Next: Architect Agent Implementation*