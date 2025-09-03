/**
 * SISO Agent Dashboard - Shared Type Definitions
 * Following the principle: "Architecture + Types + Tests = AI Cannot Fail"
 * 
 * These types serve as guardrails preventing AI hallucination during parallel development
 */

// Agent Task Management Types
export interface AgentTask {
  id: string;
  type: 'frontend' | 'backend' | 'agent-core' | 'voice-interface' | 'integration' | 'testing';
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'blocked';
  dependencies: string[];
  estimatedHours: number;
  workspace: string;
  assignedAgent?: string;
  createdAt: Date;
  updatedAt: Date;
}

// Agent Progress Tracking
export interface AgentProgress {
  agentId: string;
  taskId: string;
  completionPercentage: number;
  currentAction: string;
  filesModified: string[];
  testsStatus: 'passing' | 'failing' | 'not_run' | 'pending';
  lastUpdate: Date;
}

// Agent Coordination Events
export interface CoordinationEvent {
  type: 'task_started' | 'task_completed' | 'task_failed' | 'dependency_resolved' | 
        'merge_conflict' | 'test_failed' | 'agent_blocked' | 'context_revival';
  agentId: string;
  taskId?: string;
  payload: any;
  timestamp: Date;
  severity: 'info' | 'warning' | 'error' | 'critical';
}

// Agent Specialized Roles
export interface AgentSpecialization {
  agentId: string;
  role: 'frontend' | 'backend' | 'ai-architect' | 'integration' | 'voice' | 'qa' | 'mobile';
  expertise: string[];
  capabilities: string[];
  preferredTools: string[];
  contextPreservation: boolean;
}

// Development Environment Configuration
export interface WorkspaceConfig {
  workspaceId: string;
  gitBranch: string;
  worktreePath: string;
  agentId: string;
  isolated: boolean;
  dependencies: string[];
  mockInterfaces: boolean;
}

// Component Interface Definitions (for parallel development)
export interface ComponentInterface {
  componentName: string;
  exports: string[];
  imports: string[];
  apiContract?: APIContract;
  mockData?: any;
}

export interface APIContract {
  endpoints: EndpointDefinition[];
  schemas: Record<string, any>;
  authentication?: AuthConfig;
}

export interface EndpointDefinition {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  requestSchema?: any;
  responseSchema?: any;
  description: string;
}

export interface AuthConfig {
  type: 'none' | 'api-key' | 'bearer' | 'oauth';
  required: boolean;
}

// Context Revival Types (Zen MCP integration)
export interface ContextState {
  sessionId: string;
  projectGoals: string[];
  architecturalDecisions: string[];
  completedTasks: string[];
  currentFocus: string;
  agentStates: Record<string, AgentState>;
  lastRevival: Date;
}

export interface AgentState {
  agentId: string;
  currentTask?: string;
  understanding: string[];
  blockers: string[];
  nextSteps: string[];
  contextQuality: number; // 0-1 score
}

// Performance Metrics
export interface PerformanceMetrics {
  developmentTimeReduction: number; // Target: 76%
  parallelAgentCount: number;
  contextPreservationRate: number; // Target: >90%
  mergeConflictRate: number; // Target: <5%
  testCoveragePercentage: number;
  bugRate: number;
}

// SANDBOX Method Configuration
export interface SANDBOXConfig {
  toolType: 'conductor-ui' | 'code-conductor' | 'manual';
  maxParallelAgents: number;
  gitWorktreeEnabled: boolean;
  visualDashboard: boolean;
  conflictResolution: 'automatic' | 'manual' | 'intelligent';
  progressMonitoring: boolean;
}

// Voice Interface Types
export interface VoiceCommand {
  command: string;
  parameters: Record<string, any>;
  confidence: number;
  timestamp: Date;
  agentTarget?: string;
}

export interface VoiceResponse {
  message: string;
  actions: string[];
  success: boolean;
  audioOutput?: boolean;
}

// Error Handling and Recovery
export interface AgentError {
  errorId: string;
  agentId: string;
  taskId?: string;
  type: 'compilation' | 'runtime' | 'integration' | 'context-loss' | 'merge-conflict';
  message: string;
  stack?: string;
  recovery: 'retry' | 'rollback' | 'manual' | 'context-revival';
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// Integration Testing Types
export interface IntegrationTest {
  testId: string;
  components: string[];
  scenario: string;
  expectedResult: any;
  actualResult?: any;
  status: 'pending' | 'running' | 'passed' | 'failed';
  executionTime: number;
}

// Export all types for use across components
export type {
  AgentTask,
  AgentProgress, 
  CoordinationEvent,
  AgentSpecialization,
  WorkspaceConfig,
  ComponentInterface,
  APIContract,
  ContextState,
  AgentState,
  PerformanceMetrics,
  SANDBOXConfig,
  VoiceCommand,
  VoiceResponse,
  AgentError,
  IntegrationTest
};