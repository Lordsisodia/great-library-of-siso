# UI Patterns - AI-Optimized Interface Components

**Source**: [App Building Framework Insights](../../insights/README.md)

## AI Interface Design Principles

### Real-Time Feedback Patterns
**Core Principle**: AI operations are async - users need constant feedback
**Pattern**: Progressive disclosure of AI processing status

```jsx
// ProcessingIndicator.jsx - Universal AI status component
const ProcessingIndicator = ({ stage, progress, details }) => (
  <div className="ai-processing-indicator">
    <div className="stage-progress">
      <div className="stage-name">{stage}</div>
      <div className="progress-bar" style={{width: `${progress}%`}} />
    </div>
    <div className="processing-details">
      {details && <span className="details-text">{details}</span>}
      <div className="ai-thinking-animation">🤖 Processing...</div>
    </div>
  </div>
);

// Usage example from AI video processing
<ProcessingIndicator 
  stage="Generating AI Voice"
  progress={65}
  details="Using 11Labs API with voice ID: christopher"
/>
```

### Context Display Patterns
**Core Principle**: Show users what AI "remembers" and is working with
**Pattern**: Contextual sidebar with AI's current understanding

```jsx
// AIContextSidebar.jsx
const AIContextSidebar = ({ context, isProcessing }) => (
  <div className="ai-context-sidebar">
    <h3>AI Context</h3>
    <div className="context-items">
      {context.userInput && (
        <div className="context-item">
          <label>User Request:</label>
          <div className="context-value">{context.userInput}</div>
        </div>
      )}
      {context.filesUploaded && (
        <div className="context-item">
          <label>Files:</label>
          <div className="context-value">
            {context.filesUploaded.map(file => (
              <span key={file.name} className="file-tag">{file.name}</span>
            ))}
          </div>
        </div>
      )}
      {context.previousActions && (
        <div className="context-item">
          <label>Previous Actions:</label>
          <div className="context-value">
            {context.previousActions.map((action, i) => (
              <div key={i} className="action-item">
                ✅ {action}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
    {isProcessing && (
      <div className="ai-thinking">
        <div className="thinking-animation">🧠 AI is thinking...</div>
        <div className="thinking-details">
          Analyzing your request and context
        </div>
      </div>
    )}
  </div>
);
```

## Core UI Component Patterns

### 1. AI Chat Interface Pattern
**Use Case**: Conversational AI interactions
**Features**: Message history, typing indicators, context awareness

```jsx
// AIChatInterface.jsx
const AIChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [context, setContext] = useState({});

  return (
    <div className="ai-chat-container">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <MessageBubble 
            key={i} 
            message={msg} 
            isAI={msg.sender === 'ai'}
            context={msg.context}
          />
        ))}
        {isTyping && <TypingIndicator />}
      </div>
      <ChatInput onSend={handleSendMessage} />
      <AIContextSidebar context={context} isProcessing={isTyping} />
    </div>
  );
};
```

### 2. File Upload with AI Processing
**Use Case**: Upload files for AI analysis/processing
**Features**: Drag/drop, progress tracking, AI processing status

```jsx
// AIFileProcessor.jsx
const AIFileProcessor = ({ onProcessingComplete }) => {
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [processingStage, setProcessingStage] = useState(null);
  
  return (
    <div className="ai-file-processor">
      <FileDropZone 
        onDrop={handleFileUpload}
        disabled={uploadStatus === 'processing'}
      />
      
      {uploadStatus === 'uploading' && (
        <UploadProgress progress={uploadProgress} />
      )}
      
      {uploadStatus === 'processing' && (
        <AIProcessingStatus 
          stage={processingStage}
          onStageChange={setProcessingStage}
        />
      )}
      
      {uploadStatus === 'complete' && (
        <ProcessingResults results={processingResults} />
      )}
    </div>
  );
};
```

### 3. Real-Time AI Dashboard
**Use Case**: Monitor AI operations across multiple processes
**Features**: Live updates, status grid, error handling

```jsx
// AIDashboard.jsx
const AIDashboard = () => {
  const [aiProcesses, setAIProcesses] = useState([]);
  const [systemStatus, setSystemStatus] = useState('healthy');

  useEffect(() => {
    // Real-time updates from Firebase
    const unsubscribe = onSnapshot(
      collection(db, 'aiProcesses'),
      (snapshot) => {
        const processes = snapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data()
        }));
        setAIProcesses(processes);
      }
    );
    return unsubscribe;
  }, []);

  return (
    <div className="ai-dashboard">
      <SystemStatusHeader status={systemStatus} />
      <div className="process-grid">
        {aiProcesses.map(process => (
          <ProcessCard 
            key={process.id} 
            process={process}
            onRetry={handleProcessRetry}
          />
        ))}
      </div>
      <AIResourceMonitor />
    </div>
  );
};
```

## Specialized AI UI Components

### Prompt Engineering Interface
**Use Case**: Allow users to customize AI behavior
**Features**: Prompt templates, parameter adjustment, testing

```jsx
// PromptBuilder.jsx
const PromptBuilder = ({ onPromptChange }) => {
  const [systemPrompt, setSystemPrompt] = useState('');
  const [parameters, setParameters] = useState({
    temperature: 0.7,
    maxTokens: 1000,
    model: 'gpt-4'
  });

  return (
    <div className="prompt-builder">
      <div className="prompt-section">
        <label>System Prompt</label>
        <PromptEditor 
          value={systemPrompt}
          onChange={setSystemPrompt}
          templates={promptTemplates}
        />
      </div>
      
      <div className="parameters-section">
        <ParameterSlider 
          label="Temperature"
          value={parameters.temperature}
          min={0} max={1} step={0.1}
          onChange={(val) => setParameters({...parameters, temperature: val})}
        />
        <ModelSelector 
          value={parameters.model}
          onChange={(model) => setParameters({...parameters, model})}
        />
      </div>
      
      <div className="testing-section">
        <TestPromptButton 
          prompt={systemPrompt}
          parameters={parameters}
          onResult={handleTestResult}
        />
      </div>
    </div>
  );
};
```

### AI Error Recovery Interface
**Use Case**: Handle AI failures gracefully with user options
**Features**: Error explanation, retry options, fallback suggestions

```jsx
// AIErrorHandler.jsx
const AIErrorHandler = ({ error, onRetry, onFallback }) => {
  const getErrorSuggestion = (error) => {
    if (error.type === 'rate_limit') {
      return "AI service is busy. Try again in a few minutes.";
    }
    if (error.type === 'context_limit') {
      return "Request too long. Try breaking it into smaller parts.";
    }
    if (error.type === 'hallucination_detected') {
      return "AI gave unreliable response. Using fallback method.";
    }
    return "Unexpected AI error. Trying backup approach.";
  };

  return (
    <div className="ai-error-handler">
      <div className="error-icon">⚠️</div>
      <div className="error-message">
        <h4>AI Processing Issue</h4>
        <p>{getErrorSuggestion(error)}</p>
      </div>
      
      <div className="error-actions">
        <button onClick={onRetry} className="retry-button">
          🔄 Try Again
        </button>
        
        {error.fallbackAvailable && (
          <button onClick={onFallback} className="fallback-button">
            🛠️ Use Alternative Method
          </button>
        )}
        
        <button onClick={() => reportError(error)} className="report-button">
          📝 Report Issue
        </button>
      </div>
    </div>
  );
};
```

## Layout Patterns for AI Applications

### Split-Screen AI Workspace
**Use Case**: Show input/output side by side for AI processing
**Best for**: Content generation, code assistance, data transformation

```jsx
// SplitScreenWorkspace.jsx
const SplitScreenWorkspace = () => (
  <div className="split-workspace">
    <div className="input-panel">
      <h3>Input</h3>
      <InputArea />
      <AIConfigPanel />
    </div>
    
    <div className="processing-divider">
      <AIProcessingIndicator />
    </div>
    
    <div className="output-panel">
      <h3>AI Output</h3>
      <OutputArea />
      <OutputActions />
    </div>
  </div>
);
```

### Tabbed AI Operations
**Use Case**: Multiple AI processes running simultaneously
**Best for**: Multi-agent systems, different AI models

```jsx
// TabbedAIOperations.jsx
const TabbedAIOperations = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [operations, setOperations] = useState([]);

  return (
    <div className="tabbed-ai-operations">
      <div className="operation-tabs">
        {operations.map((op, i) => (
          <Tab 
            key={op.id}
            isActive={i === activeTab}
            onClick={() => setActiveTab(i)}
            status={op.status}
          >
            {op.name}
            <StatusIndicator status={op.status} />
          </Tab>
        ))}
        <NewOperationButton onClick={createNewOperation} />
      </div>
      
      <div className="operation-content">
        {operations[activeTab] && (
          <AIOperationPanel operation={operations[activeTab]} />
        )}
      </div>
    </div>
  );
};
```

## Responsive AI Interface Patterns

### Mobile AI Chat
**Adaptations for mobile**:
- Larger touch targets
- Swipe gestures for context
- Voice input integration
- Condensed status displays

```jsx
// MobileAIChat.jsx
const MobileAIChat = () => (
  <div className="mobile-ai-chat">
    <div className="chat-header">
      <AIStatusDot status={aiStatus} />
      <h2>AI Assistant</h2>
      <SettingsButton />
    </div>
    
    <div className="chat-messages touch-scrollable">
      {/* Messages with swipe-to-show-context */}
    </div>
    
    <div className="mobile-input-area">
      <VoiceInputButton />
      <TextInput />
      <SendButton />
    </div>
  </div>
);
```

### Desktop AI Workspace
**Features for desktop**:
- Multiple panels
- Keyboard shortcuts
- Advanced context display
- Multi-monitor support

```jsx
// DesktopAIWorkspace.jsx
const DesktopAIWorkspace = () => (
  <div className="desktop-ai-workspace">
    <Sidebar>
      <ProjectNavigation />
      <AIAgentsList />
      <ContextBrowser />
    </Sidebar>
    
    <MainWorkArea>
      <EditorPanel />
      <AIAssistantPanel />
    </MainWorkArea>
    
    <RightPanel>
      <OutputPreview />
      <ProcessingQueue />
    </RightPanel>
  </div>
);
```

## Animation Patterns for AI

### AI Thinking Animation
```css
.ai-thinking {
  display: flex;
  align-items: center;
  gap: 8px;
}

.thinking-dots {
  display: flex;
  gap: 4px;
}

.thinking-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #007bff;
  animation: thinking-pulse 1.4s infinite ease-in-out;
}

.thinking-dot:nth-child(1) { animation-delay: 0s; }
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes thinking-pulse {
  0%, 80%, 100% { transform: scale(1); opacity: 0.3; }
  40% { transform: scale(1.2); opacity: 1; }
}
```

### Progress Visualization
```css
.ai-progress-ring {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: conic-gradient(
    #007bff 0deg,
    #007bff var(--progress-angle),
    #e9ecef var(--progress-angle),
    #e9ecef 360deg
  );
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.ai-progress-inner {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
```

## Data Visualization for AI

### AI Performance Metrics Display
```jsx
// AIMetricsDisplay.jsx
const AIMetricsDisplay = ({ metrics }) => (
  <div className="ai-metrics">
    <div className="metric-card">
      <div className="metric-icon">⚡</div>
      <div className="metric-value">{metrics.responseTime}ms</div>
      <div className="metric-label">Response Time</div>
    </div>
    
    <div className="metric-card">
      <div className="metric-icon">🎯</div>
      <div className="metric-value">{metrics.accuracy}%</div>
      <div className="metric-label">Accuracy</div>
    </div>
    
    <div className="metric-card">
      <div className="metric-icon">💰</div>
      <div className="metric-value">${metrics.cost}</div>
      <div className="metric-label">API Cost</div>
    </div>
  </div>
);
```

### AI Process Flow Visualization
```jsx
// AIFlowChart.jsx
const AIFlowChart = ({ processes }) => (
  <div className="ai-flow-chart">
    {processes.map((process, i) => (
      <div key={process.id} className="flow-step">
        <div className={`step-node ${process.status}`}>
          <div className="step-icon">{process.icon}</div>
          <div className="step-name">{process.name}</div>
        </div>
        {i < processes.length - 1 && (
          <div className="flow-arrow">→</div>
        )}
      </div>
    ))}
  </div>
);
```

## Accessibility for AI Interfaces

### Screen Reader Support
```jsx
// Accessible AI status announcements
const AIStatusAnnouncer = ({ status, stage }) => (
  <div 
    role="status" 
    aria-live="polite" 
    className="sr-only"
  >
    AI is currently {status}. {stage && `Stage: ${stage}`}
  </div>
);
```

### Keyboard Navigation
```jsx
// Keyboard shortcuts for AI operations
const useAIKeyboardShortcuts = () => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'Enter':
            e.preventDefault();
            triggerAIGeneration();
            break;
          case 'r':
            e.preventDefault();
            retryLastAIOperation();
            break;
          case 'Escape':
            e.preventDefault();
            cancelAIOperation();
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
};
```

This UI pattern library ensures that AI applications provide excellent user experience with clear feedback, intuitive interactions, and robust error handling across all interface elements.