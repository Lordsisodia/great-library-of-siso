# Development Tools - AI-Enhanced Development Environment

**Source**: [App Building Framework Insights](../../insights/README.md) | [Multi-Agent Coordination](../../insights/tools-and-configs/multi-agent-coordination.md)

## Core Development Environment

### agents.md File Configuration
**Purpose**: Tool-independent AI rules and behavior patterns
**Location**: Project root (`/agents.md`)
**Use Case**: Works with Cursor, Claude Code, and any AI coding assistant

```markdown
# AI Agent Rules and Patterns

## Project Context
- **Framework**: Next.js + Firebase + AI APIs
- **Architecture**: Event-driven with real-time updates
- **Testing**: Real data integration tests (NO MOCKS)
- **Deployment**: One-command Firebase deploy

## AI Coding Rules
1. **Test-First Development**: Write integration tests BEFORE implementation
2. **Real Data Only**: Never use mock data - always test with real APIs
3. **Error Verification**: AI must run tests and show actual results
4. **Context Preservation**: Update ADR.md with all decisions
5. **Security First**: Implement proper Firestore/Storage rules

## Coding Standards
- TypeScript for all new code
- Functional components with hooks
- Proper error boundaries
- Real-time state management with Firebase
- Comprehensive logging for debugging

## AI Prompting Patterns
- Use Information Dense Keywords: CREATE, UPDATE, DELETE, ADD, REMOVE
- Always specify: "Test with real data and show me the generated files"
- Context sharing: "Reference the architecture decisions in ADR.md"
- Multi-agent coordination: "This depends on the API integration from Agent 1"

## Prohibited Patterns
- ❌ Mock data in tests during development
- ❌ Hardcoded API responses
- ❌ Claiming tests pass without showing outputs
- ❌ Ignoring existing architecture patterns
- ❌ Creating new files without checking existing structure
```

### Architecture Decision Records (ADR.md)
**Purpose**: Preserve context and decisions for AI agents
**Location**: Project root (`/ADR.md`)
**Updates**: After every major decision or completed task

```markdown
# Architecture Decision Record

## Project: [Project Name]
**Started**: [Date]
**Last Updated**: [Date]

## Tech Stack Decisions

### Frontend
- **Framework**: Next.js 13+ with App Router
- **Styling**: Tailwind CSS + Headless UI
- **State**: React hooks + Firebase real-time
- **Authentication**: Firebase Auth
- **Reasoning**: Rapid prototyping, real-time updates, one-command deploy

### Backend
- **Platform**: Firebase Functions
- **Database**: Firestore
- **Storage**: Firebase Storage
- **AI APIs**: OpenAI GPT-4, 11Labs Voice
- **Reasoning**: Serverless scaling, real-time triggers, cost-effective

## Architecture Patterns

### Real-Time Updates
```typescript
// Pattern: Firestore real-time subscriptions
const useRealtimeData = (collection: string) => {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    const unsubscribe = onSnapshot(
      collection(db, collection),
      (snapshot) => setData(snapshot.docs.map(doc => ({id: doc.id, ...doc.data()})))
    );
    return unsubscribe;
  }, [collection]);
  
  return data;
};
```

### AI Integration Pattern
```typescript
// Pattern: AI service with error handling
class AIService {
  async processWithFallback(input: any) {
    try {
      return await this.primaryAI.process(input);
    } catch (error) {
      if (error.type === 'rate_limit') {
        return await this.fallbackAI.process(input);
      }
      throw error;
    }
  }
}
```

## Agent Coordination Log
- **Agent 1**: API integrations (11Labs, OpenAI) ✅ COMPLETED
- **Agent 2**: Video processing (FFmpeg) ✅ COMPLETED  
- **Agent 3**: Security rules (Firestore/Storage) ✅ COMPLETED
- **Agent 4**: Frontend UI (React components) 🔄 IN PROGRESS
- **Agent 5**: Background processing (Cloud Functions) ⏳ WAITING

## Testing Decisions
- **Strategy**: Integration tests with real data during development
- **Tools**: Jest + Firebase emulators + real API keys
- **Verification**: All tests must generate actual output files
- **Migration**: Convert to unit tests after features are complete

## Security Decisions
- **Authentication**: Email/password + Google OAuth
- **Authorization**: Firestore rules per user scope
- **File Access**: Storage rules with user-based paths
- **API Keys**: Environment variables with Firebase Functions

## Performance Decisions
- **Caching**: Firestore offline persistence enabled
- **Images**: Next.js Image optimization
- **Code Splitting**: Dynamic imports for heavy components
- **Monitoring**: Firebase Performance + custom metrics
```

## IDE Configuration Files

### VS Code Settings (AI-Optimized)
**File**: `.vscode/settings.json`
```json
{
  "typescript.preferences.includePackageJsonAutoImports": "off",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true,
    "source.organizeImports": true
  },
  "editor.formatOnSave": true,
  "editor.rulers": [80, 120],
  "files.associations": {
    "*.md": "markdown"
  },
  "ai.completion.enabled": true,
  "ai.context.includeFiles": [
    "ADR.md",
    "agents.md",
    "README.md",
    "firebase.json"
  ]
}
```

### Cursor Configuration
**File**: `.cursor/cursor-rules`
```
# Cursor AI Rules for this project

## Context Files (Always Reference)
- ADR.md - Architecture decisions and agent coordination
- agents.md - AI behavior rules and coding standards  
- firestore.rules - Database security implementation
- firebase.json - Deployment configuration

## Code Generation Rules
1. Always use TypeScript
2. Implement real-time Firebase subscriptions
3. Add proper error boundaries
4. Include loading states for async operations
5. Write integration tests with real data

## Testing Requirements
- Use real Firebase project for testing
- Test with actual API keys (not mocks)
- Generate actual output files
- Show test results with file paths

## Multi-Agent Awareness
- Check git status before major changes
- Update ADR.md with decisions
- Reference completed agent work
- Avoid file conflicts with parallel agents
```

## Git Workflow Configuration

### .gitignore (AI Project Optimized)
```gitignore
# Dependencies
node_modules/
.pnp
.pnp.js

# Production builds
.next/
out/
dist/
build/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Firebase
.firebase/
firebase-debug.log
firestore-debug.log
ui-debug.log
functions/node_modules/
functions/.env

# AI-specific ignores
service-account-key.json
.openai-key
.anthropic-key
.elevenlabs-key

# Generated content
generated/
ai-output/
test-output/
*.generated.*

# IDE
.vscode/launch.json
.cursor/local-settings
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Testing
coverage/
.nyc_output
test-results/
```

### Pre-commit Hooks
**File**: `.husky/pre-commit`
```bash
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

# Run tests with real data
npm run test:integration

# Type checking
npm run type-check

# Lint and format
npm run lint:fix

# Security rules validation
firebase deploy --only firestore:rules --project=development --dry-run
```

## Development Scripts

### package.json Scripts (AI-Enhanced)
```json
{
  "scripts": {
    "dev": "concurrently \"npm run dev:frontend\" \"npm run dev:functions\" \"npm run dev:emulators\"",
    "dev:frontend": "cd frontend && npm run dev",
    "dev:functions": "cd functions && npm run serve",
    "dev:emulators": "firebase emulators:start",
    
    "test": "npm run test:unit && npm run test:integration",
    "test:unit": "jest",
    "test:integration": "jest --config=jest.integration.config.js",
    "test:ai": "node scripts/test-ai-integration.js",
    
    "deploy": "npm run deploy:rules && npm run deploy:functions && npm run deploy:hosting",
    "deploy:rules": "firebase deploy --only firestore:rules,storage:rules",
    "deploy:functions": "firebase deploy --only functions",
    "deploy:hosting": "firebase deploy --only hosting",
    "deploy:all": "firebase deploy",
    
    "ai:agents": "node scripts/manage-agents.js",
    "ai:context": "node scripts/build-ai-context.js",
    "ai:validate": "node scripts/validate-ai-outputs.js",
    
    "setup": "npm run setup:dependencies && npm run setup:firebase && npm run setup:env",
    "setup:dependencies": "npm install && cd frontend && npm install && cd ../functions && npm install",
    "setup:firebase": "firebase init --project=development",
    "setup:env": "node scripts/setup-environment.js"
  }
}
```

### AI Context Builder Script
**File**: `scripts/build-ai-context.js`
```javascript
// Builds context for AI agents from project state
const fs = require('fs');
const path = require('path');

const buildAIContext = () => {
  const context = {
    timestamp: new Date().toISOString(),
    architecture: fs.readFileSync('ADR.md', 'utf8'),
    agentRules: fs.readFileSync('agents.md', 'utf8'),
    firebaseConfig: JSON.parse(fs.readFileSync('firebase.json', 'utf8')),
    packageInfo: JSON.parse(fs.readFileSync('package.json', 'utf8')),
    gitStatus: require('child_process').execSync('git status --porcelain').toString(),
    recentCommits: require('child_process')
      .execSync('git log --oneline -10')
      .toString()
      .split('\n')
      .filter(Boolean)
  };

  fs.writeFileSync(
    '.ai-context.json', 
    JSON.stringify(context, null, 2)
  );
  
  console.log('AI context updated:', {
    files: Object.keys(context).length,
    lastUpdate: context.timestamp
  });
};

buildAIContext();
```

### Multi-Agent Management Script
**File**: `scripts/manage-agents.js`
```javascript
// Multi-agent coordination and monitoring
const fs = require('fs');
const path = require('path');

class AgentManager {
  constructor() {
    this.agentsStatus = this.loadAgentStatus();
  }

  loadAgentStatus() {
    try {
      return JSON.parse(fs.readFileSync('.agents-status.json', 'utf8'));
    } catch {
      return {};
    }
  }

  saveAgentStatus() {
    fs.writeFileSync('.agents-status.json', JSON.stringify(this.agentsStatus, null, 2));
  }

  startAgent(agentId, task, dependencies = []) {
    // Check if dependencies are complete
    const incompleteDeps = dependencies.filter(dep => 
      !this.agentsStatus[dep] || this.agentsStatus[dep].status !== 'completed'
    );

    if (incompleteDeps.length > 0) {
      console.log(`❌ Agent ${agentId} blocked by dependencies: ${incompleteDeps.join(', ')}`);
      return false;
    }

    this.agentsStatus[agentId] = {
      status: 'in_progress',
      task,
      startTime: new Date().toISOString(),
      dependencies: dependencies
    };
    
    this.saveAgentStatus();
    console.log(`🤖 Agent ${agentId} started: ${task}`);
    return true;
  }

  completeAgent(agentId, results = {}) {
    if (!this.agentsStatus[agentId]) {
      console.log(`❌ Agent ${agentId} not found`);
      return false;
    }

    this.agentsStatus[agentId] = {
      ...this.agentsStatus[agentId],
      status: 'completed',
      endTime: new Date().toISOString(),
      results
    };

    this.saveAgentStatus();
    console.log(`✅ Agent ${agentId} completed`);
    
    // Check what agents can now start
    this.checkPendingAgents();
    return true;
  }

  checkPendingAgents() {
    const pending = Object.entries(this.agentsStatus)
      .filter(([_, status]) => status.status === 'pending');
    
    pending.forEach(([agentId, agentInfo]) => {
      const canStart = agentInfo.dependencies.every(dep => 
        this.agentsStatus[dep]?.status === 'completed'
      );
      
      if (canStart) {
        console.log(`🔄 Agent ${agentId} can now start (dependencies complete)`);
      }
    });
  }

  getStatus() {
    return this.agentsStatus;
  }
}

// CLI interface
const command = process.argv[2];
const agentManager = new AgentManager();

switch (command) {
  case 'start':
    const [agentId, task] = process.argv.slice(3);
    agentManager.startAgent(agentId, task);
    break;
    
  case 'complete':
    const completeId = process.argv[3];
    agentManager.completeAgent(completeId);
    break;
    
  case 'status':
    console.log(JSON.stringify(agentManager.getStatus(), null, 2));
    break;
    
  default:
    console.log('Usage: npm run ai:agents [start|complete|status] [agentId] [task]');
}
```

## Environment Configuration

### Development Environment Setup
**File**: `scripts/setup-environment.js`
```javascript
const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const setupEnvironment = async () => {
  console.log('🚀 Setting up AI development environment...\n');

  const config = {};
  
  // Firebase configuration
  config.NEXT_PUBLIC_FIREBASE_PROJECT_ID = await ask('Firebase Project ID: ');
  config.NEXT_PUBLIC_FIREBASE_API_KEY = await ask('Firebase API Key: ');
  
  // AI API keys
  config.OPENAI_API_KEY = await ask('OpenAI API Key: ');
  config.ELEVENLABS_API_KEY = await ask('11Labs API Key (optional): ');
  config.ANTHROPIC_API_KEY = await ask('Anthropic API Key (optional): ');

  // Generate .env files
  const envContent = Object.entries(config)
    .map(([key, value]) => `${key}=${value}`)
    .join('\n');
  
  fs.writeFileSync('.env', envContent);
  fs.writeFileSync('frontend/.env.local', envContent);
  fs.writeFileSync('functions/.env', envContent);

  console.log('\n✅ Environment configuration complete!');
  console.log('📁 Created: .env, frontend/.env.local, functions/.env');
  
  rl.close();
};

const ask = (question) => {
  return new Promise(resolve => {
    rl.question(question, resolve);
  });
};

setupEnvironment().catch(console.error);
```

## Testing Configuration

### AI Integration Testing
**File**: `tests/ai-integration.test.js`
```javascript
// Tests AI integrations with real APIs
const { Configuration, OpenAIApi } = require('openai');
const ElevenLabs = require('elevenlabs-node');

describe('AI Integrations', () => {
  let openai, elevenlabs;
  
  beforeAll(() => {
    openai = new OpenAIApi(new Configuration({
      apiKey: process.env.OPENAI_API_KEY
    }));
    
    elevenlabs = new ElevenLabs({
      apiKey: process.env.ELEVENLABS_API_KEY
    });
  });

  test('OpenAI text generation', async () => {
    const response = await openai.createCompletion({
      model: "text-davinci-003",
      prompt: "Generate a test greeting for John",
      max_tokens: 50
    });
    
    expect(response.data.choices[0].text).toBeTruthy();
    expect(response.data.choices[0].text.toLowerCase()).toContain('john');
    
    console.log('✅ OpenAI generated:', response.data.choices[0].text.trim());
  });

  test('11Labs voice generation', async () => {
    if (!process.env.ELEVENLABS_API_KEY) {
      console.log('⏭️ Skipping 11Labs test (no API key)');
      return;
    }

    const audioBuffer = await elevenlabs.textToSpeech({
      text: "Hello John, this is a test greeting",
      voice_id: "21m00Tcm4TlvDq8ikWAM" // Default voice
    });
    
    expect(audioBuffer).toBeTruthy();
    expect(audioBuffer.length).toBeGreaterThan(1000);
    
    // Save test output
    const fs = require('fs');
    fs.writeFileSync('test-output/test-voice.mp3', audioBuffer);
    console.log('✅ 11Labs generated: test-output/test-voice.mp3');
  });

  test('Firebase integration', async () => {
    const { initializeApp } = require('firebase/app');
    const { getFirestore, collection, addDoc, getDocs } = require('firebase/firestore');
    
    const app = initializeApp({
      projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID
    });
    const db = getFirestore(app);
    
    // Add test document
    const docRef = await addDoc(collection(db, 'test'), {
      message: 'AI integration test',
      timestamp: new Date()
    });
    
    expect(docRef.id).toBeTruthy();
    
    // Read back
    const querySnapshot = await getDocs(collection(db, 'test'));
    const docs = querySnapshot.docs.map(doc => doc.data());
    
    expect(docs).toHaveLength(1);
    expect(docs[0].message).toBe('AI integration test');
    
    console.log('✅ Firebase document created:', docRef.id);
  });
});
```

This development tools configuration ensures AI agents have all necessary context, rules, and utilities to work effectively in parallel while maintaining code quality and coordination.