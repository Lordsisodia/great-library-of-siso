# Test-First Development for AI Coding

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## The Most Valuable Technique

**Quote**: "This is hands down the most valuable technique in AI coding"

**Core Principle**: Write tests FIRST when AI still has complete context for the feature

## Why Test-First is Critical for AI

### Context Loss Prevention
- **Problem**: Complex features = long conversation history
- **Result**: "All coding assistants typically remove some messages in the middle or beginning"
- **Solution**: Write tests while context is complete
- **Benefit**: "AI literally can't fool itself because it's going to run the test"

### AI Memory Limitation
- **Issue**: "AI simply loses the necessary context in order to complete it reliably"
- **Prevention**: Tests preserve original intent even when context is lost
- **Recovery**: Tests catch bugs when AI forgets details from earlier in conversation

## Integration Tests > Unit Tests (During Development)

### Development Phase Strategy
**Quote**: "I personally prefer to always run integration tests until the functionality is completed. And then after the functionality is completed you can replace the integration test with the unit test."

### Integration Test Requirements
- **Real APIs**: Not mock data
- **Real files**: Actual video/audio processing
- **Real databases**: Connected to actual services
- **Real credentials**: Use test API keys, not fake data

### Unit Test Timing
- **When**: After feature is 100% complete and working
- **Why**: Faster execution in CI/CD pipelines
- **Benefit**: Cheaper to run at scale

## Test Verification Strategies

### The Reality Check Method
**Quote**: "Ask it like where can I find the generated video. And if it's not there, then it's probably not testing something properly."

**Process**:
1. AI claims tests are passing
2. Ask: "Where can I find the actual result?"
3. If result doesn't exist → AI is using mock data
4. Force AI to use real APIs and generate real outputs

### Test Reading for Non-Coders
**Quote**: "If you don't know how to code, test cases are actually really simple to read. All you need to do is simply read the name of the function and that typically describes quite clearly what the test case actually does."

**Example**: `test_synthesize_greeting_loan_name` → Testing loan name generation like "Christopher"

## Test-Driven Feedback Loop

### Close the Loop Strategy
**Quote**: "Make sure you tell your AI to run the test. Do not run the test yourself because this essentially going to allow you to close the feedback loop."

**Benefits**:
- AI sees errors directly
- No manual error passing between terminal and AI
- AI keeps working until tests pass
- Continuous improvement without human intervention

### Feedback Loop Components
```
1. AI writes test with real data
2. AI runs test automatically  
3. AI sees actual results/errors
4. AI fixes issues based on test feedback
5. Repeat until test passes
```

## Real Data Testing Requirements

### What Constitutes "Real Data"
**11Labs API Example**:
- ❌ Mock: Fake API responses
- ✅ Real: Actual API calls with test keys generating audio files

**FFmpeg Service Example**:
- ❌ Mock: Fake video processing
- ✅ Real: Actual video file in test data folder being processed

**Database Example**:
- ❌ Mock: In-memory fake database
- ✅ Real: Firebase emulators with actual Firestore operations

### Test Data Management
```markdown
## Test Data Setup
1. Create `/test-data/` folder in project
2. Add sample files (video, audio, images) 
3. Use real but non-production API keys
4. Test with actual user scenarios
5. Validate real file outputs are generated
```

## Multi-Agent Test Coordination

### Individual Agent Testing
**Strategy**: Each agent must have integration tests with real data before handoff

**Example**:
- **Agent 1 (11Labs)**: Must generate actual audio file
- **Agent 2 (FFmpeg)**: Must process real video file  
- **Agent 3 (Firebase)**: Must create real Firestore documents

### Cross-Agent Integration Testing
**Final Integration**: Test that combines all agent outputs
- Use Agent 1's real audio output
- Process with Agent 2's real video processing
- Store results with Agent 3's real database operations
- Verify end-to-end real workflow

## Test Simplification Strategy

### Avoid Over-Testing During Development
**Problem**: "It added way too many different test cases for a lot of different names"

**Solution**: "Simplify this test and leave only one test case for hey John"

**Principle**: During development, focus on one good test case that proves the concept works

### Test Expansion Timeline
```
Phase 1 (Development): One solid integration test with real data
Phase 2 (Feature Complete): Add edge cases and error scenarios  
Phase 3 (Production Ready): Convert to unit tests + comprehensive suite
```

## AI Testing Pitfalls

### The Mock Data Trap
**Warning Signs**:
- AI claims tests pass but no output files generated
- Test uses hardcoded responses instead of API calls
- AI references test files that don't exist

**Recovery**: Force AI to show you the actual test output files

### The Hallucination Test
**Problem**: "Claude tells me that it tested all functionality end to end. However, I can't actually see the final video"

**Solution**: Always ask AI to prove test results with actual file paths and outputs

### The Lazy Implementation
**Issue**: "AI didn't even run the test itself"
**Fix**: Explicitly tell AI to run the test and show results
**Benefit**: Prevents AI from claiming success without actual execution

## Test-First Implementation Example

### Step-by-Step Process:
```markdown
1. Define feature requirements while context is fresh
2. Write integration test with real API calls
3. AI runs test - should fail initially (red)
4. AI implements feature to make test pass (green)
5. AI refactors while keeping test passing (refactor)
6. AI documents any issues encountered in ADR
```

### Template Test Structure:
```typescript
// Integration test template
describe('GreetingVideoProcessor', () => {
  it('should process real video with 11Labs API', async () => {
    // Arrange - use real test data
    const testVideo = './test-data/sample-greeting.mp4';
    const voiceId = process.env.ELEVENLABS_VOICE_ID;
    const prospectName = 'John';
    
    // Act - call real services
    const result = await processGreetingVideo({
      videoFile: testVideo,
      voiceId,
      prospectName,
      greetingEndSecond: 1.5
    });
    
    // Assert - verify real outputs
    expect(result.downloadUrl).toBeDefined();
    expect(fs.existsSync(result.localPath)).toBe(true);
    
    // Cleanup
    await cleanupTestFiles(result.localPath);
  });
});
```

This test-first approach is the foundation that makes the entire 5-step workflow reliable for AI development.