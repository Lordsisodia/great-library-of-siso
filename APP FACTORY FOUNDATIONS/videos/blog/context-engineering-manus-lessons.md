# Context Engineering for AI Agents: Lessons from Building Manus

- **Source**: Manus AI Blog - https://manus.ai/blog/context-engineering-lessons
- **Date**: July 19, 2025
- **Author**: Yichao 'Peak' Ji
- **Company**: Manus AI
- **Key Topics**: Context engineering, AI agent architecture, production optimization, KV-cache optimization

## Core Business Decision & Philosophy

### **Build vs Train Decision**
**Critical Choice**: Context engineering vs end-to-end model training
- **Historical Context**: Previous startup experience with training models from scratch
- **Learning**: Custom models became obsolete when GPT-3 and BERT emerged
- **Strategic Decision**: Bet on context engineering for faster iteration cycles
- **Result**: Ship improvements in hours instead of weeks
- **Philosophy**: If model progress is rising tide, be the boat (context engineering), not the pillar (custom models)

### **"Stochastic Graduate Descent" Approach**
**Reality**: Context engineering is experimental science requiring iteration
- **Process**: Architecture searching + prompt fiddling + empirical guesswork
- **Result**: Rebuilt agent framework 4 times to find optimal approach
- **Methodology**: Manual, experimental, but effective approach
- **Goal**: Share lessons to help others converge faster

## Six Core Context Engineering Principles

### 1. **Design Around the KV-Cache** 
**Most Important Metric**: KV-cache hit rate directly affects latency and cost

#### **Why KV-Cache Matters**
- **Agent Architecture**: Chain of tool uses with growing context per iteration
- **Context Growth**: Action → Observation → Append to context → Next iteration
- **Token Ratio**: Manus averages 100:1 input-to-output ratio (vs chatbots)
- **Cost Impact**: Claude Sonnet cached tokens: $0.30/MTok vs uncached: $3.00/MTok (10x difference)

#### **KV-Cache Optimization Strategies**
- **Stable Prompt Prefix**: Single token difference invalidates entire cache downstream
- **Avoid Timestamps**: Precise timestamps (to second) kill cache hit rates completely
- **Append-Only Context**: Never modify previous actions/observations
- **Deterministic Serialization**: Ensure JSON key ordering is consistent
- **Explicit Cache Breakpoints**: Manual insertion for frameworks without auto-caching
- **Session ID Routing**: Consistent routing across distributed workers

### 2. **Mask, Don't Remove** 
**Problem**: Growing action space makes agents dumber as tool count explodes

#### **Anti-Pattern: Dynamic Tool Loading**
- **Temptation**: Dynamically add/remove tools mid-iteration
- **Problem 1**: Tool definitions invalidate KV-cache for all subsequent content
- **Problem 2**: Previous actions reference undefined tools, causing confusion
- **Reality**: Tool explosion from user-configurable integrations (MCP, etc.)

#### **Solution: Context-Aware Tool Masking**
- **Method**: Mask token logits during decoding instead of removing tools
- **Implementation**: Response prefill with three modes:
  - **Auto**: Model chooses to call function or not
  - **Required**: Must call function, choice unconstrained  
  - **Specified**: Must call from specific subset
- **Design**: Consistent tool naming prefixes (browser_, shell_) enable group masking
- **Result**: Stable agent loop under model-driven architecture

### 3. **Use File System as Context**
**Problem**: Even 128K+ context windows insufficient for real agent tasks

#### **Context Window Limitations**
- **Huge Observations**: Web pages, PDFs blow past context limits
- **Performance Degradation**: Model performance drops with long contexts
- **Cost Issues**: Long inputs expensive even with prefix caching
- **Compression Risk**: Aggressive compression causes information loss
- **Prediction Challenge**: Can't know which observation becomes critical later

#### **File System as Ultimate Context**
- **Advantages**: Unlimited size, persistent, directly operable by agent
- **Strategy**: Model learns to write/read files as structured external memory
- **Compression**: Always restorable (keep URLs, file paths, references)
- **Vision**: SSMs could excel with file-based memory vs context-based attention

### 4. **Manipulate Attention Through Recitation**
**Observation**: Manus creates todo.md files and updates them step-by-step

#### **Long Context Attention Problems**
- **Scale**: Typical Manus task requires ~50 tool calls
- **Drift Risk**: LLMs lose focus on original objectives
- **Lost-in-Middle**: Early goals forgotten in long contexts

#### **Solution: Goal Recitation**
- **Method**: Constantly rewrite todo list at end of context
- **Effect**: Pushes global plan into model's recent attention span
- **Mechanism**: Natural language biases model focus toward task objective
- **Result**: Reduced goal misalignment without architectural changes

### 5. **Keep the Wrong Stuff In**
**Counterintuitive**: Don't hide errors and failures from agent context

#### **Common Anti-Pattern**
- **Impulse**: Clean up traces, retry actions, reset model state
- **Problem**: Erasing failure removes evidence for learning
- **Result**: Model can't adapt or learn from mistakes

#### **Error-Inclusive Context Strategy**
- **Method**: Leave failed actions and stack traces in context
- **Effect**: Model implicitly updates beliefs, shifts prior away from similar actions
- **Benefit**: Reduces chance of repeating same mistakes
- **Philosophy**: Error recovery indicates true agentic behavior
- **Gap**: Underrepresented in academic work and benchmarks

### 6. **Don't Get Few-Shotted**
**Problem**: Few-shot examples can create harmful repetitive patterns

#### **The Mimicry Trap**
- **Behavior**: Models excel at imitating context patterns
- **Risk**: Similar action-observation pairs create rigid patterns
- **Danger**: Model follows pattern even when no longer optimal
- **Example**: Resume review batch - agent falls into repetitive rhythm

#### **Pattern-Breaking Solutions**
- **Increase Diversity**: Structured variation in actions and observations
- **Randomization**: Different serialization templates, alternate phrasing
- **Format Variation**: Minor noise in order and formatting
- **Result**: Controlled randomness breaks patterns, improves attention
- **Principle**: Uniform context = brittle agent

## Technical Implementation Details

### **Agent Loop Architecture**
```
User Input → Context + Tool Definitions → Model Selection → 
Action Execution → Observation → Context Append → Next Iteration
```

### **Tool Definition Strategy**
- **Placement**: Near front of context after serialization
- **Naming Convention**: Consistent prefixes enable group operations
- **Cache Optimization**: Stable definitions preserve KV-cache
- **Logits Masking**: Runtime constraint without definition changes

### **Context Management Pipeline**
1. **Prefix Stability**: System prompt, tool definitions unchanged
2. **Append Operations**: New actions/observations only
3. **Compression**: Restorable reduction (keep references, drop content)
4. **Cache Breakpoints**: Strategic placement for inference optimization
5. **Attention Manipulation**: Goal recitation at context end

## Production Insights & Lessons

### **Infrastructure Considerations**
- **KV-Cache Architecture**: Most critical performance metric
- **Distributed Inference**: Session consistency across workers
- **Cost Optimization**: 10x savings through cache optimization
- **Latency Impact**: TTFT dramatically improved with cache hits

### **Model Selection Criteria**
- **Context Engineering vs Training**: Faster iteration wins
- **Orthogonal to Progress**: Context engineering improves regardless of base model
- **Future-Proofing**: Same techniques work across model generations
- **Development Speed**: Hours vs weeks for improvements

### **Real-World Validation**
- **Scale**: Tested across millions of users
- **Iteration Count**: Rebuilt framework 4 times
- **Task Complexity**: Average 50 tool calls per task
- **Success Metrics**: Error recovery, goal alignment, performance

## Strategic Business Implications

### **Technical Debt Prevention**
- **Avoid Custom Models**: High risk of obsolescence
- **Bet on Context**: Orthogonal improvements to base capabilities
- **Fast Feedback**: Critical for pre-PMF product development
- **Architecture Resilience**: Survives model generation changes

### **Competitive Advantages**
- **Development Speed**: Hours vs weeks for improvements
- **Cost Efficiency**: 10x savings through optimization
- **Performance**: Better attention, error recovery, goal alignment
- **Scalability**: File system enables unlimited context expansion

### **Future Vision**
- **SSM Potential**: State space models with file-based memory
- **Agentic Evolution**: Beyond transformer attention limitations
- **Error Recovery**: Key differentiator for real agentic behavior
- **Context Science**: Emerging field requiring empirical approach

## Key Takeaways for Practitioners

### **Architecture Decisions**
1. **Choose Context Engineering** over custom model training
2. **Design for KV-Cache** as primary optimization target
3. **Use File System** as unlimited, persistent context store
4. **Implement Tool Masking** instead of dynamic loading
5. **Preserve Error Context** for agent learning
6. **Break Pattern Mimicry** through controlled variation

### **Implementation Priorities**
1. **Stable Prefixes**: Eliminate timestamp precision, ensure deterministic serialization
2. **Cache Breakpoints**: Strategic placement for inference frameworks
3. **Tool Naming**: Consistent prefixes for group operations
4. **Goal Recitation**: Todo lists and objective reminders
5. **Error Inclusion**: Failed actions remain in context
6. **Format Diversity**: Prevent few-shot rigidity

### **Performance Metrics**
- **Primary**: KV-cache hit rate
- **Secondary**: Task completion rate, error recovery success
- **Cost**: Input token optimization through caching
- **Latency**: Time-to-first-token improvements
- **Quality**: Goal alignment maintenance in long tasks

## Conclusion Philosophy

### **Engineering Principles**
- **Context engineering is essential**: No amount of raw capability replaces proper memory/environment/feedback design
- **Empirical approach required**: "Stochastic Graduate Descent" through experimentation
- **Error recovery indicates true agency**: Distinguish from chatbot behavior
- **Pattern diversity prevents brittleness**: Uniform context creates fragile agents

### **Business Strategy**
- **Fast iteration cycles**: Context engineering enables hour-level improvements
- **Model orthogonality**: Benefits persist across model generations  
- **Production focus**: Real-world testing at scale reveals true patterns
- **Future preparation**: Context management principles will remain relevant

### **Industry Impact**
- **Emerging science**: Context engineering becoming essential discipline
- **Practical wisdom**: Lessons from millions of users and multiple rebuilds
- **Shared knowledge**: Helping others avoid painful iterations
- **Foundational work**: "The agentic future will be built one context at a time"

This comprehensive analysis from Manus AI provides battle-tested insights for anyone building production AI agent systems, emphasizing that context engineering is not just a technical consideration but a fundamental architectural discipline for successful AI agent deployment.