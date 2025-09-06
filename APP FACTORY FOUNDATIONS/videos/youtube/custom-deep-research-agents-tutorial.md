# How to Build Custom Deep Research Agents (Better than OpenAI)

**Video Length:** ~19 minutes  
**Key Topic:** Building custom deep research agents using local data and context engineering  
**Source:** YouTube Transcript  
**URL:** https://www.youtube.com/watch?v=dkQMchRJkBk

---

## Core Value Proposition
- Create deep research agents **10x better than OpenAI** through customization
- Use **4 context engineering techniques** for superior performance
- **Business opportunity:** Charge per report (hundreds of dollars value vs human analyst)
- Works with local data and custom business contexts

## Key Technologies & Models

### OpenAI Deep Research Models
- **Two new deep research models** recently released by OpenAI
- **Web search model** - $10 for 1K tool calls (incredibly cheap)
- Only supported via **Responses API** (not Assistants or Chat Completions)
- **MCP (Model Context Protocol)** officially supported by OpenAI

### Framework Architecture
- **New Agencies Swarm Framework** - fully based on Responses API
- Extension of **Agents SDK** for future feature compatibility
- **Hosted MCP Tools** - calls MCP servers remotely from OpenAI servers
- **Vector stores** integration for file search capabilities

## Two Research Agent Architectures

### 1. Basic Research Agent (Single Agent)
- **One agent** powered by offer mini model
- **Web search tool** for internet research
- **Hosted MCP tool** for local data integration
- Simple, direct research execution

### 2. Multi-Agent Research System (3 Agents)
**Exact copy of OpenAI's deep research but fully customizable:**
1. **Clarifying Agent** - asks follow-up questions to understand requirements
2. **Instruction Builder Agent** - creates comprehensive research prompts
3. **Research Agent** - performs the actual deep research
4. **Routing Agent** - directs requests between agents

## Technical Implementation

### MCP Server Setup
- **Hosted MCP requirement** - must be internet-accessible (not localhost)
- **Two required tools:** Search tool and Fetch tool
- **Dedicated parameter list** as per OpenAI guidelines
- **Ngrok integration** for exposing local servers to internet
- **File search recreation** using OpenAI vector stores

### Environment Setup
```bash
# Clone repo and setup
git clone [repo]
python -m venv venv
pip install -r requirements.txt

# Start MCP server
python MCP/start_mcp_server.py

# Expose via Ngrok
ngrok http [port]/sse
```

### Configuration
- **OpenAI API key** required
- **Vector store ID** (optional - auto-created if not specified)
- **Max tool calls parameter** to control research duration
- **Custom instructions** for business context

## Context Engineering Techniques

### 1. Custom File Integration
- Add business documents to `/files` folder
- **LLM-full.txt documentation** from modern platforms
- Framework docs, business data, previous reports
- Automatic vector store creation and indexing

### 2. Custom MCP Development
- **GitHub MCP server example** for code analysis
- **Prompt engineering** with coding assistants (Claude, Cursor, Gemini)
- **Parameter consistency** requirement for compatibility
- **Business-specific data connectors**

### 3. Instruction Customization
- **Essay-style instructions** (not AI-generated)
- **Business overview and context**
- **Specific research objectives**
- **Quality standards and guidelines**

### 4. Multi-Agent Orchestration
- **Structured outputs** for question generation
- **Sequential agent handoffs**
- **Context preservation** across agents
- **Dynamic prompt building**

## Real-World Business Applications

### Marketing Agency Example
- **Customer campaign data** integration
- **Previous campaign analysis** via MCP
- **Customer detail databases**
- **ROI and performance analytics**

### Software Framework Analysis
- **Framework documentation** comparison
- **Missing features identification**
- **Strategic recommendations**
- **Architecture improvement suggestions**

### Pricing Models
- **Per-report pricing** (hundreds of dollars value)
- **Human analyst replacement** (days of work → minutes)
- **Custom business intelligence**
- **Competitive analysis reports**

## Report Quality & Output

### Generated Report Features
- **PDF format** with professional formatting
- **18+ page comprehensive analysis**
- **Code snippets and examples**
- **Comparison tables**
- **Strategic recommendations**
- **Reference citations**
- **Executive summary**

### Example Analysis Results
- **Framework comparison** (Agency Swarm vs LangGraph vs CrewAI)
- **Missing feature identification** (Graph abstraction, UI/DSL)
- **Strategic recommendations** (Workflow layer, memory integration)
- **Architecture patterns** (ActFrom LangGraph method)
- **Strength analysis** (Lightweight framework benefits)

## Framework Development Insights

### Current Framework Status
- **200+ tests** in new version
- **Improved concurrency** and modularity
- **Lightweight architecture** advantage
- **Workflow layer** development in progress
- **Memory integration** planned features

### Platform Integration
- **No-code agent building** platform
- **GitHub deployment** direct integration
- **Client business integration**
- **Massive updates** coming soon

## Business Scalability

### Revenue Opportunities
- **Data-heavy client services** (marketing agencies)
- **Custom research reports**
- **Business intelligence automation**
- **Competitive analysis services**

### Time Savings
- **20+ minutes** automated research vs **multiple days** human effort
- **30+ tool calls** for comprehensive analysis
- **Immediate PDF generation**
- **Professional formatting** included

---

*Key Insight: The combination of custom MCP servers, local data integration, and multi-agent orchestration creates research capabilities that exceed OpenAI's default offerings through deep business context and specialized data access.*