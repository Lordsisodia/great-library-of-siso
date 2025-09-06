# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is "The Great Library of SISO" - a multi-domain AI research and development repository focused on autonomous agents, productivity optimization, and AI-assisted development methodologies. The repository implements production-validated patterns for multi-agent coordination with demonstrated 76% development time reduction.

## Development Commands

### Core Development
```bash
# Frontend/Backend/Voice concurrent development
npm run dev

# Run all tests
npm run test

# Build for production
npm run build

# Test specific components
npm run test:frontend
npm run test:backend
npm run test:voice

# Linting and type checking
npm run lint
npm run typecheck
```

### Testing Individual Components
```bash
# Run specific test file with Vitest
npm test -- path/to/test.spec.ts

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage
```

## High-Level Architecture

The repository is organized into 5 core research domains that require cross-file understanding:

### 1. **Trillion-Dollar-Intelligence/**
Contains the core AI intelligence system with 31+ YAML modules defining cognitive architecture. The central configuration is in `Scripts-And-Configs/shared/siso-intelligence-system.yml` which orchestrates all AI behaviors and patterns.

### 2. **Production-Development-Systems/**
Implements the "Architecture + Types + Tests = AI Railroad" methodology:
- Always create types first to prevent AI hallucination
- Use integration tests with real data over mocks
- Follow the 5-step workflow: Architecture → Types → Tests → Build → Document

### 3. **SISO-Ecosystem/**
Multi-agent coordination system using git worktrees for parallel development:
- Each agent operates in isolated worktree branches
- Agents coordinate through the SISO Legacy Wrapper
- System overview in `SISO-IDE/SYSTEM_OVERVIEW.md`

### 4. **MCP-Workhouse/**
Model Context Protocol implementations for advanced integrations. Contains 15+ MCP servers for various services (GitHub, databases, Notion, etc.). These enable context preservation across sessions.

### 5. **APP FACTORY FOUNDATIONS/**
Practical implementation frameworks including:
- YouTube integration tools (video fetcher, transcript analysis)
- Development templates and architecture patterns
- Component libraries in `components/development-tools/`

## Critical Development Patterns

### Multi-Agent Development
When working on features that span multiple components, use concurrent development:
- Frontend changes: Modify files in `/frontend` directories
- Backend changes: Update `/api-wrapper` or `/backend` components
- Voice interface: Edit `/voice-interface` modules

### Testing Philosophy
1. **Always write integration tests first** - test with real APIs and data
2. **Types before implementation** - define TypeScript interfaces/types before coding
3. **Test to prevent AI hallucination** - tests validate AI-generated code
4. **Use Vitest for new tests** - modern testing framework preferred

### Component Architecture
Each major component has its own package.json with specific scripts:
- Development tools: `APP FACTORY FOUNDATIONS/components/development-tools/package.json`
- Dashboard experiments: `SISO-Ecosystem/SISO-IDE/Experiments/sandbox-poc/siso-agent-dashboard-test/package.json`

### Intelligence System Integration
When implementing AI features, reference the core intelligence configuration:
- Primary config: `Trillion-Dollar-Intelligence/Scripts-And-Configs/shared/siso-intelligence-system.yml`
- This defines cognitive modules, decision patterns, and agent behaviors

## Important Notes

- **Node.js >= 16.0.0 required** for all components
- **TypeScript-first development** - always define types before implementation
- **Git worktrees** used for multi-agent coordination - check current branch before commits
- **Environment variables** needed for API integrations - check `.env.example` files
- **Real data testing** - avoid mocks, test with actual API responses when possible