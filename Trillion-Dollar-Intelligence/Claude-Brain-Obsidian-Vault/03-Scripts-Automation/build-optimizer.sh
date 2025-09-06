#!/bin/bash

# 10x Build Optimizer - Smart build process enhancement
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

log "🏗️ Build optimization started"

# Pre-build optimizations
optimize_prebuild() {
    # Clear cache for clean build
    if [[ -d "node_modules/.cache" ]]; then
        log "🧹 Clearing build cache"
        rm -rf node_modules/.cache/* 2>/dev/null || true
    fi
    
    # Check for TypeScript errors before build
    if [[ -f "tsconfig.json" ]]; then
        log "📘 Pre-build TypeScript check"
        npx tsc --noEmit 2>/dev/null || log "⚠️ TypeScript errors detected"
    fi
}

# Build performance monitoring
monitor_build() {
    local start_time=$(date +%s)
    
    # Run the actual build with monitoring
    npm run build 2>&1 | tee ~/.claude/logs/build-output.log
    local build_exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "⏱️ Build completed in ${duration}s (exit code: $build_exit_code)"
    
    # Log build metrics
    echo "$TIMESTAMP,$duration,$build_exit_code" >> ~/.claude/analytics/build-metrics.csv
    
    return $build_exit_code
}

# Post-build optimizations
optimize_postbuild() {
    local build_success=$1
    
    if [[ $build_success -eq 0 ]]; then
        log "✅ Build successful - running optimizations"
        
        # Bundle analysis for large builds
        if [[ -d "dist" ]] && command -v du >/dev/null; then
            size=$(du -sh dist 2>/dev/null | cut -f1)
            log "📦 Build size: $size"
            
            # Alert for large builds
            size_mb=$(du -sm dist 2>/dev/null | cut -f1)
            if [[ $size_mb -gt 10 ]]; then
                log "⚠️ Large build detected (${size_mb}MB) - consider optimization"
            fi
        fi
        
        # Auto-generate build report for SISO projects
        if pwd | grep -q "SISO"; then
            log "🏢 SISO project - generating build report"
            echo "Build completed at $TIMESTAMP" > dist/build-info.txt
            echo "Build size: $(du -sh dist 2>/dev/null | cut -f1)" >> dist/build-info.txt
        fi
        
    else
        log "❌ Build failed - analyzing errors"
        
        # Extract common error patterns
        if [[ -f ~/.claude/logs/build-output.log ]]; then
            if grep -q "out of memory" ~/.claude/logs/build-output.log; then
                log "🧠 Memory issue detected - suggesting NODE_OPTIONS=--max-old-space-size=4096"
            fi
            
            if grep -q "Module not found" ~/.claude/logs/build-output.log; then
                log "📦 Missing dependency detected - check package.json"
            fi
        fi
    fi
}

# Initialize analytics directory
mkdir -p ~/.claude/analytics

# Run optimization sequence
optimize_prebuild
monitor_build
build_success=$?
optimize_postbuild $build_success

log "🏁 Build optimization complete"
exit $build_success