#!/bin/bash
while true; do
    echo "⏰ $(date): Autonomous session running..."
    echo "🧠 Orchestrator PID: $ORCHESTRATOR_PID ($(ps -p $ORCHESTRATOR_PID >/dev/null && echo "RUNNING" || echo "STOPPED"))"
    echo "🤖 Agents PID: $AGENTS_PID ($(ps -p $AGENTS_PID >/dev/null && echo "RUNNING" || echo "STOPPED"))"
    echo "📊 Session files: $(ls -la | wc -l) files created"
    echo "---"
    sleep 300  # Check every 5 minutes
done
