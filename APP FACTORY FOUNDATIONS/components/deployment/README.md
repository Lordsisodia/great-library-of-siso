# 🚀 Deployment Components

Production-ready deployment configurations and DevOps automation for Firebase and multi-environment workflows.

## 📁 Component Structure

```
deployment/
├── firebase-configs/        # Firebase deployment configurations
│   ├── firebase.json        # Complete Firebase configuration
│   ├── .firebaserc          # Project environment mapping
│   ├── firebase-functions/  # Cloud Functions deployment
│   └── firebase-hosting/    # Static site hosting config
├── environment-management/  # Multi-environment setup
│   ├── staging-config/      # Staging environment
│   ├── production-config/   # Production environment
│   ├── development-config/  # Local development
│   └── env-switcher.js      # Environment switching utility
├── ci-cd-pipelines/         # Continuous deployment
│   ├── github-actions/      # GitHub Actions workflows
│   ├── gitlab-ci/           # GitLab CI configurations
│   ├── deployment-scripts/  # Automated deployment scripts
│   └── rollback-procedures/ # Rollback and recovery
├── monitoring-setup/        # Production monitoring
│   ├── error-tracking/      # Error monitoring setup
│   ├── performance-monitoring/ # Performance tracking
│   ├── logging-config/      # Centralized logging
│   └── alerting-rules/      # Alert configurations
└── security-configs/        # Production security
    ├── ssl-certificates/    # SSL/TLS configuration
    ├── firewall-rules/      # Security rules
    ├── backup-strategies/   # Data backup automation
    └── disaster-recovery/   # Disaster recovery plans
```

## 🚀 Quick Deployment

### 1. One-Command Firebase Deployment
```bash
# Copy Firebase configuration
cp components/deployment/firebase-configs/* ./

# Deploy everything
firebase deploy
```

### 2. Environment-Specific Deployment
```bash
# Deploy to staging
firebase use staging
firebase deploy

# Deploy to production  
firebase use production
firebase deploy
```

### 3. Targeted Deployments
```bash
# Deploy only functions
firebase deploy --only functions

# Deploy only hosting
firebase deploy --only hosting

# Deploy only rules
firebase deploy --only firestore:rules,storage:rules
```

## 🔧 Firebase Configuration

### Complete firebase.json
```json
{
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "functions": {
    "source": "functions",
    "predeploy": ["npm --prefix \"$RESOURCE_DIR\" run build"],
    "runtime": "nodejs18"
  },
  "hosting": {
    "public": "build",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ],
    "headers": [
      {
        "source": "/static/**",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000"
          }
        ]
      }
    ]
  },
  "storage": {
    "rules": "storage.rules"
  },
  "emulators": {
    "auth": {
      "port": 9099
    },
    "firestore": {
      "port": 8080
    },
    "functions": {
      "port": 5001
    },
    "hosting": {
      "port": 5000
    },
    "storage": {
      "port": 9199
    },
    "ui": {
      "enabled": true,
      "port": 4000
    }
  }
}
```

### Environment Configuration (.firebaserc)
```json
{
  "projects": {
    "development": "your-project-dev",
    "staging": "your-project-staging", 
    "production": "your-project-prod"
  },
  "targets": {
    "your-project-prod": {
      "hosting": {
        "web": ["your-project-web"],
        "admin": ["your-project-admin"]
      }
    }
  }
}
```

## 🌍 Environment Management

### Staging Environment Setup
```bash
# staging-config/deploy-staging.sh
#!/bin/bash
echo "🚀 Deploying to staging..."

# Switch to staging project
firebase use staging

# Deploy with staging-specific configs
firebase deploy --only functions,firestore:rules,hosting

# Run smoke tests
npm run test:smoke-staging

echo "✅ Staging deployment complete"
```

### Production Deployment
```bash
# production-config/deploy-production.sh  
#!/bin/bash
echo "🚀 Deploying to production..."

# Safety checks
npm run test:all
npm run lint
npm run type-check

# Switch to production
firebase use production

# Deploy with zero downtime
firebase deploy --only functions,firestore:rules
firebase deploy --only hosting

# Run post-deployment tests
npm run test:production-health

echo "✅ Production deployment complete"
```

### Environment Switching Utility
```javascript
// environment-management/env-switcher.js
class EnvironmentSwitcher {
  constructor() {
    this.environments = ['development', 'staging', 'production'];
  }
  
  async switchTo(environment) {
    if (!this.environments.includes(environment)) {
      throw new Error(`Invalid environment: ${environment}`);
    }
    
    console.log(`🔄 Switching to ${environment}...`);
    
    // Switch Firebase project
    await this.runCommand(`firebase use ${environment}`);
    
    // Update environment variables
    await this.updateEnvFile(environment);
    
    // Restart development server if needed
    if (environment === 'development') {
      await this.restartDevServer();
    }
    
    console.log(`✅ Switched to ${environment}`);
  }
  
  async updateEnvFile(environment) {
    const envConfig = require(`./configs/${environment}.json`);
    const envContent = Object.entries(envConfig)
      .map(([key, value]) => `${key}=${value}`)
      .join('\n');
    
    await fs.writeFile('.env.local', envContent);
  }
}

module.exports = EnvironmentSwitcher;
```

## 🔄 CI/CD Pipeline Integration

### GitHub Actions Workflow
```yaml
# ci-cd-pipelines/github-actions/deploy.yml
name: Deploy to Firebase

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          npm ci
          cd functions && npm ci
          
      - name: Run tests
        run: |
          npm run test:all
          npm run lint
          npm run type-check
          
      - name: Build project
        run: npm run build

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          npm ci
          cd functions && npm ci
          
      - name: Build project
        run: npm run build
        
      - name: Deploy to staging
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          repoToken: '${{ secrets.GITHUB_TOKEN }}'
          firebaseServiceAccount: '${{ secrets.FIREBASE_SERVICE_ACCOUNT_STAGING }}'
          projectId: your-project-staging
          
  deploy-production:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          npm ci  
          cd functions && npm ci
          
      - name: Build project
        run: npm run build
        
      - name: Deploy to production
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          repoToken: '${{ secrets.GITHUB_TOKEN }}'
          firebaseServiceAccount: '${{ secrets.FIREBASE_SERVICE_ACCOUNT_PRODUCTION }}'
          projectId: your-project-prod
```

### Automated Deployment Script
```javascript
// deployment-scripts/auto-deploy.js
class AutoDeployer {
  constructor(options = {}) {
    this.environment = options.environment || 'staging';
    this.skipTests = options.skipTests || false;
    this.rollbackOnFailure = options.rollbackOnFailure || true;
  }
  
  async deploy() {
    console.log(`🚀 Starting deployment to ${this.environment}...`);
    
    try {
      // Pre-deployment checks
      if (!this.skipTests) {
        await this.runTests();
      }
      
      // Build project
      await this.buildProject();
      
      // Deploy to Firebase
      await this.deployToFirebase();
      
      // Post-deployment verification
      await this.verifyDeployment();
      
      console.log(`✅ Deployment to ${this.environment} successful!`);
      
    } catch (error) {
      console.error(`❌ Deployment failed: ${error.message}`);
      
      if (this.rollbackOnFailure) {
        await this.rollback();
      }
      
      throw error;
    }
  }
  
  async runTests() {
    console.log('🧪 Running tests...');
    await this.runCommand('npm run test:all');
    await this.runCommand('npm run lint');
    await this.runCommand('npm run type-check');
  }
  
  async buildProject() {
    console.log('🏗️ Building project...');
    await this.runCommand('npm run build');
  }
  
  async deployToFirebase() {
    console.log(`🚀 Deploying to ${this.environment}...`);
    await this.runCommand(`firebase use ${this.environment}`);
    await this.runCommand('firebase deploy');
  }
  
  async verifyDeployment() {
    console.log('🔍 Verifying deployment...');
    await this.runCommand(`npm run test:smoke-${this.environment}`);
  }
  
  async rollback() {
    console.log('🔄 Rolling back deployment...');
    await this.runCommand('firebase rollback functions');
  }
}

module.exports = AutoDeployer;
```

## 📊 Monitoring and Alerting

### Error Tracking Setup
```javascript
// monitoring-setup/error-tracking/sentry-config.js
import * as Sentry from '@sentry/node';
import * as functions from 'firebase-functions';

// Initialize Sentry for Firebase Functions
Sentry.init({
  dsn: functions.config().sentry.dsn,
  environment: functions.config().environment.name,
  integrations: [
    new Sentry.Integrations.Http({ tracing: true }),
    new Sentry.Integrations.Express({ app }),
  ],
  tracesSampleRate: 0.1,
});

// Error tracking middleware
export const errorTracker = {
  captureException: (error, context = {}) => {
    Sentry.withScope((scope) => {
      Object.keys(context).forEach(key => {
        scope.setTag(key, context[key]);
      });
      Sentry.captureException(error);
    });
  },
  
  captureMessage: (message, level = 'info', context = {}) => {
    Sentry.withScope((scope) => {
      Object.keys(context).forEach(key => {
        scope.setTag(key, context[key]);
      });
      Sentry.captureMessage(message, level);
    });
  }
};
```

### Performance Monitoring
```typescript
// monitoring-setup/performance-monitoring/performance-tracker.ts
export class PerformanceTracker {
  private metrics = new Map<string, number>();
  
  startTimer(operation: string): void {
    this.metrics.set(operation, Date.now());
  }
  
  endTimer(operation: string): number {
    const startTime = this.metrics.get(operation);
    if (!startTime) {
      throw new Error(`Timer not started for operation: ${operation}`);
    }
    
    const duration = Date.now() - startTime;
    this.metrics.delete(operation);
    
    // Log to monitoring service
    this.logMetric(operation, duration);
    
    return duration;
  }
  
  private logMetric(operation: string, duration: number): void {
    // Send to Firebase Performance Monitoring
    const trace = firebase.performance().trace(operation);
    trace.putMetric('duration', duration);
    trace.stop();
    
    // Also log to custom analytics
    console.log(`Performance: ${operation} took ${duration}ms`);
  }
}
```

## 🔐 Security Configuration

### SSL/TLS Setup
```javascript
// security-configs/ssl-certificates/ssl-config.js
export const sslConfig = {
  // Force HTTPS in production
  enforceSSL: process.env.NODE_ENV === 'production',
  
  // Security headers
  securityHeaders: {
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin'
  },
  
  // Content Security Policy
  csp: {
    'default-src': ["'self'"],
    'script-src': ["'self'", "'unsafe-inline'", "https://apis.google.com"],
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    'font-src': ["'self'", "https://fonts.gstatic.com"],
    'img-src': ["'self'", "data:", "https:"],
    'connect-src': ["'self'", "https://api.example.com"]
  }
};
```

### Backup Automation
```bash
# security-configs/backup-strategies/backup-firestore.sh
#!/bin/bash

PROJECT_ID="your-project-prod"
BUCKET_NAME="your-backup-bucket"
DATE=$(date +%Y-%m-%d-%H-%M-%S)
BACKUP_NAME="firestore-backup-$DATE"

echo "🔄 Starting Firestore backup..."

# Export Firestore data
gcloud firestore export gs://$BUCKET_NAME/backups/$BACKUP_NAME \
  --project=$PROJECT_ID \
  --async

echo "✅ Backup initiated: $BACKUP_NAME"

# Clean up old backups (keep last 30 days)
gsutil -m rm -r gs://$BUCKET_NAME/backups/firestore-backup-$(date -d '30 days ago' +%Y-%m-%d)*
```

## 🧪 Testing Integration

### Deployment Testing
```typescript
// Test deployment process
describe('Deployment Process', () => {
  test('staging deployment completes successfully', async () => {
    const deployer = new AutoDeployer({ environment: 'staging' });
    
    // Mock deployment steps
    const deployResult = await deployer.deploy();
    
    expect(deployResult.success).toBe(true);
    expect(deployResult.environment).toBe('staging');
  });
  
  test('production deployment has safety checks', async () => {
    const deployer = new AutoDeployer({ 
      environment: 'production',
      skipTests: false 
    });
    
    // Should run all tests before deployment
    await expect(deployer.deploy()).resolves.not.toThrow();
  });
});
```

### Smoke Tests
```typescript
// Verify deployment health
export const smokeTests = {
  async checkApiHealth(baseUrl: string): Promise<boolean> {
    try {
      const response = await fetch(`${baseUrl}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  },
  
  async checkDatabaseConnectivity(): Promise<boolean> {
    try {
      const testDoc = await firestore.collection('_test').doc('connectivity').get();
      return true;
    } catch {
      return false;
    }
  },
  
  async checkAuthenticationFlow(): Promise<boolean> {
    try {
      await firebase.auth().signInAnonymously();
      await firebase.auth().signOut();
      return true;
    } catch {
      return false;
    }
  }
};
```

## 🔄 Integration with AI Workflows

### Automated Deployment in AI Development
```typescript
// AI agents can trigger deployments
export const aiDeploymentIntegration = {
  async deployFeature(featureName: string, environment: 'staging' | 'production') {
    console.log(`🤖 AI Agent deploying ${featureName} to ${environment}...`);
    
    const deployer = new AutoDeployer({ 
      environment,
      rollbackOnFailure: true 
    });
    
    try {
      await deployer.deploy();
      
      // Notify AI agent of success
      return {
        success: true,
        message: `${featureName} deployed successfully to ${environment}`,
        url: `https://${environment}.your-app.com`
      };
      
    } catch (error) {
      // Notify AI agent of failure
      return {
        success: false,
        message: `Deployment failed: ${error.message}`,
        rollbackExecuted: true
      };
    }
  }
};
```

This deployment system provides comprehensive automation, monitoring, and security for production applications while integrating seamlessly with AI development workflows.