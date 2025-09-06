# Firebase Setup Guide for AI Development

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## Why Firebase for AI Development

**Developer Experience Benefits**:
- "Firebase significantly improves the developer experience, which means that you can ship much faster"
- One command deployment: `firebase deploy`
- Built-in security with authentication
- Real-time updates without additional setup
- Extremely cheap pricing with no-cost tiers

## Step-by-Step Setup Process

### 1. Create Firebase Project
```
1. Go to Firebase Console
2. Click "Create Project"
3. Enter project name (e.g., "dm-outreach-agent")
4. Optional: Disable Google Analytics (unless needed)
5. Click "Create Project"
```

### 2. Enable Required Services

#### Firestore Database:
```
1. Go to "Build" → "Firestore Database"
2. Click "Create database"
3. Choose "Start in test mode" (for development)
4. Select location closest to users
5. Click "Done"
```

#### Firebase Storage:
```
1. Go to "Build" → "Storage"  
2. Click "Get started"
3. Start in test mode
4. Choose same location as Firestore
5. Note: May require upgrading to paid plan
```

#### Authentication:
```
1. Go to "Build" → "Authentication"
2. Click "Get started"
3. Go to "Sign-in method" tab
4. Enable "Email/Password"
5. Optional: Enable "Google" provider
```

### 3. Create Web App Registration
```
1. Go to Project Settings (gear icon)
2. Scroll to "Your apps" section
3. Click web icon (</>)
4. Enter app nickname
5. Optional: Check "Set up Firebase Hosting"
6. Click "Register app"
7. **IMPORTANT**: Copy the config object
```

**Config Object Example**:
```javascript
const firebaseConfig = {
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id", 
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "your-app-id"
};
```

## Frontend Configuration

### Environment Setup:
**File**: `frontend/.env`
```bash
NEXT_PUBLIC_FIREBASE_API_KEY=your-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=123456789
NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id
```

### Firebase Initialization:
**File**: `frontend/lib/firebase.js`
```javascript
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const auth = getAuth(app);
export const storage = getStorage(app);
```

## Backend Configuration (Functions)

### Service Account Setup:
```
1. Go to Project Settings → "Service accounts" tab
2. Click "Generate new private key"
3. Download JSON file
4. Move to `functions/` folder
5. Rename to `service-account-key.json`
6. Add to .gitignore
```

### Environment Setup:
**File**: `functions/.env`
```bash
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account-key.json
```

### Functions Initialization:
**File**: `functions/index.js`
```javascript
const functions = require('firebase-functions');
const admin = require('firebase-admin');

// Initialize Firebase Admin
const serviceAccount = require('./service-account-key.json');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: 'your-project.appspot.com'
});

const db = admin.firestore();
const bucket = admin.storage().bucket();

// Export your functions here
exports.processGreetingJob = functions.firestore
  .document('greetingJobs/{jobId}')
  .onCreate(async (snap, context) => {
    // Function logic here
  });
```

## Security Rules Configuration

### Firestore Rules:
**File**: `firestore.rules`
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow users to read/write their own data
    match /greetingJobs/{jobId} {
      allow read, write: if request.auth != null 
        && resource.data.userId == request.auth.uid;
    }
  }
}
```

### Storage Rules:  
**File**: `storage.rules`
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Allow users to upload to their own folder
    match /user-uploads/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null 
        && request.auth.uid == userId;
    }
  }
}
```

## Deployment Configuration

### Firebase Config:
**File**: `firebase.json`
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
  "storage": {
    "rules": "storage.rules"
  },
  "hosting": {
    "public": "frontend/out",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"]
  }
}
```

### Project Configuration:
**File**: `.firebaserc`
```json
{
  "projects": {
    "default": "your-project-id"
  }
}
```

## Common Issues and Solutions

### 1. Index Creation Required
**Problem**: Error message about missing indexes
**Solution**: 
- Click the provided link in console
- Click "Create Index" 
- Wait for completion

### 2. Permission Denied Errors
**Problem**: Firestore access denied in development
**Solution**: 
- Temporarily set rules to `allow read, write: if true;`
- Deploy proper rules before production
- **Never leave open rules in production**

### 3. Google Sign-In Setup
**Problem**: Google authentication not working
**Solution**:
- Enable Google provider in Firebase Console
- Add authorized domains in Authentication settings
- Configure OAuth consent screen in Google Cloud Console

## Development vs Production

### Development Setup:
```bash
# Use emulators for local development
firebase emulators:start

# Connect to emulators in code
if (process.env.NODE_ENV === 'development') {
  connectFirestoreEmulator(db, 'localhost', 8080);
  connectAuthEmulator(auth, 'http://localhost:9099');
}
```

### Production Deployment:
```bash
# Deploy all services
firebase deploy

# Deploy specific services
firebase deploy --only functions
firebase deploy --only firestore:rules
firebase deploy --only hosting
```

## Integration with AI Workflow

### Template Benefits:
- **Rules files included**: Security handled by AI templates
- **Full control**: All templates in repo, not external services
- **AI-friendly**: Clear structure for agent understanding

### Workflow Integration:
1. **Architecture Phase**: Use Firebase template structure
2. **Types Phase**: Define Firestore document interfaces
3. **Tests Phase**: Use Firebase emulators for testing
4. **Implementation**: Deploy functions and frontend
5. **Documentation**: Update ADR with Firebase decisions

## Cost Optimization

### Free Tier Limits:
- **Firestore**: 1 GiB storage, 50K reads/day
- **Functions**: 125K invocations/month
- **Storage**: 5 GB, 1 GB/day downloads
- **Hosting**: 10 GB storage, 10 GB/month transfer

### Monitoring Usage:
- Check Firebase Console usage tab
- Set up billing alerts
- Monitor function execution times
- Optimize queries to reduce reads

This setup provides a complete Firebase configuration optimized for AI-driven development workflows.