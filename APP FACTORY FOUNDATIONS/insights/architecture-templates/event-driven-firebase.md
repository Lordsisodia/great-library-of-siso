# Event-Driven Firebase Architecture Template

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## Overview

**Architecture Type**: Event-driven broker architecture
**Platform**: Firebase
**Use Case**: Real-time applications with background processing
**Quote**: "This is by far my favorite architecture that we are currently using on our own SAS product"

## Core Architecture Pattern

### Event Flow:
```
Frontend Action → Firestore Document → Triggered Function → Background Processing → Real-time Update
```

### Example Implementation (DM Outreach System):
1. **User uploads video** → Creates document in Firestore
2. **Firestore change triggers function** → Processes video in background
3. **Function updates document** → Frontend receives real-time update
4. **User sees live status** → Downloads result when complete

## Firebase Components

### 1. Firestore (Database)
- **Purpose**: Event broker and data storage
- **Pattern**: Documents trigger functions on changes
- **Benefit**: Real-time subscriptions for frontend updates
- **Security**: Rules-based access control

### 2. Firebase Functions (Backend Processing)
- **Purpose**: Serverless background processing
- **Trigger**: Firestore document changes
- **Scaling**: Automatic based on load
- **Integration**: Direct access to other Firebase services

### 3. Firebase Storage (File Handling)  
- **Purpose**: File uploads and downloads
- **Integration**: Seamless with Functions and Firestore
- **Security**: Rules-based access like Firestore

### 4. Firebase Authentication
- **Purpose**: User management and security
- **Integration**: Built-in with all Firebase services
- **Feature**: `onCall` functions handle auth automatically

## Implementation Benefits

### Developer Experience
**Quote**: "Firebase significantly improves the developer experience, which means that you can ship much faster"

**Key Advantages**:
- **One command deployment**: `firebase deploy`
- **Automatic scaling**: No server management
- **Built-in security**: Authentication handled by platform
- **Real-time updates**: No additional WebSocket setup needed

### Cost Efficiency
- **Firebase pricing**: "Extremely cheap"
- **No-cost tier**: Available for storage and other services
- **Pay-per-use**: Only pay for what you consume
- **No server maintenance**: Eliminates DevOps overhead

## Architecture Template Structure

### Frontend (React/Next.js):
```typescript
// Real-time subscription to job status
useEffect(() => {
  const unsubscribe = onSnapshot(
    collection(db, 'greetingJobs'),
    (snapshot) => {
      setJobs(snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })));
    }
  );
  return unsubscribe;
}, []);
```

### Backend (Firebase Functions):
```typescript
// Triggered function on document creation
export const processGreetingJob = functions.firestore
  .document('greetingJobs/{jobId}')
  .onCreate(async (snap, context) => {
    const jobData = snap.data();
    
    // Process in background
    const result = await processVideo(jobData);
    
    // Update document with result
    await snap.ref.update({
      status: 'completed',
      downloadUrl: result.url,
      updatedAt: FieldValue.serverTimestamp()
    });
  });
```

### Database Structure (Firestore):
```javascript
// Collection: greetingJobs
{
  id: "auto-generated",
  userId: "user-id",
  status: "processing", // pending -> processing -> completed -> failed
  inputVideo: "storage-url",
  prospectName: "John Doe",
  voiceId: "11labs-voice-id",
  greetingEndSecond: 1.5,
  downloadUrl: null, // populated when complete
  createdAt: timestamp,
  updatedAt: timestamp
}
```

## Security Implementation

### Firestore Rules:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own greeting jobs
    match /greetingJobs/{jobId} {
      allow read, write: if request.auth != null 
        && resource.data.userId == request.auth.uid;
    }
  }
}
```

### Storage Rules:
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Users can upload to their own folder
    match /user-uploads/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null 
        && request.auth.uid == userId;
    }
  }
}
```

## Deployment Process

### 1. Initial Setup:
```bash
# Initialize Firebase project
firebase init

# Select: Functions, Firestore, Storage, Hosting (as needed)
```

### 2. Configuration:
```bash
# Frontend environment (.env)
NEXT_PUBLIC_FIREBASE_API_KEY=your-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-domain
# ... other config

# Backend service account (functions folder)
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account-key.json
```

### 3. Deployment:
```bash
# Deploy everything
firebase deploy

# Deploy specific services
firebase deploy --only functions
firebase deploy --only firestore:rules
```

## Advanced Patterns

### 1. Multi-Stage Processing:
```
Upload → Validation Job → Processing Job → Notification Job → Completion
```

### 2. Error Handling:
```typescript
// Retry logic in functions
export const retryFailedJobs = functions.pubsub
  .schedule('every 5 minutes')
  .onRun(async (context) => {
    const failedJobs = await db.collection('greetingJobs')
      .where('status', '==', 'failed')
      .where('retryCount', '<', 3)
      .get();
      
    // Process retries...
  });
```

### 3. Real-time Status Updates:
```typescript
// Granular status updates
const statusUpdates = [
  'pending',
  'validating-input', 
  'processing-audio',
  'generating-video',
  'uploading-result',
  'completed'
];
```

## Integration with AI Coding Workflow

### Step 1 (Architecture): Use this template
### Step 2 (Types): Define Firestore document interfaces
### Step 3 (Tests): Create integration tests with Firebase emulators  
### Step 4 (Implementation): Build functions and frontend
### Step 5 (Documentation): Update ADR with Firebase-specific decisions

## When to Use This Architecture

### ✅ Good For:
- Real-time applications
- Background job processing
- File upload/processing workflows
- User-facing dashboards with live updates
- MVPs and rapid prototyping

### ❌ Not Ideal For:
- High-frequency trading systems
- Complex relational data models
- Applications requiring custom server logic
- Systems needing fine-grained performance control

## Template Files Structure

```
project/
├── frontend/
│   ├── .env.example
│   └── lib/firebase.js
├── functions/
│   ├── .env.example
│   ├── index.js
│   └── service-account-key.json.example
├── firestore.rules
├── storage.rules
└── firebase.json
```

This architecture template provides a solid foundation for event-driven applications with Firebase, eliminating much of the complexity around real-time updates and background processing.