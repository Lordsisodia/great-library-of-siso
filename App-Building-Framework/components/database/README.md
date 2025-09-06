# 🗄️ Database Components

Production-ready database schemas, integration patterns, and real-time data management systems optimized for Firebase and AI development workflows.

## 📁 Component Structure

```
database/
├── firestore-schemas/       # Complete Firestore data models
│   ├── user-management/     # User profiles and authentication
│   ├── task-management/     # Task and project schemas
│   ├── real-time-updates/   # Event-driven data patterns
│   └── analytics-tracking/  # User analytics and metrics
├── security-rules/          # Firebase security configurations
│   ├── firestore.rules     # Database access control
│   ├── storage.rules       # File storage security
│   └── functions.rules     # Cloud function permissions
├── migration-scripts/       # Database migration utilities
│   ├── schema-updater.js   # Update existing schemas
│   ├── data-migrator.ts    # Migrate between versions
│   └── backup-system.js    # Automated backups
├── query-patterns/          # Optimized database queries
│   ├── pagination.ts       # Efficient pagination
│   ├── real-time-sync.js   # Real-time data synchronization
│   └── aggregations.ts     # Complex data aggregations
└── integration-helpers/     # Database integration utilities
    ├── firestore-client.ts  # Configured Firestore client
    ├── data-validators.ts   # Input validation
    └── error-handlers.ts    # Database error handling
```

## 🚀 Quick Integration

### 1. Firestore Schema Setup
```typescript
// Copy schema definitions
import { UserSchema, TaskSchema } from './database/firestore-schemas';

// Auto-generate TypeScript interfaces
interface User extends UserSchema {}
interface Task extends TaskSchema {}

// Use with type safety
const createUser = (userData: Partial<User>) => {
  return firestore.collection('users').add(userData);
};
```

### 2. Security Rules Deployment
```bash
# Copy security rules
cp components/database/security-rules/* ./

# Deploy rules
firebase deploy --only firestore:rules,storage:rules
```

### 3. Real-time Data Integration
```typescript
// Real-time updates with event-driven patterns
import { RealtimeSync } from './database/query-patterns/real-time-sync';

const sync = new RealtimeSync({
  collection: 'tasks',
  filters: { userId: currentUser.uid },
  onUpdate: (tasks) => setTasks(tasks),
  onError: (error) => handleError(error)
});

// Automatically syncs data changes
sync.start();
```

## 🏗️ Schema Architecture

### User Management Schema
```typescript
// firestore-schemas/user-management/user.schema.ts
export interface UserProfile {
  id: string;
  email: string;
  displayName: string;
  photoURL?: string;
  role: 'user' | 'admin' | 'moderator';
  permissions: string[];
  createdAt: Timestamp;
  updatedAt: Timestamp;
  preferences: UserPreferences;
  progress: UserProgress;
}

export interface UserProgress {
  level: number;
  xp: number;
  streak: number;
  achievements: Achievement[];
  stats: UserStats;
}
```

### Task Management Schema
```typescript
// firestore-schemas/task-management/task.schema.ts
export interface PersonalTask {
  id: string;
  userId: string;
  title: string;
  description?: string;
  priority: 'urgent' | 'high' | 'medium' | 'low';
  status: 'todo' | 'in-progress' | 'completed' | 'cancelled';
  category: string;
  tags: string[];
  estimatedDuration?: number;
  actualDuration?: number;
  dueDate?: Timestamp;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  subtasks: SubTask[];
}
```

### Real-time Events Schema
```typescript
// firestore-schemas/real-time-updates/events.schema.ts
export interface ProcessingJob {
  id: string;
  userId: string;
  type: 'video-processing' | 'audio-generation' | 'batch-operation';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  input: JobInput;
  output?: JobOutput;
  progress: number;
  error?: string;
  createdAt: Timestamp;
  completedAt?: Timestamp;
}
```

## 🔐 Security Rules

### User Data Protection
```javascript
// security-rules/firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null 
        && request.auth.uid == userId;
    }
    
    // Tasks belong to specific users
    match /tasks/{taskId} {
      allow read, write: if request.auth != null 
        && resource.data.userId == request.auth.uid;
    }
    
    // Processing jobs with user ownership
    match /processingJobs/{jobId} {
      allow read, write: if request.auth != null 
        && resource.data.userId == request.auth.uid;
    }
    
    // Admin-only collections
    match /adminData/{document} {
      allow read, write: if request.auth != null 
        && request.auth.token.role == 'admin';
    }
  }
}
```

### Storage Security
```javascript
// security-rules/storage.rules
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // User uploads in their own folder
    match /user-uploads/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null 
        && request.auth.uid == userId;
    }
    
    // Public assets (read-only)
    match /public/{allPaths=**} {
      allow read;
      allow write: if request.auth != null 
        && request.auth.token.role == 'admin';
    }
  }
}
```

## 🔄 Real-time Data Patterns

### Event-Driven Architecture
```typescript
// query-patterns/real-time-sync.js
export class RealtimeSync {
  constructor(options) {
    this.collection = options.collection;
    this.filters = options.filters;
    this.onUpdate = options.onUpdate;
    this.onError = options.onError;
  }
  
  start() {
    let query = firestore.collection(this.collection);
    
    // Apply filters
    Object.entries(this.filters).forEach(([field, value]) => {
      query = query.where(field, '==', value);
    });
    
    // Listen for real-time updates
    this.unsubscribe = query.onSnapshot(
      (snapshot) => {
        const data = snapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data()
        }));
        this.onUpdate(data);
      },
      (error) => this.onError(error)
    );
  }
  
  stop() {
    if (this.unsubscribe) {
      this.unsubscribe();
    }
  }
}
```

### Background Job Processing
```typescript
// Integration with Firebase Functions for background processing
export const createProcessingJob = async (jobData: ProcessingJobInput) => {
  // Create job document
  const jobRef = await firestore.collection('processingJobs').add({
    ...jobData,
    status: 'pending',
    progress: 0,
    createdAt: FieldValue.serverTimestamp()
  });
  
  // Trigger background processing
  // Firebase function automatically processes jobs with status 'pending'
  
  return jobRef.id;
};

// Real-time job status monitoring
export const monitorJob = (jobId: string, onUpdate: (job: ProcessingJob) => void) => {
  return firestore.collection('processingJobs').doc(jobId)
    .onSnapshot((doc) => {
      if (doc.exists) {
        onUpdate({ id: doc.id, ...doc.data() } as ProcessingJob);
      }
    });
};
```

## 🧪 Testing Integration

### Real Database Testing
```typescript
// integration-helpers/test-utils.ts
export class DatabaseTestUtils {
  static async createTestUser(userData: Partial<UserProfile>) {
    const testUser = {
      email: `test-${Date.now()}@example.com`,
      displayName: 'Test User',
      role: 'user' as const,
      permissions: [],
      createdAt: FieldValue.serverTimestamp(),
      ...userData
    };
    
    return await firestore.collection('users').add(testUser);
  }
  
  static async cleanupTestData(userId: string) {
    // Delete user and all associated data
    const batch = firestore.batch();
    
    // Delete user tasks
    const tasks = await firestore.collection('tasks')
      .where('userId', '==', userId).get();
    
    tasks.docs.forEach(doc => batch.delete(doc.ref));
    
    // Delete user document
    batch.delete(firestore.collection('users').doc(userId));
    
    await batch.commit();
  }
}
```

### AI Workflow Testing
```typescript
// Test with real Firestore operations
describe('Task Management', () => {
  test('create and retrieve tasks with real database', async () => {
    const testUser = await DatabaseTestUtils.createTestUser({
      displayName: 'AI Test User'
    });
    
    const taskData = {
      userId: testUser.id,
      title: 'Test Task',
      priority: 'high' as const,
      status: 'todo' as const,
      createdAt: FieldValue.serverTimestamp()
    };
    
    const taskRef = await firestore.collection('tasks').add(taskData);
    const retrievedTask = await taskRef.get();
    
    expect(retrievedTask.exists).toBe(true);
    expect(retrievedTask.data()?.title).toBe('Test Task');
    
    // Cleanup
    await DatabaseTestUtils.cleanupTestData(testUser.id);
  });
});
```

## 📊 Performance Optimization

### Query Optimization
```typescript
// query-patterns/pagination.ts
export class PaginatedQuery {
  constructor(collection: string, pageSize: number = 25) {
    this.collection = collection;
    this.pageSize = pageSize;
    this.lastVisible = null;
  }
  
  async getPage(filters = {}) {
    let query = firestore.collection(this.collection)
      .limit(this.pageSize);
    
    // Apply filters
    Object.entries(filters).forEach(([field, value]) => {
      query = query.where(field, '==', value);
    });
    
    // Add cursor for pagination
    if (this.lastVisible) {
      query = query.startAfter(this.lastVisible);
    }
    
    const snapshot = await query.get();
    this.lastVisible = snapshot.docs[snapshot.docs.length - 1];
    
    return snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
  }
}
```

### Index Management
```json
// firestore.indexes.json - Auto-generated composite indexes
{
  "indexes": [
    {
      "collectionGroup": "tasks",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "userId", "order": "ASCENDING" },
        { "fieldPath": "priority", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    },
    {
      "collectionGroup": "processingJobs", 
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "userId", "order": "ASCENDING" },
        { "fieldPath": "status", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    }
  ]
}
```

## 🔄 Integration with Other Components

### Authentication Integration
```typescript
// Automatic user profile creation on auth
export const createUserProfile = async (authUser: User) => {
  const userProfile: UserProfile = {
    id: authUser.uid,
    email: authUser.email!,
    displayName: authUser.displayName || 'New User',
    role: 'user',
    permissions: ['read:own-data', 'write:own-data'],
    createdAt: FieldValue.serverTimestamp(),
    preferences: defaultPreferences,
    progress: defaultProgress
  };
  
  await firestore.collection('users').doc(authUser.uid).set(userProfile);
};
```

### AI Workflow Integration
```typescript
// Database operations available to AI agents
export const DatabaseOperations = {
  createTask: (userId: string, taskData: Partial<PersonalTask>) => {
    return firestore.collection('tasks').add({
      userId,
      createdAt: FieldValue.serverTimestamp(),
      ...taskData
    });
  },
  
  updateTaskStatus: (taskId: string, status: TaskStatus) => {
    return firestore.collection('tasks').doc(taskId).update({
      status,
      updatedAt: FieldValue.serverTimestamp()
    });
  },
  
  getUserTasks: (userId: string, filters = {}) => {
    let query = firestore.collection('tasks').where('userId', '==', userId);
    
    Object.entries(filters).forEach(([field, value]) => {
      query = query.where(field, '==', value);
    });
    
    return query.get();
  }
};
```

This database system provides the foundation for scalable, secure, and real-time applications with comprehensive data management capabilities.