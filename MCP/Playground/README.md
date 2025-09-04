# 🎮 MCP Playground - Interactive Learning Environment

**Hands-on Examples and Interactive Tutorials for Model Context Protocol**

## 🚀 **Quick Start Playground**

### **1. Hello World MCP Server**
```typescript
// playground/hello-world/server.ts
import { Server } from '@modelcontextprotocol/server';

const server = new Server({
  name: 'hello-world-mcp',
  version: '1.0.0'
});

server.addTool({
  name: 'greet',
  description: 'Greets a person with their name',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'The person\'s name' }
    },
    required: ['name']
  },
  handler: async ({ name }) => {
    return {
      content: [
        {
          type: 'text',
          text: `Hello, ${name}! Welcome to the MCP Playground! 🎉`
        }
      ]
    };
  }
});

export default server;
```

**Try it now**:
```bash
cd playground/hello-world
npm install
npm start
```

### **2. Calculator MCP - Interactive Math Operations**
```typescript
// playground/calculator/server.ts
import { Server } from '@modelcontextprotocol/server';

const calculator = new Server({
  name: 'calculator-mcp',
  version: '1.0.0'
});

calculator.addTool({
  name: 'calculate',
  description: 'Perform mathematical operations',
  inputSchema: {
    type: 'object',
    properties: {
      operation: {
        type: 'string',
        enum: ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'],
        description: 'The operation to perform'
      },
      a: { type: 'number', description: 'First number' },
      b: { type: 'number', description: 'Second number (optional for sqrt)' }
    },
    required: ['operation', 'a']
  },
  handler: async ({ operation, a, b }) => {
    let result;
    
    switch (operation) {
      case 'add': result = a + (b || 0); break;
      case 'subtract': result = a - (b || 0); break;
      case 'multiply': result = a * (b || 1); break;
      case 'divide': 
        if (b === 0) throw new Error('Division by zero!');
        result = a / b; 
        break;
      case 'power': result = Math.pow(a, b || 2); break;
      case 'sqrt': result = Math.sqrt(a); break;
      default: throw new Error('Unknown operation');
    }
    
    return {
      content: [
        {
          type: 'text',
          text: `Result: ${result}`
        }
      ]
    };
  }
});

export default calculator;
```

### **3. File System Explorer MCP**
```typescript
// playground/file-explorer/server.ts
import { Server } from '@modelcontextprotocol/server';
import { promises as fs } from 'fs';
import path from 'path';

const fileExplorer = new Server({
  name: 'file-explorer-mcp',
  version: '1.0.0'
});

fileExplorer.addTool({
  name: 'list_directory',
  description: 'List files and directories in a given path',
  inputSchema: {
    type: 'object',
    properties: {
      path: { 
        type: 'string', 
        description: 'Directory path to list',
        default: '.'
      }
    }
  },
  handler: async ({ path: dirPath = '.' }) => {
    try {
      const items = await fs.readdir(dirPath, { withFileTypes: true });
      const result = items.map(item => ({
        name: item.name,
        type: item.isDirectory() ? 'directory' : 'file',
        path: path.join(dirPath, item.name)
      }));
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2)
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to list directory: ${error.message}`);
    }
  }
});

fileExplorer.addTool({
  name: 'read_file',
  description: 'Read the contents of a file',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'File path to read' }
    },
    required: ['path']
  },
  handler: async ({ path: filePath }) => {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return {
        content: [
          {
            type: 'text',
            text: content
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to read file: ${error.message}`);
    }
  }
});

export default fileExplorer;
```

## 🎯 **Interactive Examples**

### **Database Connection Playground**
```javascript
// playground/database/interactive-example.js
const { DatabaseMCP } = require('./database-mcp');

// Example: Connect to different databases
const examples = {
  postgresql: {
    host: 'localhost',
    port: 5432,
    database: 'playground',
    username: 'user',
    password: 'password'
  },
  
  sqlite: {
    filename: './playground.db'
  },
  
  mongodb: {
    url: 'mongodb://localhost:27017/playground'
  }
};

// Interactive query builder
const queryBuilder = {
  select: (table, columns = '*') => `SELECT ${columns} FROM ${table}`,
  insert: (table, data) => {
    const keys = Object.keys(data).join(', ');
    const values = Object.values(data).map(v => `'${v}'`).join(', ');
    return `INSERT INTO ${table} (${keys}) VALUES (${values})`;
  },
  update: (table, data, where) => {
    const sets = Object.entries(data)
      .map(([k, v]) => `${k} = '${v}'`)
      .join(', ');
    return `UPDATE ${table} SET ${sets} WHERE ${where}`;
  }
};

// Try it out:
console.log(queryBuilder.select('users', 'name, email'));
console.log(queryBuilder.insert('users', { name: 'John', email: 'john@example.com' }));
```

### **Real-time Chat MCP**
```typescript
// playground/chat/real-time-chat.ts
import { Server } from '@modelcontextprotocol/server';
import { EventEmitter } from 'events';

class ChatRoom extends EventEmitter {
  private messages: Array<{ user: string; message: string; timestamp: Date }> = [];
  
  addMessage(user: string, message: string) {
    const chatMessage = { user, message, timestamp: new Date() };
    this.messages.push(chatMessage);
    this.emit('message', chatMessage);
    return chatMessage;
  }
  
  getMessages(limit = 10) {
    return this.messages.slice(-limit);
  }
}

const chatRoom = new ChatRoom();
const chatServer = new Server({
  name: 'chat-mcp',
  version: '1.0.0'
});

chatServer.addTool({
  name: 'send_message',
  description: 'Send a message to the chat room',
  inputSchema: {
    type: 'object',
    properties: {
      user: { type: 'string', description: 'Username' },
      message: { type: 'string', description: 'Message content' }
    },
    required: ['user', 'message']
  },
  handler: async ({ user, message }) => {
    const chatMessage = chatRoom.addMessage(user, message);
    
    return {
      content: [
        {
          type: 'text',
          text: `Message sent! ${user}: ${message}`
        }
      ]
    };
  }
});

chatServer.addTool({
  name: 'get_recent_messages',
  description: 'Get recent chat messages',
  inputSchema: {
    type: 'object',
    properties: {
      limit: { type: 'number', description: 'Number of messages to retrieve', default: 10 }
    }
  },
  handler: async ({ limit = 10 }) => {
    const messages = chatRoom.getMessages(limit);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(messages, null, 2)
        }
      ]
    };
  }
});
```

## 🛠️ **Development Challenges**

### **Challenge 1: Weather API MCP**
```typescript
// playground/challenges/weather-api.ts
/*
CHALLENGE: Build a Weather MCP Server
Requirements:
1. Connect to a weather API (OpenWeatherMap, WeatherAPI, etc.)
2. Implement current weather lookup by city
3. Add 5-day forecast functionality
4. Include weather alerts if available
5. Handle errors gracefully

Bonus Points:
- Cache responses for 10 minutes
- Support multiple units (metric, imperial)
- Add weather-based recommendations

Your solution here:
*/
```

### **Challenge 2: Task Management MCP**
```typescript
// playground/challenges/task-manager.ts
/*
CHALLENGE: Build a Task Management MCP Server
Requirements:
1. Create, read, update, delete tasks
2. Task priorities and due dates
3. Task categories/tags
4. Search and filter functionality
5. Persistent storage (JSON file or database)

Bonus Points:
- Task dependencies
- Recurring tasks
- Time tracking
- Export to different formats

Your solution here:
*/
```

### **Challenge 3: AI Image Generator MCP**
```typescript
// playground/challenges/image-generator.ts
/*
CHALLENGE: Build an AI Image Generator MCP
Requirements:
1. Connect to image generation API (DALL-E, Midjourney, Stable Diffusion)
2. Support different image sizes and styles
3. Image history and management
4. Prompt optimization suggestions
5. Error handling for failed generations

Bonus Points:
- Image editing capabilities
- Style presets
- Batch generation
- Local storage management

Your solution here:
*/
```

## 🎓 **Learning Path**

### **Beginner (Week 1-2)**
1. **Start Here**: Hello World MCP Server
2. **Build**: Calculator MCP with basic operations
3. **Learn**: File System Explorer for understanding I/O
4. **Practice**: Complete Weather API Challenge

### **Intermediate (Week 3-4)**
1. **Database Integration**: Connect to PostgreSQL/MongoDB
2. **Real-time Features**: Chat Room MCP with WebSocket
3. **API Integration**: External service connections
4. **Practice**: Complete Task Management Challenge

### **Advanced (Week 5-6)**
1. **Performance Optimization**: Caching and connection pooling
2. **Security**: Authentication and authorization
3. **Scalability**: Load balancing and clustering
4. **Practice**: Complete AI Image Generator Challenge

## 🔧 **Playground Tools**

### **MCP Development Kit**
```bash
# Install the playground development kit
npm install -g @mcp-playground/dev-kit

# Create new MCP server
mcp-create my-server --template=basic

# Test server locally
mcp-test my-server --port=3000

# Deploy to playground environment
mcp-deploy my-server --environment=playground
```

### **Interactive Testing Console**
```javascript
// playground/testing/console.js
const { MCPClient } = require('@modelcontextprotocol/client');

class PlaygroundConsole {
  constructor() {
    this.clients = new Map();
  }
  
  async connect(serverName, config) {
    const client = new MCPClient(config);
    await client.connect();
    this.clients.set(serverName, client);
    console.log(`Connected to ${serverName} ✅`);
  }
  
  async call(serverName, tool, params) {
    const client = this.clients.get(serverName);
    if (!client) {
      throw new Error(`Server ${serverName} not connected`);
    }
    
    const result = await client.callTool(tool, params);
    console.log(`Result from ${serverName}.${tool}:`, result);
    return result;
  }
  
  list() {
    console.log('Connected servers:', Array.from(this.clients.keys()));
  }
}

// Usage example:
// const console = new PlaygroundConsole();
// await console.connect('calculator', { port: 3000 });
// await console.call('calculator', 'calculate', { operation: 'add', a: 5, b: 3 });
```

## 📚 **Resources**

### **Documentation Links**
- [MCP Protocol Specification](https://modelcontextprotocol.io/docs/specification)
- [Server Development Guide](https://modelcontextprotocol.io/docs/server-development)
- [Client Integration Guide](https://modelcontextprotocol.io/docs/client-integration)

### **Community Examples**
- [GitHub: MCP Examples Repository](https://github.com/modelcontextprotocol/examples)
- [Discord: MCP Developer Community](https://discord.gg/mcp-developers)
- [Reddit: r/ModelContextProtocol](https://reddit.com/r/ModelContextProtocol)

### **Video Tutorials**
- "MCP From Zero to Hero" - 12-part series
- "Building Production MCP Servers" - Advanced course
- "MCP Performance Optimization" - Technical deep dive

## 🎯 **Next Steps**

1. **Choose Your Level**: Start with beginner, intermediate, or advanced
2. **Pick an Example**: Try one of the interactive examples above
3. **Complete Challenges**: Work through the development challenges
4. **Build Something**: Create your own MCP server
5. **Share**: Contribute your examples back to the playground

---

**Interactive Environment**: All code examples are runnable in the MCP Playground  
**Live Updates**: New examples added weekly based on community requests  
**Support**: Community Discord for help and collaboration