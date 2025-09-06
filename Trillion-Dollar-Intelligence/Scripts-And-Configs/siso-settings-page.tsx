import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Simplified UI Components for SISO IDE (extracted from Claudia)
const Card = ({ children, className = '' }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children }) => (
  <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
    {children}
  </div>
);

const CardTitle = ({ children }) => (
  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
    {children}
  </h3>
);

const CardContent = ({ children }) => (
  <div className="px-6 py-4">
    {children}
  </div>
);

const Button = ({ children, onClick, variant = 'default', size = 'md', className = '' }) => {
  const baseClasses = "inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2";
  const variants = {
    default: "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500",
    outline: "border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:ring-blue-500",
    ghost: "text-gray-700 hover:bg-gray-100 focus:ring-gray-500"
  };
  const sizes = {
    sm: "px-3 py-2 text-sm",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base"
  };
  
  return (
    <button
      onClick={onClick}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
};

const Input = ({ value, onChange, placeholder, type = 'text', className = '' }) => (
  <input
    type={type}
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    className={`w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${className}`}
  />
);

const Switch = ({ checked, onCheckedChange, className = '' }) => (
  <button
    role="switch"
    aria-checked={checked}
    onClick={() => onCheckedChange(!checked)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
      checked ? 'bg-blue-600' : 'bg-gray-300'
    } ${className}`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
        checked ? 'translate-x-6' : 'translate-x-1'
      }`}
    />
  </button>
);

const Tabs = ({ value, onValueChange, children }) => (
  <div className="w-full">
    {React.Children.map(children, child =>
      React.cloneElement(child, { value, onValueChange })
    )}
  </div>
);

const TabsList = ({ children, value, onValueChange }) => (
  <div className="inline-flex h-10 items-center justify-center rounded-md bg-gray-100 p-1 dark:bg-gray-800">
    {React.Children.map(children, child =>
      React.cloneElement(child, { value, onValueChange })
    )}
  </div>
);

const TabsTrigger = ({ value: tabValue, children, value, onValueChange }) => (
  <button
    onClick={() => onValueChange(tabValue)}
    className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 ${
      value === tabValue
        ? 'bg-white text-gray-950 shadow-sm dark:bg-gray-950 dark:text-gray-50'
        : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50'
    }`}
  >
    {children}
  </button>
);

const TabsContent = ({ value: tabValue, children, value }) => {
  if (value !== tabValue) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
      className="mt-6 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
    >
      {children}
    </motion.div>
  );
};

const Label = ({ children, className = '' }) => (
  <label className={`text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-gray-900 dark:text-gray-100 ${className}`}>
    {children}
  </label>
);

const Textarea = ({ value, onChange, placeholder, rows = 3, className = '' }) => (
  <textarea
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    rows={rows}
    className={`w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white resize-none ${className}`}
  />
);

// SISO IDE Settings Component
export default function SISOSettingsPage() {
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState({
    general: {
      projectName: 'SISO IDE',
      autoSave: true,
      theme: 'dark',
      language: 'en'
    },
    editor: {
      fontSize: 14,
      tabSize: 2,
      wordWrap: true,
      showMinimap: true,
      autoComplete: true
    },
    ai: {
      provider: 'claude',
      model: 'claude-3.5-sonnet',
      maxTokens: 4096,
      temperature: 0.7,
      systemPrompt: 'You are SISO, an intelligent coding assistant.'
    },
    integrations: {
      github: {
        enabled: false,
        token: '',
        defaultRepo: ''
      },
      notion: {
        enabled: false,
        token: '',
        database: ''
      }
    }
  });

  const updateSetting = (section, key, value) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        key === 'nested' ? { ...prev[section], ...value } : { ...prev[section], [key]: value }
      }
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              SISO IDE Settings
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Configure your SISO IDE experience
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="general">General</TabsTrigger>
              <TabsTrigger value="editor">Editor</TabsTrigger>
              <TabsTrigger value="ai">AI Assistant</TabsTrigger>
              <TabsTrigger value="integrations">Integrations</TabsTrigger>
            </TabsList>

            <AnimatePresence mode="wait">
              <TabsContent value="general">
                <Card>
                  <CardHeader>
                    <CardTitle>General Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label>Project Name</Label>
                        <Input
                          value={settings.general.projectName}
                          onChange={(e) => updateSetting('general', 'projectName', e.target.value)}
                          placeholder="Enter project name"
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Theme</Label>
                        <select
                          value={settings.general.theme}
                          onChange={(e) => updateSetting('general', 'theme', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                        >
                          <option value="light">Light</option>
                          <option value="dark">Dark</option>
                          <option value="auto">Auto</option>
                        </select>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto Save</Label>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Automatically save changes as you type
                        </p>
                      </div>
                      <Switch
                        checked={settings.general.autoSave}
                        onCheckedChange={(checked) => updateSetting('general', 'autoSave', checked)}
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="editor">
                <Card>
                  <CardHeader>
                    <CardTitle>Editor Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label>Font Size</Label>
                        <Input
                          type="number"
                          value={settings.editor.fontSize}
                          onChange={(e) => updateSetting('editor', 'fontSize', parseInt(e.target.value))}
                          min="10"
                          max="24"
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Tab Size</Label>
                        <Input
                          type="number"
                          value={settings.editor.tabSize}
                          onChange={(e) => updateSetting('editor', 'tabSize', parseInt(e.target.value))}
                          min="2"
                          max="8"
                        />
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Word Wrap</Label>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Wrap long lines of code
                          </p>
                        </div>
                        <Switch
                          checked={settings.editor.wordWrap}
                          onCheckedChange={(checked) => updateSetting('editor', 'wordWrap', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Show Minimap</Label>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Display code minimap on the right
                          </p>
                        </div>
                        <Switch
                          checked={settings.editor.showMinimap}
                          onCheckedChange={(checked) => updateSetting('editor', 'showMinimap', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Auto Complete</Label>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Enable intelligent code completion
                          </p>
                        </div>
                        <Switch
                          checked={settings.editor.autoComplete}
                          onCheckedChange={(checked) => updateSetting('editor', 'autoComplete', checked)}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="ai">
                <Card>
                  <CardHeader>
                    <CardTitle>AI Assistant Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label>AI Provider</Label>
                        <select
                          value={settings.ai.provider}
                          onChange={(e) => updateSetting('ai', 'provider', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                        >
                          <option value="claude">Claude</option>
                          <option value="openai">OpenAI</option>
                          <option value="local">Local Model</option>
                        </select>
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Model</Label>
                        <select
                          value={settings.ai.model}
                          onChange={(e) => updateSetting('ai', 'model', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                        >
                          <option value="claude-3.5-sonnet">Claude 3.5 Sonnet</option>
                          <option value="claude-3-haiku">Claude 3 Haiku</option>
                          <option value="gpt-4">GPT-4</option>
                          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label>Max Tokens</Label>
                        <Input
                          type="number"
                          value={settings.ai.maxTokens}
                          onChange={(e) => updateSetting('ai', 'maxTokens', parseInt(e.target.value))}
                          min="1024"
                          max="8192"
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Temperature</Label>
                        <Input
                          type="number"
                          step="0.1"
                          value={settings.ai.temperature}
                          onChange={(e) => updateSetting('ai', 'temperature', parseFloat(e.target.value))}
                          min="0"
                          max="2"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>System Prompt</Label>
                      <Textarea
                        value={settings.ai.systemPrompt}
                        onChange={(e) => updateSetting('ai', 'systemPrompt', e.target.value)}
                        placeholder="Enter system prompt for the AI assistant"
                        rows={4}
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="integrations">
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>GitHub Integration</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Enable GitHub</Label>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Connect to GitHub for repository management
                          </p>
                        </div>
                        <Switch
                          checked={settings.integrations.github.enabled}
                          onCheckedChange={(checked) => updateSetting('integrations', 'nested', { github: { ...settings.integrations.github, enabled: checked } })}
                        />
                      </div>

                      {settings.integrations.github.enabled && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-4"
                        >
                          <div className="space-y-2">
                            <Label>GitHub Token</Label>
                            <Input
                              type="password"
                              value={settings.integrations.github.token}
                              onChange={(e) => updateSetting('integrations', 'nested', { github: { ...settings.integrations.github, token: e.target.value } })}
                              placeholder="ghp_..."
                            />
                          </div>
                          
                          <div className="space-y-2">
                            <Label>Default Repository</Label>
                            <Input
                              value={settings.integrations.github.defaultRepo}
                              onChange={(e) => updateSetting('integrations', 'nested', { github: { ...settings.integrations.github, defaultRepo: e.target.value } })}
                              placeholder="username/repository"
                            />
                          </div>
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Notion Integration</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Enable Notion</Label>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Connect to Notion for documentation
                          </p>
                        </div>
                        <Switch
                          checked={settings.integrations.notion.enabled}
                          onCheckedChange={(checked) => updateSetting('integrations', 'nested', { notion: { ...settings.integrations.notion, enabled: checked } })}
                        />
                      </div>

                      {settings.integrations.notion.enabled && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-4"
                        >
                          <div className="space-y-2">
                            <Label>Notion Token</Label>
                            <Input
                              type="password"
                              value={settings.integrations.notion.token}
                              onChange={(e) => updateSetting('integrations', 'nested', { notion: { ...settings.integrations.notion, token: e.target.value } })}
                              placeholder="secret_..."
                            />
                          </div>
                          
                          <div className="space-y-2">
                            <Label>Database ID</Label>
                            <Input
                              value={settings.integrations.notion.database}
                              onChange={(e) => updateSetting('integrations', 'nested', { notion: { ...settings.integrations.notion, database: e.target.value } })}
                              placeholder="Database ID"
                            />
                          </div>
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </AnimatePresence>
          </Tabs>

          <div className="flex justify-end mt-8 space-x-4">
            <Button variant="outline">
              Reset to Defaults
            </Button>
            <Button onClick={() => console.log('Settings saved:', settings)}>
              Save Settings
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}