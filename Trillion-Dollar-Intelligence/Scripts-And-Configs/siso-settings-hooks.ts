// SISO IDE Settings Hooks and API Integration
// Based on Claudia GUI patterns, adapted for SISO IDE

import { useState, useEffect, useCallback } from 'react';

// Types for SISO Settings
export interface SISOSettings {
  general: {
    projectName: string;
    autoSave: boolean;
    theme: 'light' | 'dark' | 'auto';
    language: string;
  };
  editor: {
    fontSize: number;
    tabSize: number;
    wordWrap: boolean;
    showMinimap: boolean;
    autoComplete: boolean;
  };
  ai: {
    provider: 'claude' | 'openai' | 'local';
    model: string;
    maxTokens: number;
    temperature: number;
    systemPrompt: string;
  };
  integrations: {
    github: {
      enabled: boolean;
      token: string;
      defaultRepo: string;
    };
    notion: {
      enabled: boolean;
      token: string;
      database: string;
    };
  };
}

// Default settings
export const defaultSettings: SISOSettings = {
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
};

// Local Storage Keys
const STORAGE_KEYS = {
  SETTINGS: 'siso-ide-settings',
  THEME: 'siso-ide-theme',
  CACHE: 'siso-ide-cache'
};

// API Endpoints (adjust based on your backend)
const API_ENDPOINTS = {
  SETTINGS: '/api/settings',
  VALIDATE_GITHUB: '/api/integrations/github/validate',
  VALIDATE_NOTION: '/api/integrations/notion/validate',
  AI_MODELS: '/api/ai/models'
};

// Settings Storage Utility
class SettingsStorage {
  static save(settings: SISOSettings): void {
    try {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (error) {
      console.warn('Failed to save settings to localStorage:', error);
    }
  }

  static load(): SISOSettings | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.warn('Failed to load settings from localStorage:', error);
      return null;
    }
  }

  static clear(): void {
    localStorage.removeItem(STORAGE_KEYS.SETTINGS);
  }
}

// Main Settings Hook
export function useSettings() {
  const [settings, setSettings] = useState<SISOSettings>(defaultSettings);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    setError(null);

    try {
      // Try to load from localStorage first
      const localSettings = SettingsStorage.load();
      if (localSettings) {
        setSettings(mergeSettings(defaultSettings, localSettings));
      }

      // Then try to sync with server
      const response = await fetch(API_ENDPOINTS.SETTINGS);
      if (response.ok) {
        const serverSettings = await response.json();
        const mergedSettings = mergeSettings(defaultSettings, serverSettings);
        setSettings(mergedSettings);
        SettingsStorage.save(mergedSettings);
      }
    } catch (err) {
      console.warn('Failed to load settings from server, using local cache');
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async (newSettings: SISOSettings) => {
    setSaving(true);
    setError(null);

    try {
      // Save locally immediately for responsiveness
      setSettings(newSettings);
      SettingsStorage.save(newSettings);

      // Then sync with server
      const response = await fetch(API_ENDPOINTS.SETTINGS, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings)
      });

      if (!response.ok) {
        throw new Error('Failed to save settings to server');
      }
    } catch (err) {
      console.warn('Failed to save settings to server, saved locally only');
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const updateSetting = useCallback((section: keyof SISOSettings, key: string, value: any) => {
    setSettings(prev => {
      const newSettings = {
        ...prev,
        [section]: {
          ...prev[section],
          [key]: value
        }
      };
      
      // Auto-save after a delay
      setTimeout(() => saveSettings(newSettings), 1000);
      
      return newSettings;
    });
  }, []);

  const resetSettings = useCallback(() => {
    const resetSettings = { ...defaultSettings };
    setSettings(resetSettings);
    saveSettings(resetSettings);
  }, []);

  return {
    settings,
    loading,
    saving,
    error,
    updateSetting,
    saveSettings,
    resetSettings,
    loadSettings
  };
}

// Theme Hook
export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark' | 'auto'>('dark');

  useEffect(() => {
    // Load saved theme
    const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME) as 'light' | 'dark' | 'auto';
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    // Apply theme to document
    const root = document.documentElement;
    
    if (theme === 'auto') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const applyTheme = (e: MediaQueryListEvent | MediaQueryList) => {
        root.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      };
      
      applyTheme(mediaQuery);
      mediaQuery.addEventListener('change', applyTheme);
      
      return () => mediaQuery.removeEventListener('change', applyTheme);
    } else {
      root.setAttribute('data-theme', theme);
    }
  }, [theme]);

  const setAndSaveTheme = (newTheme: 'light' | 'dark' | 'auto') => {
    setTheme(newTheme);
    localStorage.setItem(STORAGE_KEYS.THEME, newTheme);
  };

  return { theme, setTheme: setAndSaveTheme };
}

// Integration Validation Hook
export function useIntegrationValidation() {
  const [validating, setValidating] = useState<Record<string, boolean>>({});
  const [validationResults, setValidationResults] = useState<Record<string, boolean>>({});

  const validateGitHub = async (token: string) => {
    setValidating(prev => ({ ...prev, github: true }));
    
    try {
      const response = await fetch(API_ENDPOINTS.VALIDATE_GITHUB, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token })
      });
      
      const isValid = response.ok;
      setValidationResults(prev => ({ ...prev, github: isValid }));
      return isValid;
    } catch (error) {
      setValidationResults(prev => ({ ...prev, github: false }));
      return false;
    } finally {
      setValidating(prev => ({ ...prev, github: false }));
    }
  };

  const validateNotion = async (token: string, database?: string) => {
    setValidating(prev => ({ ...prev, notion: true }));
    
    try {
      const response = await fetch(API_ENDPOINTS.VALIDATE_NOTION, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, database })
      });
      
      const isValid = response.ok;
      setValidationResults(prev => ({ ...prev, notion: isValid }));
      return isValid;
    } catch (error) {
      setValidationResults(prev => ({ ...prev, notion: false }));
      return false;
    } finally {
      setValidating(prev => ({ ...prev, notion: false }));
    }
  };

  return {
    validating,
    validationResults,
    validateGitHub,
    validateNotion
  };
}

// AI Models Hook
export function useAIModels() {
  const [models, setModels] = useState<Record<string, string[]>>({
    claude: ['claude-3.5-sonnet', 'claude-3-haiku'],
    openai: ['gpt-4', 'gpt-3.5-turbo'],
    local: ['llama2', 'codellama']
  });
  const [loading, setLoading] = useState(false);

  const loadModels = async () => {
    setLoading(true);
    
    try {
      const response = await fetch(API_ENDPOINTS.AI_MODELS);
      if (response.ok) {
        const data = await response.json();
        setModels(data);
      }
    } catch (error) {
      console.warn('Failed to load AI models, using defaults');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  return { models, loading, refreshModels: loadModels };
}

// Auto-save Hook
export function useAutoSave<T>(
  value: T,
  saveFunction: (value: T) => Promise<void>,
  delay: number = 1000
) {
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setSaving(true);
    const timeoutId = setTimeout(async () => {
      try {
        await saveFunction(value);
      } catch (error) {
        console.warn('Auto-save failed:', error);
      } finally {
        setSaving(false);
      }
    }, delay);

    return () => clearTimeout(timeoutId);
  }, [value, saveFunction, delay]);

  return { saving };
}

// Export/Import Hook
export function useSettingsImportExport() {
  const exportSettings = (settings: SISOSettings) => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = 'siso-settings.json';
    link.click();
  };

  const importSettings = (file: File): Promise<SISOSettings> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const settings = JSON.parse(e.target?.result as string);
          const validatedSettings = mergeSettings(defaultSettings, settings);
          resolve(validatedSettings);
        } catch (error) {
          reject(new Error('Invalid settings file'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  return { exportSettings, importSettings };
}

// Utility Functions
function mergeSettings(defaults: SISOSettings, override: Partial<SISOSettings>): SISOSettings {
  return {
    general: { ...defaults.general, ...override.general },
    editor: { ...defaults.editor, ...override.editor },
    ai: { ...defaults.ai, ...override.ai },
    integrations: {
      github: { ...defaults.integrations.github, ...override.integrations?.github },
      notion: { ...defaults.integrations.notion, ...override.integrations?.notion }
    }
  };
}

// Validation utilities
export const validators = {
  githubToken: (token: string) => /^ghp_[a-zA-Z0-9_]{36}$/.test(token),
  notionToken: (token: string) => /^secret_[a-zA-Z0-9_]{43}$/.test(token),
  temperature: (value: number) => value >= 0 && value <= 2,
  maxTokens: (value: number) => value >= 1 && value <= 32000,
  fontSize: (value: number) => value >= 10 && value <= 32,
  tabSize: (value: number) => value >= 1 && value <= 8
};

// Settings change tracking
export function useSettingsChangeTracking(settings: SISOSettings) {
  const [originalSettings, setOriginalSettings] = useState<SISOSettings>(settings);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    const changed = JSON.stringify(settings) !== JSON.stringify(originalSettings);
    setHasChanges(changed);
  }, [settings, originalSettings]);

  const markAsSaved = () => {
    setOriginalSettings(settings);
    setHasChanges(false);
  };

  const discardChanges = () => {
    setHasChanges(false);
    return originalSettings;
  };

  return { hasChanges, markAsSaved, discardChanges };
}