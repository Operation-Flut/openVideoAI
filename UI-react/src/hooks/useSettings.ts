import { Settings, defaultSettings } from '../types/settings';
import { settingsAPI } from '../services/api';

export const useSettings = () => {
  const loadSettings = async () => {
    try {
      const response = await settingsAPI.getSettings();
      return { ...defaultSettings, ...response };
    } catch (error) {
      console.error('Failed to load settings:', error);
      return defaultSettings;
    }
  };
  
  const saveSettings = async (settings: Partial<Settings>) => {
    try {
      const response = await settingsAPI.updateSettings(settings);
      return response;
    } catch (error) {
      console.error('Failed to save settings:', error);
      throw error;
    }
  };
  
  return {
    loadSettings,
    saveSettings,
    defaultSettings,
  };
};