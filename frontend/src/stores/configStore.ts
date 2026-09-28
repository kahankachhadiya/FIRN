import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Config } from '@/services/types';
import { mockConfig } from '@/services/mockData';

interface ConfigState {
  config: Config;
  isLoaded: boolean;
  setConfig: (config: Partial<Config>) => void;
  resetConfig: () => void;
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      config: mockConfig,
      isLoaded: false,
      setConfig: (updates) =>
        set((state) => ({
          config: { ...state.config, ...updates },
          isLoaded: true,
        })),
      resetConfig: () => set({ config: mockConfig }),
    }),
    {
      name: 'nemoire-config',
    }
  )
);
