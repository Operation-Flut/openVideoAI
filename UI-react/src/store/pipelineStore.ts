import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface PipelineStep {
  id: string;
  name: string;
  description: string;
  icon?: string;
  config: Record<string, any>;
}

interface PipelineState {
  steps: PipelineStep[];
  models: Record<string, any>;
  loading: boolean;
  error: string | null;
}

const initialState: PipelineState = {
  steps: [],
  models: {},
  loading: false,
  error: null,
};

const pipelineSlice = createSlice({
  name: 'pipeline',
  initialState,
  reducers: {
    addPipelineStep: (state, action: PayloadAction<PipelineStep>) => {
      const newStep = action.payload;
      if (!state.steps.find(s => s.id === newStep.id)) {
        state.steps.push(newStep);
      }
    },
    removePipelineStep: (state, action: PayloadAction<number>) => {
      state.steps.splice(action.payload, 1);
    },
    updatePipelineStepConfig: (state, action: PayloadAction<{ index: number; config: Record<string, any> }>) => {
      const { index, config } = action.payload;
      if (state.steps[index]) {
        state.steps[index].config = { ...state.steps[index].config, ...config };
      }
    },
    clearPipeline: (state) => {
      state.steps = [];
    },
    setModels: (state, action: PayloadAction<Record<string, any>>) => {
      state.models = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  addPipelineStep,
  removePipelineStep,
  updatePipelineStepConfig,
  clearPipeline,
  setModels,
  setLoading,
  setError,
} = pipelineSlice.actions;

export default pipelineSlice.reducer;