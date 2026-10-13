import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface ProgressState {
  progress: number;
  currentStage: string;
  status: 'idle' | 'processing' | 'completed' | 'error';
  error: string | null;
  taskId: string | null;
}

const initialState: ProgressState = {
  progress: 0,
  currentStage: '',
  status: 'idle',
  error: null,
  taskId: null,
};

const progressSlice = createSlice({
  name: 'progress',
  initialState,
  reducers: {
    setProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
    setCurrentStage: (state, action: PayloadAction<string>) => {
      state.currentStage = action.payload;
    },
    setStatus: (state, action: PayloadAction<'idle' | 'processing' | 'completed' | 'error'>) => {
      state.status = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setTaskId: (state, action: PayloadAction<string | null>) => {
      state.taskId = action.payload;
    },
    resetProgress: (state) => {
      state.progress = 0;
      state.currentStage = '';
      state.status = 'idle';
      state.error = null;
      state.taskId = null;
    },
  },
});

export const {
  setProgress,
  setCurrentStage,
  setStatus,
  setError,
  setTaskId,
  resetProgress,
} = progressSlice.actions;

export default progressSlice.reducer;