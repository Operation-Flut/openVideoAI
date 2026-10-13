import { configureStore } from '@reduxjs/toolkit';
import pipelineReducer from './pipelineStore';
import progressReducer from './progressStore';

export const store = configureStore({
  reducer: {
    pipeline: pipelineReducer,
    progress: progressReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;