# openVideoAI React Frontend

Modern React + TypeScript frontend for openVideoAI with pipeline builder, model management, and real-time progress tracking.

## Features

- **Pipeline Builder**: Drag-and-drop visual pipeline construction
- **Model Selector**: Searchable model picker with filtering
- **Progress Monitor**: Real-time progress tracking with SSE
- **File Drop Zone**: Drag-and-drop file upload with preview
- **Settings Panel**: Comprehensive configuration management
- **Real-time Updates**: Server-Sent Events for live progress

## Tech Stack

- React 18 + TypeScript
- Vite for fast development
- Redux Toolkit for state management
- React Query for server state
- Tailwind CSS for styling
- React Beautiful DnD for drag-and-drop

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint

# Type checking
npm run type-check
```

## Project Structure

```
src/
├── components/
│   ├── PipelineBuilder/     # Drag-drop pipeline builder
│   ├── ModelSelector/       # Searchable model picker
│   ├── ProgressMonitor/     # Real-time progress modal
│   ├── FileDropZone/        # File upload with preview
│   └── SettingsPanel/       # Configuration settings
├── hooks/
│   ├── useModels.ts         # Model data fetching
│   ├── useProcessing.ts     # Pipeline execution
│   ├── useProgress.ts       # Progress state
│   ├── useSettings.ts       # Settings management
│   └── store.ts             # Typed Redux hooks
├── store/
│   ├── pipelineStore.ts     # Pipeline state
│   ├── progressStore.ts     # Progress tracking
│   └── index.ts             # Store configuration
├── services/
│   ├── api.ts               # API client
│   └── sse.ts               # Server-Sent Events
├── types/
│   ├── models.ts            # Model types
│   ├── processing.ts        # Processing types
│   └── settings.ts          # Settings types
├── App.tsx                  # Main app component
├── main.tsx                 # Entry point
└── index.css                # Global styles + Tailwind
```

## API Integration

The frontend connects to the openVideoAI FastAPI backend at `http://localhost:8000/api` (configurable via `VITE_API_BASE_URL`).

Key endpoints used:
- `POST /api/processing/pipeline/file` - Execute pipeline
- `GET /api/processing/task/{task_id}` - Get task status
- `GET /api/processing/features` - Get available features
- `GET /api/models` - List models
- `POST /api/models/download` - Download model
- `GET /api/settings` - Get settings
- `POST /api/settings` - Update settings

## Environment Variables

Create a `.env.local` file:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## Building for Production

```bash
npm run build
```

Output will be in the `dist/` directory, ready for deployment.