# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LiveKit Vision Demo showcasing a voice AI assistant with realtime audio and video input. The project consists of:

- **Agent Backend**: Python-based agent using LiveKit's Agents framework and Google's Gemini Live API
- **iOS Frontend**: Native Swift app built on LiveKit's Swift SDK with camera, screen sharing, and voice capabilities

## Development Commands

### Python Agent Development

```bash
# Set up Python environment
cd agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the agent locally
python main.py dev

# Code formatting and linting
ruff format .
ruff check .
```

### iOS App Development

- Open `swift-frontend/VisionDemo/VisionDemo.xcodeproj` in Xcode 16+
- Build and run on physical iOS 17+ device (simulator not supported)
- Requires proper bundle identifier configuration and App Group setup

## Architecture

### Agent Architecture (`agent/main.py`)
- Built on `MultimodalAgent` class with Google Gemini Live API integration
- Handles realtime video/audio streams with configurable frame sampling rates
- Uses byte stream handlers for image processing and chat context management
- Implements noise cancellation via LiveKit Cloud plugin

### iOS App Architecture
- **Entry Point**: `VisionDemoApp.swift` - Main app with `ChatContextProvider` wrapper
- **Views**: Modular SwiftUI views for camera, chat, connection, and action bar
- **Services**: `TokenService.swift` handles LiveKit authentication via sandbox or hardcoded tokens
- **Providers**: `ChatContextProvider.swift` manages chat state across the app
- **Broadcast Extension**: `SampleHandler.swift` enables screen sharing functionality

### Key iOS Components
- `ConnectionView`: Initial connection interface
- `ChatView`: Main chat interface with agent interaction
- `CameraView`: Camera feed display and controls
- `ActionBarView`: Bottom navigation with mic/camera/screen/disconnect buttons
- `AgentView`: Agent status and interaction display

## Configuration Requirements

### Agent Setup
- Create `agent/.env` with: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`
- LiveKit Cloud project required
- Google Gemini API key required

### iOS App Setup
- Create `swift-frontend/VisionDemo/Resources/Secrets.xcconfig` with `LK_SANDBOX_TOKEN_SERVER_ID`
- Configure unique bundle identifiers for main app and broadcast extension
- Set up App Group: `group.<your-bundle-identifier>`
- LiveKit Cloud Sandbox token server required

## Testing

The project supports testing via:
- LiveKit Agents Playground (browser-based testing at agents-playground.livekit.io)
- Physical iOS device testing (camera/screen sharing requires hardware)

## Key Dependencies

### Python Agent
- `livekit-agents[google,images]~=1.0,>=1.0.18`
- `livekit-plugins-noise-cancellation~=0.2`
- `python-dotenv`, `asyncio`

### iOS App
- LiveKit Swift SDK (managed via Swift Package Manager)
- iOS 17+ deployment target
- ReplayKit framework for screen recording