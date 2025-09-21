# API Documentation

## Authentication Endpoints

### Base URL
```
https://your-api-domain.com/api/v1/auth
```

### Endpoints

#### POST /register
Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123",
  "username": "optional_username",
  "full_name": "John Doe",
  "is_active": true
}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "optional_username",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "avatar_url": null,
  "created_at": "2024-01-01T12:00:00Z",
  "last_login": null
}
```

**Error Responses:**
- `400 Bad Request`: Email already registered
- `422 Unprocessable Entity`: Invalid email format or password too short

#### POST /login
Login with email and password.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error Responses:**
- `401 Unauthorized`: Incorrect email or password

#### POST /refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid refresh token

#### POST /logout
Logout user (invalidates tokens on client side).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200):**
```json
{
  "message": "Successfully logged out"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or missing token

#### GET /me
Get current user information.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "optional_username",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "avatar_url": null,
  "created_at": "2024-01-01T12:00:00Z",
  "last_login": "2024-01-01T12:30:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or missing token
- `400 Bad Request`: Inactive user

### Google OAuth 2.0 Endpoints

#### GET /google
Initiate Google OAuth 2.0 authentication flow.

**Response:**
Redirects to Google's OAuth 2.0 authorization server.

#### GET /google/callback
Handle Google OAuth 2.0 callback.

**Query Parameters:**
- `code`: Authorization code from Google
- `state`: CSRF protection parameter

**Response (200):**
```json
{
  "message": "Successfully authenticated with Google",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@gmail.com",
    "full_name": "John Doe",
    "is_active": true,
    "is_verified": true,
    "avatar_url": "https://lh3.googleusercontent.com/...",
    "created_at": "2024-01-01T12:00:00Z",
    "last_login": "2024-01-01T12:30:00Z"
  }
}
```

**Error Responses:**
- `400 Bad Request`: Failed to authenticate with Google
- `500 Internal Server Error`: Authentication error
- `501 Not Implemented`: Google OAuth not configured

## Authentication Flow

### Token-Based Authentication
1. User registers or logs in
2. Server returns access token (30 min expiry) and refresh token (7 days expiry)
3. Client includes access token in Authorization header for protected endpoints
4. When access token expires, use refresh token to get new tokens

### Google OAuth 2.0 Flow
1. User clicks "Login with Google"
2. Frontend redirects to `/auth/google`
3. Server redirects to Google's OAuth page
4. User grants permission
5. Google redirects to `/auth/google/callback` with authorization code
6. Server exchanges code for user profile and creates/updates user account
7. Server returns JWT tokens for the user

## Protected Endpoints

All existing API endpoints can be protected by adding the authentication dependency:

```python
@router.get("/protected-endpoint")
async def protected_endpoint(current_user: User = Depends(get_current_user)):
    # Only accessible with valid JWT token
    return {"message": f"Hello {current_user.email}"}
```

## Environment Variables Required

```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here-minimum-32-characters
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Google OAuth 2.0
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Database (existing)
DATABASE_URL=postgresql://username:password@host:port/database
```

## CORS Configuration

The backend is already configured to accept requests from all origins (`*`). For production, update the `CORS_ORIGINS` environment variable to restrict to specific domains:

```bash
CORS_ORIGINS=https://your-electron-app.com,http://localhost:3000
```

## Example Usage

### Frontend JavaScript Example

```javascript
// Login
const response = await fetch('/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password123'
  })
});
const tokens = await response.json();
localStorage.setItem('access_token', tokens.access_token);
localStorage.setItem('refresh_token', tokens.refresh_token);

// Protected request
const protectedResponse = await fetch('/api/v1/some-protected-endpoint', {
  headers: { 
    'Authorization': `Bearer ${localStorage.getItem('access_token')}` 
  }
});
```

### Electron App Integration

```javascript
// In your Electron renderer process
const { ipcRenderer } = require('electron');

// Handle Google OAuth
document.getElementById('google-login').addEventListener('click', () => {
  ipcRenderer.send('start-google-auth');
});

// In your Electron main process
const { BrowserWindow } = require('electron');

ipcMain.on('start-google-auth', () => {
  const authWindow = new BrowserWindow({
    width: 500,
    height: 600,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  authWindow.loadURL('https://your-api-domain.com/api/v1/auth/google');
  authWindow.show();

  // Handle successful authentication
  authWindow.webContents.on('will-navigate', (event, url) => {
    if (url.includes('access_token')) {
      // Parse tokens from URL or response
      authWindow.close();
      // Send tokens to renderer process
    }
  });
});
```