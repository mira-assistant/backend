#!/bin/bash
# Authentication System Demo Script

echo "🎯 Mira Backend Authentication System Demo"
echo "=========================================="
echo

# Set environment variables for demo
export JWT_SECRET_KEY="demo-secret-key-for-testing-32-characters"
export DATABASE_URL="sqlite:///./test.db"  # For demo purposes

echo "📋 Authentication System Features:"
echo "  ✅ JWT Token-based authentication"
echo "  ✅ User registration and login"
echo "  ✅ Password hashing with bcrypt"
echo "  ✅ Google OAuth 2.0 integration"
echo "  ✅ Protected endpoint middleware"
echo "  ✅ Token refresh functionality"
echo "  ✅ Comprehensive API documentation"
echo

echo "📱 Available Endpoints:"
echo "  POST   /api/v1/auth/register       - Register new user"
echo "  POST   /api/v1/auth/login          - Login with email/password"
echo "  POST   /api/v1/auth/refresh        - Refresh access token"
echo "  POST   /api/v1/auth/logout         - Logout (client-side)"
echo "  GET    /api/v1/auth/me             - Get current user info"
echo "  GET    /api/v1/auth/google         - Google OAuth redirect"
echo "  GET    /api/v1/auth/google/callback - Google OAuth callback"
echo

echo "🔧 Configuration Required:"
echo "  Environment Variables:"
echo "    JWT_SECRET_KEY                  - Secret key for JWT tokens"
echo "    GOOGLE_CLIENT_ID                - Google OAuth client ID"
echo "    GOOGLE_CLIENT_SECRET            - Google OAuth client secret"
echo "    DATABASE_URL                    - Database connection string"
echo

echo "🚀 To start the server:"
echo "  cd app"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
echo

echo "🔐 Example Usage:"
echo "  # Register a new user"
echo '  curl -X POST http://localhost:8000/api/v1/auth/register \'
echo '    -H "Content-Type: application/json" \'
echo '    -d {"email":"user@example.com","password":"password123"}'
echo
echo "  # Login to get tokens"
echo '  curl -X POST http://localhost:8000/api/v1/auth/login \'
echo '    -H "Content-Type: application/json" \'
echo '    -d {"email":"user@example.com","password":"password123"}'
echo
echo "  # Access protected endpoint"
echo '  curl -X GET http://localhost:8000/api/v1/auth/me \'
echo '    -H "Authorization: Bearer YOUR_ACCESS_TOKEN"'
echo

echo "📚 Full documentation available in API_DOCUMENTATION.md"
echo "🎉 Authentication system is ready for production!"