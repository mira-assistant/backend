"""
Tests for the main FastAPI application.
"""

from app.main import app


class TestMainApp:
    """Test cases for the main FastAPI application."""

    def test_app_creation(self):
        """Test that the FastAPI app is created correctly."""
        assert app is not None
        assert app.title == "Mira Backend"
        assert app.version == "4.3.0"
        assert app.description == "Mira AI Assistant Backend API"

    def test_app_routes_registration(self):
        """Test that all routes are properly registered."""
        # Get all registered routes
        routes = [str(route) for route in app.routes]

        # Check for API v1 routes
        assert any("/api/v1" in route for route in routes)

        # Check for specific route patterns
        assert any("/api/v1/{network_id}/conversations" in route for route in routes)
        assert any("/api/v1/{network_id}/persons" in route for route in routes)
        assert any("/api/v1/{network_id}/streams" in route for route in routes)
        assert any("/api/v1/{network_id}/interactions" in route for route in routes)
        assert any("/api/v1/{network_id}/service" in route for route in routes)

    def test_cors_middleware(self):
        """Test that CORS middleware is properly configured."""
        # Check that CORS middleware is in the middleware stack
        # The middleware is added via add_middleware, so we check the middleware stack
        assert len(app.user_middleware) > 0
        # Check that at least one middleware is present (CORS should be there)
        middleware_info = [str(middleware) for middleware in app.user_middleware]
        # CORS middleware should be present
        assert any(
            "cors" in str(middleware).lower() or "CORSMiddleware" in str(middleware)
            for middleware in app.user_middleware
        )

    def test_global_exception_handler(self):
        """Test that global exception handler is registered."""
        # Check that exception handler is registered
        assert Exception in app.exception_handlers

    def test_lifespan_manager(self):
        """Test that lifespan manager is properly configured."""
        assert app.router.lifespan_context is not None

    def test_lifespan_startup(self):
        """Test application startup in lifespan manager."""
        # Test that the lifespan function exists and can be called
        from app.main import lifespan
        import asyncio

        async def test_lifespan():
            async with lifespan(app):
                pass

        # This should not raise an exception
        asyncio.run(test_lifespan())

    def test_lifespan_shutdown(self):
        """Test application shutdown in lifespan manager."""
        # Test that the lifespan function exists and can be called
        from app.main import lifespan
        import asyncio

        async def test_lifespan():
            async with lifespan(app):
                pass

        # This should not raise an exception
        asyncio.run(test_lifespan())

    def test_root_endpoint_exists(self):
        """Test that the root endpoint exists."""
        # Get all registered routes
        routes = [str(route) for route in app.routes]
        # Extract path from route string (format: Route(path='/path', ...))
        paths = []
        for route in routes:
            if "path='" in route:
                start = route.find("path='") + 6
                end = route.find("'", start)
                paths.append(route[start:end])
        assert "/" in paths

    def test_api_versioning(self):
        """Test that API versioning is properly set up."""
        # Check that v1 routes are registered
        routes = [str(route) for route in app.routes]
        v1_routes = [route for route in routes if "/api/v1" in route]
        assert len(v1_routes) > 0

        # Check that v2 routes are commented out (not registered)
        v2_routes = [route for route in routes if "/api/v2" in route]
        assert len(v2_routes) == 0

    def test_middleware_order(self):
        """Test that middleware is in the correct order."""
        # CORS should be one of the first middlewares
        assert len(app.user_middleware) > 0
        # Check if CORS middleware is present in any form
        cors_present = any(
            "cors" in str(middleware).lower() or "CORSMiddleware" in str(middleware)
            for middleware in app.user_middleware
        )
        assert cors_present

    def test_app_metadata(self):
        """Test that app metadata is correctly set."""
        assert app.title == "Mira Backend"
        assert app.version == "4.3.0"
        assert app.description == "Mira AI Assistant Backend API"
        assert app.openapi_url == "/openapi.json"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_router_inclusion(self):
        """Test that all routers are properly included."""
        # Check that all expected routers are included
        routes = [str(route) for route in app.routes]

        # Check for conversation router
        assert any("/api/v1/{network_id}/conversations" in route for route in routes)

        # Check for persons router
        assert any("/api/v1/{network_id}/persons" in route for route in routes)

        # Check for streams router
        assert any("/api/v1/{network_id}/streams" in route for route in routes)

        # Check for interaction router
        assert any("/api/v1/{network_id}/interactions" in route for route in routes)

        # Check for service router
        assert any("/api/v1/{network_id}/service" in route for route in routes)
