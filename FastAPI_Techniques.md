# FastAPI Complete Study Guide

## 1. Core Concepts & Fundamentals

### Path Operations
- Define endpoints using HTTP method decorators: `@app.get()`, `@app.post()`, `@app.put()`, `@app.patch()`, `@app.delete()`, `@app.options()`, `@app.head()`
- Each decorator maps to an HTTP method and URL path
- Path operations are executed in order of definition

### Parameters

#### Path Parameters
- Extract values from URL: `/items/{item_id}`
- Automatically validated and converted to specified type
- Support for `int`, `str`, `float`, `bool`, `UUID`, `Path` types

#### Query Parameters
- Extract from URL query string: `?q=search&skip=0&limit=10`
- Optional by default (use `Optional[str]` or default values)
- Support multiple values with `List[str]`

#### Header Parameters
- Access HTTP headers with `Header()` dependency
- Automatic conversion of hyphenated names to underscores

#### Cookie Parameters
- Read cookies using `Cookie()` dependency
- Type validation and conversion

### Request Body
- Parse JSON bodies using Pydantic models
- Automatic validation, serialization, and documentation
- Support for nested models and complex structures

### Response Handling
- Define response models with `response_model` parameter
- Control status codes with `status_code` parameter
- Filter response fields with `response_model_exclude` and `response_model_include`
- Return custom response classes: `JSONResponse`, `HTMLResponse`, `FileResponse`, `StreamingResponse`, `RedirectResponse`

---

## 2. Pydantic Models & Validation

### BaseModel Basics
- Define request/response schemas using Pydantic `BaseModel`
- Automatic type validation and conversion
- Generate JSON schemas for OpenAPI documentation

### Field Validation
- Use `Field()` for constraints: `min_length`, `max_length`, `ge`, `le`, `gt`, `lt`, `regex`
- Set default values and mark fields as required/optional
- Add descriptions and examples for documentation

### Custom Validators
- Create field validators with `@field_validator` (Pydantic v2)
- Create model validators with `@model_validator`
- Use `@validator` decorator (Pydantic v1)

### Model Features
- Model inheritance for code reuse
- Nested models for complex data structures
- Field aliases for different naming conventions
- Config class for model behavior customization
- Enum support for restricted value sets

---

## 3. Dependency Injection System

### Depends() Basics
- Create reusable dependencies with `Depends()`
- Share logic across multiple endpoints
- Dependencies can have their own dependencies (nested dependencies)

### Common Use Cases
- Database session management
- Authentication and authorization
- Shared configuration
- Request validation
- Service layer injection

### Dependency Scopes
- Function dependencies (most common)
- Class-based dependencies
- Generator dependencies with cleanup (using `yield`)

### Dependency Overrides
- Override dependencies for testing with `app.dependency_overrides`
- Useful for mocking databases, external services

---

## 4. Routing & Organization

### APIRouter
- Organize routes into modules with `APIRouter`
- Prefix routes: `APIRouter(prefix="/items")`
- Add tags for documentation grouping
- Include routers in main app with `app.include_router()`

### Route Organization
- Group related endpoints in separate router files
- Use tags to organize documentation
- Version APIs using prefixes: `/v1/`, `/v2/`
- Modular architecture for large applications

### Static Files
- Serve static files with `StaticFiles` middleware
- Mount at specific path: `app.mount("/static", StaticFiles(directory="static"), name="static")`

---

## 5. Authentication & Security

### OAuth2 Implementation
- Use `OAuth2PasswordBearer` for token-based auth
- Implement `OAuth2PasswordRequestForm` for login
- Support OAuth2 scopes for granular permissions

### JWT Tokens
- Create tokens with claims (user ID, expiration, scopes)
- Verify and decode tokens in dependencies
- Implement refresh token mechanism

### Authentication Methods
- Bearer token authentication
- API key authentication with `APIKeyHeader`, `APIKeyQuery`, `APIKeyCookie`
- HTTP Basic authentication with `HTTPBasic`
- Session-based authentication

### Password Security
- Hash passwords with bcrypt or passlib
- Never store plain-text passwords
- Validate password strength

### Authorization
- Role-based access control (RBAC)
- Permission-based access control
- Scope-based authorization with OAuth2

### CORS Configuration
- Configure CORS middleware with `CORSMiddleware`
- Set allowed origins, methods, headers
- Handle preflight requests

---

## 6. Middleware

### Built-in Middleware
- `CORSMiddleware` for cross-origin requests
- `GZipMiddleware` for response compression
- `TrustedHostMiddleware` for host validation
- `HTTPSRedirectMiddleware` for enforcing HTTPS

### Custom Middleware
- Create middleware functions or classes
- Access request before and after processing
- Use for logging, timing, request modification
- Add with `@app.middleware("http")`

---

## 7. Background Tasks

### BackgroundTasks
- Execute tasks after returning response
- Use `BackgroundTasks` dependency
- Ideal for emails, notifications, cleanup operations

### Task Queue Integration
- Celery for distributed task processing
- RQ (Redis Queue) for simpler use cases
- Dramatiq as alternative to Celery
- APScheduler for scheduled/periodic tasks

---

## 8. Asynchronous Programming

### Async/Await Support
- Define async endpoints with `async def`
- Non-blocking I/O operations
- Better concurrency and performance

### Async Operations
- Async database queries (SQLAlchemy async, Tortoise ORM)
- Async HTTP requests (httpx, aiohttp)
- Async file I/O
- Async background tasks

### When to Use Async
- I/O-bound operations (database, network, files)
- High concurrency requirements
- WebSocket connections
- Not needed for CPU-bound tasks

---

## 9. Database Integration

### ORM Options
- SQLAlchemy (sync and async)
- Tortoise ORM (async-first)
- Peewee (lightweight)
- Beanie (MongoDB async)

### Database Patterns
- Dependency injection for sessions
- Connection pooling
- Transaction management
- Query optimization

### Migrations
- Alembic for SQLAlchemy migrations
- Version control for database schema
- Automated migration generation

### Caching
- Redis for caching frequently accessed data
- In-memory caching for session data
- Cache invalidation strategies

---

## 10. Form Data & File Handling

### Form Data
- Parse form fields with `Form()` dependency
- Handle `application/x-www-form-urlencoded`
- Handle `multipart/form-data`

### File Uploads
- Single file upload with `UploadFile`
- Multiple files with `List[UploadFile]`
- Async file reading/writing
- Validate file types and sizes

### File Downloads
- Return files with `FileResponse`
- Stream large files with `StreamingResponse`
- Set content disposition headers

---

## 11. Exception Handling

### Built-in Exceptions
- `HTTPException` for standard HTTP errors
- Set status code and detail message
- Add custom headers to exceptions

### Custom Exception Handlers
- Register handlers with `@app.exception_handler()`
- Handle specific exception types
- Return custom error responses

### Global Error Handling
- Catch all exceptions with generic handler
- Log errors for debugging
- Return user-friendly error messages

---

## 12. WebSockets

### WebSocket Endpoints
- Define WebSocket routes with `@app.websocket()`
- Accept connections with `await websocket.accept()`
- Send/receive data with `await websocket.send_text()`, `await websocket.receive_text()`

### WebSocket Patterns
- Real-time chat applications
- Live notifications
- Broadcasting to multiple clients
- WebSocket authentication

---

## 13. Testing

### TestClient
- Test endpoints without running server
- Built on `requests` library (sync) or `httpx` (async)
- Create test client: `client = TestClient(app)`

### Testing Strategies
- Unit tests with pytest
- Integration tests for complete flows
- Mock dependencies with `app.dependency_overrides`
- Test authentication and authorization
- Use fixtures for reusable test data

### Async Testing
- Use `@pytest.mark.anyio` or `@pytest.mark.asyncio`
- Test async endpoints properly
- Mock async dependencies

---

## 14. Documentation & OpenAPI

### Automatic Documentation
- Swagger UI at `/docs`
- ReDoc at `/redoc`
- OpenAPI schema at `/openapi.json`

### Customization
- Add tags for endpoint grouping
- Set summary and description for operations
- Provide examples for request/response
- Add metadata (title, version, description)

### Documentation Control
- Hide endpoints with `include_in_schema=False`
- Customize OpenAPI schema with `openapi()` override
- Change documentation URLs

---

## 15. Performance Optimization

### Server Configuration
- Run with Uvicorn for ASGI support
- Use multiple workers with Gunicorn + Uvicorn
- Configure worker timeout and keep-alive

### Application Optimization
- Use async for I/O-bound operations
- Implement caching (Redis, in-memory)
- Database query optimization
- Connection pooling
- Response compression with GZip

### Pydantic Optimization
- Use `model_validate()` for performance
- Lazy imports for faster startup
- Avoid unnecessary validation

### HTTP Optimization
- Set ETags for caching
- Use cache-control headers
- Response streaming for large data

---

## 16. Configuration & Settings

### Environment Variables
- Use `python-dotenv` to load `.env` files
- Read variables with `os.getenv()`
- Type-safe settings with Pydantic `BaseSettings`

### Settings Management
- Create settings class inheriting from `BaseSettings`
- Automatic validation of environment variables
- Support for nested configuration

### Event Handlers
- Startup events with `@app.on_event("startup")`
- Shutdown events with `@app.on_event("shutdown")`
- Initialize resources (database, cache) on startup
- Cleanup resources on shutdown

---

## 17. Deployment

### Containerization
- Dockerize FastAPI applications
- Multi-stage builds for smaller images
- Use official Python base images

### ASGI Servers
- Uvicorn (development and production)
- Hypercorn (alternative to Uvicorn)
- Daphne (Django-based ASGI server)

### Production Deployment
- Use process managers (Gunicorn, Supervisor)
- Reverse proxy with Nginx
- Load balancing across multiple workers

### Cloud Platforms
- AWS Lambda with Mangum adapter
- Google Cloud Run
- Azure Functions
- Render, Railway, Fly.io
- Heroku with Procfile

### CI/CD
- Automated testing in pipelines
- Docker image building and pushing
- Environment-specific deployments
- Health check endpoints

---

## 18. Advanced Features

### GraphQL Integration
- Use Strawberry or Ariadne for GraphQL
- Combine REST and GraphQL endpoints
- Type-safe GraphQL with Pydantic

### Rate Limiting
- Implement custom rate limiting middleware
- Use libraries like `slowapi`
- Throttle requests per user/IP

### Logging
- Configure Python logging module
- Custom logger for structured logging
- Log requests, responses, errors
- Integration with monitoring tools

### Template Rendering
- Use Jinja2 for HTML templates
- Return `HTMLResponse` with rendered templates
- Serve dynamic web pages

---

## 19. Best Practices

### Code Organization
- Separate routers, models, services, dependencies
- Use dependency injection for loose coupling
- Keep business logic in service layer

### Security
- Validate all input data
- Use HTTPS in production
- Implement rate limiting
- Sanitize user input
- Keep dependencies updated

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Log errors for debugging
- Don't expose sensitive information

### Documentation
- Write clear docstrings
- Provide examples in OpenAPI
- Keep API versioned
- Document breaking changes

### Testing
- Write tests for all endpoints
- Test error cases and edge cases
- Use fixtures for test data
- Mock external dependencies

---

## 20. Common Patterns

### Repository Pattern
- Separate database operations from business logic
- Create repository classes for each model
- Inject repositories as dependencies

### Service Layer
- Business logic in service classes
- Services use repositories for data access
- Inject services into route handlers

### Request/Response Models
- Separate models for create, update, read operations
- Use inheritance to avoid duplication
- Validate input, sanitize output

### Error Response Structure
- Consistent error format across API
- Include error code, message, details
- Use appropriate HTTP status codes

---

## Quick Reference: Common Decorators & Functions

- `@app.get()`, `@app.post()`, `@app.put()`, `@app.delete()`, `@app.patch()` - Define endpoints
- `@app.on_event("startup")`, `@app.on_event("shutdown")` - Lifecycle events
- `@app.middleware("http")` - Add custom middleware
- `@app.exception_handler()` - Handle exceptions
- `@app.websocket()` - WebSocket endpoint
- `Depends()` - Dependency injection
- `Path()`, `Query()`, `Header()`, `Cookie()`, `Body()`, `Form()`, `File()` - Parameter extraction
- `HTTPException()` - Raise HTTP errors
- `BackgroundTasks` - Execute tasks after response
- `status` - HTTP status codes module

---

## Study Tips for Exam Preparation

1. Practice implementing each feature hands-on
2. Understand when to use sync vs async
3. Master Pydantic models and validation
4. Know dependency injection patterns
5. Understand authentication flows
6. Practice error handling and testing
7. Review OpenAPI documentation customization
8. Study deployment and production configurations