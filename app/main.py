from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn

from .api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Fake Review Detector API",
    description="AI-powered fake review detection for e-commerce platforms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Review Analysis"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fake Review Detector API",
        "version": "1.0.0",
        "description": "Detect fake reviews using AI/ML techniques",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "analyze_single": "/api/v1/analyze/single",
            "analyze_batch": "/api/v1/analyze/batch",
            "quick_analyze": "/api/v1/analyze/quick",
            "model_info": "/api/v1/model/info"
        },
        "features": [
            "Text pattern analysis",
            "Sentiment-rating correlation",
            "Reviewer behavior analysis",
            "Temporal pattern detection",
            "Batch processing support"
        ]
    }

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )