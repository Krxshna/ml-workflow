import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from core.config import Settings, get_settings
from core.data_loader import load_dataset
from core.feature_engine import FeatureEngine
from core.logging_config import configure_logging
from core.models import HealthResponse, LivenessResponse, ProductDetail, SimilarProductsResponse
from core.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)

_engine: Optional[SimilarityEngine] = None
_df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _df
    settings: Settings = get_settings()
    logger.info("Starting up")
    start = time.perf_counter()

    _df = load_dataset(settings.data_path)
    feature_engine = FeatureEngine(settings)
    features = feature_engine.fit_transform(_df)
    _engine = SimilarityEngine(features, _df.index.tolist(), settings)

    logger.info(
        "Startup complete in %.2fs — %d products ready",
        time.perf_counter() - start,
        _engine.product_count,
    )
    yield
    logger.info("Shutting down")
    _engine = None
    _df = None


def _create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    return FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
    )


app = _create_app()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "error_code": "INTERNAL_ERROR"},
    )


@app.get("/healthz", response_model=LivenessResponse, tags=["System"])
def liveness() -> LivenessResponse:
    """Liveness probe: the process is up and serving requests. Never depends
    on the index being built — that's what /health (readiness) checks."""
    return LivenessResponse(status="alive")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service is initializing")
    return HealthResponse(
        status="ok",
        products_loaded=_engine.product_count,
        vector_dim=_engine.vector_dim,
    )


@app.get("/find_similar_products", response_model=list[str], tags=["Similarity"])
def find_similar_products(
    product_id: Annotated[str, Query(description="Unique product ID (uniq_id field)")],
    num_similar: Annotated[
        int,
        Query(ge=1, le=get_settings().max_similar, description="Number of similar products to return"),
    ],
) -> list[str]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service is initializing")

    result = _engine.find_similar(product_id, num_similar)

    if result is None:
        logger.info("find_similar_products: unknown product_id=%s", product_id)
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in dataset",
        )

    logger.debug(
        "find_similar_products: product_id=%s num_similar=%d -> %d results",
        product_id, num_similar, len(result),
    )
    return result


@app.get("/products/{product_id}", response_model=ProductDetail, tags=["Products"])
def get_product(product_id: str) -> ProductDetail:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service is initializing")

    if not _engine.product_exists(product_id):
        logger.info("get_product: unknown product_id=%s", product_id)
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in dataset",
        )

    row = _df.loc[product_id]
    return ProductDetail(
        product_id=product_id,
        product_name=row.get("product_name") or None,
        brand=row.get("brand") or None,
        colour=row.get("colour") or None,
        sales_price=float(row["sales_price"]) if row.get("sales_price") == row.get("sales_price") else None,
        rating=float(row["rating"]) if row.get("rating") == row.get("rating") else None,
        category=row.get("category") or None,
    )


@app.get("/similar_with_details", response_model=SimilarProductsResponse, tags=["Similarity"])
def find_similar_with_details(
    product_id: Annotated[str, Query(description="Unique product ID")],
    num_similar: Annotated[
        int,
        Query(ge=1, le=get_settings().max_similar, description="Number of similar products"),
    ],
) -> SimilarProductsResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service is initializing")

    result = _engine.find_similar(product_id, num_similar)

    if result is None:
        logger.info("similar_with_details: unknown product_id=%s", product_id)
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in dataset",
        )

    logger.debug(
        "similar_with_details: product_id=%s num_similar=%d -> %d results",
        product_id, num_similar, len(result),
    )
    return SimilarProductsResponse(
        product_id=product_id,
        similar_product_ids=result,
        count=len(result),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
