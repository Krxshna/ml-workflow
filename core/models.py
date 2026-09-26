from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    products_loaded: int
    index_type: str = "FAISS HNSWFlat"
    vector_dim: int


class LivenessResponse(BaseModel):
    status: str


class ProductDetail(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    brand: Optional[str] = None
    colour: Optional[str] = None
    sales_price: Optional[float] = None
    rating: Optional[float] = None
    category: Optional[str] = None


class SimilarProductsResponse(BaseModel):
    product_id: str
    similar_product_ids: list[str]
    count: int


class ErrorDetail(BaseModel):
    detail: str
    error_code: str
