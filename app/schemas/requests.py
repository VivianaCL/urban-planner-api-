from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    lat_min: float = Field(..., description="Latitud mínima")
    lat_max: float = Field(..., description="Latitud máxima")
    lon_min: float = Field(..., description="Longitud mínima")
    lon_max: float = Field(..., description="Longitud máxima")

class PlannerRequest(BaseModel):
    bbox: BoundingBox
    pop_size: int = Field(default=50, ge=10, le=200)
    n_gen: int = Field(default=50, ge=10, le=200)
    mut_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    max_acciones: int = Field(default=10, ge=1, le=50)
    presupuesto_parques: int = Field(default=5, ge=1)
    presupuesto_escuelas: int = Field(default=2, ge=1)

class RecomendacionResponse(BaseModel):
    lat: float
    lon: float
    tipo: str  # "Parque" o "Escuela"

class PlannerResponse(BaseModel):
    recomendaciones: List[RecomendacionResponse]
    insights: dict
    mapa_html: Optional[str] = None