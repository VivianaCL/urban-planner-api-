from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .schemas.requests import PlannerRequest, PlannerResponse
from .services.planner_service import PlannerService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Urban Planner API",
    description="API para planificación urbana usando algoritmos genéticos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

planner_service = PlannerService()

@app.on_event("startup")
async def startup_event():
    """Carga datos al iniciar"""
    logger.info("Cargando datos geoespaciales...")
    planner_service.load_data()
    logger.info("Datos cargados exitosamente")

@app.get("/")
async def root():
    return {"message": "Urban Planner API - Running"}

@app.post("/plan", response_model=PlannerResponse)
async def create_plan(request: PlannerRequest):
    """
    Genera plan de parques y escuelas para zona especificada
    """
    try:
        result = planner_service.process_request(request)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan/map")
async def get_map(request: PlannerRequest):
    """
    Retorna solo el mapa HTML
    """
    try:
        recomendaciones, _ = planner_service.run_planner(request)
        mapa_html = planner_service.generate_map(recomendaciones)
        return HTMLResponse(content=mapa_html)
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}