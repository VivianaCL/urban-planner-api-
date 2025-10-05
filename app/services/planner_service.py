import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import folium
from typing import List, Tuple
from ..models.GAPlanner import Park_School_Planner
from ..schemas.requests import PlannerRequest, RecomendacionResponse

class PlannerService:
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.inequality = None
        self.inegi = None
        self.merged = None
        self.cols_principales = [
            "CICLOVIA_C", "CICLOCAR_C", "RECUCALL_C", "BANQUETA_C",
            "GUARNICI_C", "LETRERO_C", "ACESOAUT_C", "PUESAMBU_C",
            "ACESOPER_C", "PUESSEMI_C", "POBTOT", "VPH_C_EL", 
            "VPH_DREN", "VPH_EXSA", "TVIPAHAB", "VPH_PIDT", 
            "POBMAS", "POBFEM", "VIVPAR", "VIVTOT", "Salud_cons", 
            "Abasto", "Empleo", "E_basica", "E_superior", "E_media", 
            "Salud_cama", "Est_Tpte", "GRAPROES", "GRAPROES_F",
            "GRAPROES_M"
        ]
        
    def load_data(self):
        """Carga los datos geoespaciales"""
        if self.merged is None:
            self.inequality = gpd.read_file(f'{self.data_path}M19_01_Monterrey.shp')
            self.inequality = self.inequality.to_crs(epsg=4326)
            
            self.inegi = pd.read_excel(f'{self.data_path}datos_monterrey.xlsx')
            from shapely import wkt
            self.inegi["geometry"] = self.inegi["geometry"].apply(wkt.loads)
            self.inegi = gpd.GeoDataFrame(self.inegi, geometry="geometry", crs="EPSG:4326")
            
            self.merged = gpd.sjoin(self.inegi, self.inequality, how="left", predicate="intersects")
            self.merged = self.merged.dropna()
    
    def filter_zone(self, bbox: dict) -> gpd.GeoDataFrame:
        """Filtra zona según bounding box"""
        zona_bbox = box(bbox['lon_min'], bbox['lat_min'], 
                       bbox['lon_max'], bbox['lat_max'])
        zona_filtrada = self.merged[
            self.merged['geometry'].centroid.within(zona_bbox)
        ].copy()
        return zona_filtrada
    
    def run_planner(self, request: PlannerRequest) -> Tuple[List, dict]:
        """Ejecuta el algoritmo genético"""
        self.load_data()
        zona_filtrada = self.filter_zone(request.bbox.dict())
        
        planner = Park_School_Planner(
            df=zona_filtrada,
            cols_principales=self.cols_principales,
            pop_size=request.pop_size,
            n_gen=request.n_gen,
            mut_rate=request.mut_rate,
            max_acciones=request.max_acciones,
            presupuesto_parques=request.presupuesto_parques,
            presupuesto_escuelas=request.presupuesto_escuelas
        )
        
        recomendaciones = planner.run()
        insights = planner.get_insights()
        
        return recomendaciones, insights
    
    def generate_map(self, recomendaciones: List) -> str:
        """Genera mapa HTML con folium"""
        if not recomendaciones:
            return None
            
        lat = self.merged['lat'].mean()
        lon = self.merged['lon'].mean()
        m = folium.Map(location=[lat, lon], zoom_start=12)
        
        for geom_recom, accion in recomendaciones:
            centroide = geom_recom.centroid
            
            for poly in self.inequality.geometry:
                if centroide.within(poly):
                    color = "green" if accion == "Parque" else "blue"
                    folium.GeoJson(
                        poly.__geo_interface__,
                        style_function=lambda feature, col=color: {
                            "fillColor": col,
                            "color": col,
                            "weight": 1,
                            "fillOpacity": 0.9
                        }
                    ).add_to(m)
                    break
        
        return m._repr_html_()
    
    def process_request(self, request: PlannerRequest) -> dict:
        """Procesa request completo"""
        recomendaciones, insights = self.run_planner(request)
        
        recomendaciones_response = [
            RecomendacionResponse(
                lat=geom.centroid.y,
                lon=geom.centroid.x,
                tipo=tipo
            )
            for geom, tipo in recomendaciones
        ]
        
        mapa_html = self.generate_map(recomendaciones)
        
        return {
            "recomendaciones": recomendaciones_response,
            "insights": insights,
            "mapa_html": mapa_html
        }