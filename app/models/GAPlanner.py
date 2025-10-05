import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

class Park_School_Planner:
    def __init__(self, df, cols_principales, suelo_col="SUELO",
                 w_infraestructura=0.3, w_poblacion=0.5, w_servicios=0.2,
                 pop_size=30, n_gen=50, mut_rate=0.2,
                 max_acciones=20, presupuesto_parques=30, presupuesto_escuelas=20,
                 suelo_weights=None):
        
        self.df = df
        self.cols = cols_principales
        self.SUELO = df[suelo_col].values
        self.X_scaled = MinMaxScaler().fit_transform(df[cols_principales].fillna(0).values)
        self.N_ZONAS = self.X_scaled.shape[0]

        # pesos
        self.w_infraestructura = w_infraestructura
        self.w_poblacion = w_poblacion
        self.w_servicios = w_servicios
        
        # GA params
        self.POP_SIZE = pop_size
        self.N_GEN = n_gen
        self.MUT_RATE = mut_rate
        self.MAX_ACCIONES = max_acciones
        self.PRESUPUESTO_PARQUES = presupuesto_parques
        self.PRESUPUESTO_ESCUELAS = presupuesto_escuelas

        # pesos suelo
        self.suelo_weights = suelo_weights or {
            "residential":   [0.9, 0.8, 0.3],
            "park":          [0.2, 0.4, 0.9],
            "meadow":        [0.7, 0.5, 0.6],
            "scrub":         [0.6, 0.4, 0.5],
            "cemetery":      [0.1, 0.1, 0.9],
            "industrial":    [0.1, 0.3, 0.7],
            "grass":         [0.6, 0.5, 0.6],
            "retail":        [0.4, 0.7, 0.5],
            "nature_reserve":[0.2, 0.3, 0.8],
            "commercial":    [0.3, 0.8, 0.5],
            "quarry":        [0.1, 0.2, 0.9],
            "forest":        [0.2, 0.3, 0.8]
        }
    
    # ---------------- GA core functions ----------------
    def create_individual(self):
        individual = np.zeros(self.N_ZONAS, dtype=int)
        indices = np.random.choice(self.N_ZONAS, self.MAX_ACCIONES, replace=False)
        individual[indices] = np.random.choice([1, 2], size=self.MAX_ACCIONES)
        return individual

    def fitness(self, individual):
        n_parques  = np.sum(individual == 1)
        n_escuelas = np.sum(individual == 2)
        
        penalizacion = 0
        if n_parques > self.PRESUPUESTO_PARQUES:
            penalizacion += (n_parques - self.PRESUPUESTO_PARQUES) * 0.8
        if n_escuelas > self.PRESUPUESTO_ESCUELAS:
            penalizacion += (n_escuelas - self.PRESUPUESTO_ESCUELAS) * 0.8

        score = 0
        for i, accion in enumerate(individual):
            if accion == 0:
                continue
            zona = self.X_scaled[i]
            suelo = self.SUELO[i]
            
            if accion == 1:  # parque
                base = self.w_poblacion * zona[10] + self.w_infraestructura * zona[0:10].sum() + self.w_servicios * zona[20:30].sum()
            elif accion == 2:  # escuela
                base = self.w_poblacion * zona[10]*0.8 + self.w_infraestructura * zona[0:10].sum()*0.5 + self.w_servicios * zona[20:30].sum()*0.7
            
            ajuste = self.suelo_weights.get(suelo, [0.5,0.5,0.5])[accion]
            score += base * ajuste
        
        return score - penalizacion

    def crossover(self, p1, p2):
        mask = np.random.rand(self.N_ZONAS) < 0.5
        child = p1.copy()
        child[mask] = p2[mask]
        return child

    def mutate(self, individual):
        for i in range(self.N_ZONAS):
            if np.random.rand() < self.MUT_RATE:
                individual[i] = np.random.choice([0,1,2])
        return individual

    # ---------------- GA main loop ----------------
    def run(self):
        population = [self.create_individual() for _ in range(self.POP_SIZE)]

        for gen in range(self.N_GEN):
            scores = [self.fitness(ind) for ind in population]
            sorted_pop = [ind for _, ind in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)]
            population = sorted_pop[:self.POP_SIZE//2]
            
            children = []
            while len(children) < self.POP_SIZE//2:
                p1, p2 = random.sample(population, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                children.append(child)
            population += children

        self.best = max(population, key=self.fitness)

        recomendaciones = []
        for i, accion in enumerate(self.best):
            if accion == 1:
                recomendaciones.append([self.df.iloc[i].geometry, "Parque"])
            elif accion == 2:
                recomendaciones.append([self.df.iloc[i].geometry, "Escuela"])
        return recomendaciones

    # ---------------- Insights del GA ----------------
    def get_insights(self):
        if not hasattr(self, "best"):
            raise ValueError("Debes ejecutar `run()` primero para tener el mejor individuo.")

        n_parques = np.sum(self.best == 1)
        n_escuelas = np.sum(self.best == 2)
        score_total = self.fitness(self.best)

        contrib_infra = 0
        contrib_pobla = 0
        contrib_serv = 0
        suelo_count = {}

        for i, accion in enumerate(self.best):
            if accion == 0:
                continue
            zona = self.X_scaled[i]
            suelo = self.SUELO[i]
            if accion == 1:  # Parque
                base_infra = self.w_infraestructura * zona[0:10].sum()
                base_pobla = self.w_poblacion * zona[10]
                base_serv = self.w_servicios * zona[20:30].sum()
            elif accion == 2:  # Escuela
                base_infra = self.w_infraestructura * zona[0:10].sum() * 0.5
                base_pobla = self.w_poblacion * zona[10] * 0.8
                base_serv = self.w_servicios * zona[20:30].sum() * 0.7
            
            ajuste = self.suelo_weights.get(suelo, [0.5,0.5,0.5])[accion]
            contrib_infra += base_infra * ajuste
            contrib_pobla += base_pobla * ajuste
            contrib_serv += base_serv * ajuste

            suelo_count[suelo] = suelo_count.get(suelo, 0) + 1

        insights = {
            "score_total": float(score_total),
            "n_parques": int(n_parques),
            "n_escuelas": int(n_escuelas),
            "contrib_infra": float(contrib_infra),
            "contrib_pobla": float(contrib_pobla),
            "contrib_serv": float(contrib_serv),
            "suelo_count": {k: int(v) for k, v in suelo_count.items()}
        }
        return insights