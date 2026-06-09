import math, random
import numpy as np
from scipy.spatial import cKDTree
from numba import njit

# ── 공유 상수 ────────────────────────────────────────────
MAP_W, MAP_H   = 4000, 2000
SPLIT_BASE     = 0.8
HUNT_RATIO     = 1.30
EAT_RATIO      = 1
MAX_SPEED      = 20.0
MAX_SIZE       = 40.0
MIN_SIZE       = 0.0
MAX_ORGANISMS  = 9999999
RESPAWN_TICK   = 30
HATCH_PERCENT  = 20

# ── 색상 상수 ────────────────────────────────────────────
BG       = (245, 243, 236)
CHART_BG = (22, 22, 28)
UI_BG    = (40, 39, 37)
GRID_COL = (215, 213, 206)
WHITE    = (235, 233, 228)
RED      = (220, 60, 40)
GOLD     = (255, 210, 50)
BLUE     = (80, 140, 220)
GREEN    = (60, 200, 80)

# ── Numba JIT 가속 함수 ──────────────────────────────────
@njit(fastmath=True)
def fast_hypot(dx, dy):
    return math.sqrt(dx * dx + dy * dy)

@njit(fastmath=True)
def fast_atan2(dy, dx):
    return math.atan2(dy, dx)

# ── 유틸 함수 ────────────────────────────────────────────
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rnd(a, b):
    return random.uniform(a, b)

def hsl_to_rgb(h, s, l):
    h /= 360
    if s == 0:
        v = int(l * 255)
        return (v, v, v)
    def hue2rgb(p, q, t):
        t %= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (int(hue2rgb(p, q, h + 1/3) * 255),
            int(hue2rgb(p, q, h      ) * 255),
            int(hue2rgb(p, q, h - 1/3) * 255))

def draw_star(surface, color, cx, cy, r, width=0):
    import pygame
    pts = []
    for i in range(10):
        angle  = math.pi / 2 + i * math.pi / 5
        radius = r if i % 2 == 0 else r * 0.45
        pts.append((cx + radius * math.cos(angle),
                    cy - radius * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts, width)

HUE_BINS = 12
def hue_to_base_size(hue):
    bucket = int(hue / 360 * HUE_BINS) % HUE_BINS
    rng    = random.Random(bucket * 137)
    return rng.uniform(5.0, 16.0)

# ── KD-Tree 공간 그리드 ──────────────────────────────────
class KDTreeGrid:
    def __init__(self, objects):
        self.objects = objects
        if objects:
            coords = np.array([(o.x, o.y) for o in objects], dtype=np.float32)
            self.tree = cKDTree(coords)
        else:
            self.tree = None

    def query(self, x, y, radius):
        if not self.tree:
            return []
        indices = self.tree.query_ball_point((x, y), radius)
        return [self.objects[i] for i in indices]

# ── 식물 ─────────────────────────────────────────────────
PLANT_TIERS = [
    ('small',  0.90, (0,   3.0), (50,  100, 40)),
    ('medium', 0.07, (3,   5.0), (70,  150, 55)),
    ('large',  0.02, (5,  10.0), (40,  120, 35)),
    ('giant',  0.01, (10.0,15.0),(25,   80, 20)),
]

class Plant:
    __slots__ = ('x','y','tier','size','base_color','color','alive')

    def __init__(self, tier=None):
        self.x, self.y = rnd(10, MAP_W - 10), rnd(10, MAP_H - 10)
        self.alive = True
        if tier is None:
            r = random.random(); cum = 0
            for name, prob, _, _ in PLANT_TIERS:
                cum += prob
                if r <= cum:
                    tier = name
                    break
            else:
                tier = 'medium'
        for name, prob, size_rng, col in PLANT_TIERS:
            if name == tier:
                self.tier  = name
                self.size  = rnd(*size_rng)
                self.color = col
                break

    def energy_value(self):
        return self.size ** 2

    def draw(self, surf, ox, oy, zoom, sim_w, sim_h):
        import pygame
        sx = int((self.x - ox) * zoom)
        sy = int((self.y - oy) * zoom)
        r  = max(1, int(self.size * 0.75 * zoom))

        if sx < -r or sy < -r or sx > sim_w + r or sy > sim_h + r:
            return

        if self.size >= 11.0:
            draw_star(surf, self.color, sx, sy, r)
        elif self.size >= 6.5:
            pygame.draw.circle(surf, self.color, (sx, sy), r)
            pygame.draw.circle(surf,
                (min(self.color[0]+40, 255), min(self.color[1]+40, 255), self.color[2]),
                (sx, sy), max(1, r - 2))
        else:
            pygame.draw.circle(surf, self.color, (sx, sy), r)


def _tier_targets(total):
    targets = {}; remaining = total
    for i, (name, prob, _, _) in enumerate(PLANT_TIERS):
        n = round(total * prob) if i < len(PLANT_TIERS) - 1 else remaining
        targets[name] = max(0, n); remaining -= targets[name]
    return targets

def make_plants_by_ratio(total):
    plants = []
    for name, n in _tier_targets(total).items():
        for _ in range(n):
            plants.append(Plant(tier=name))
    return plants

def replenish_plants_by_ratio(plants, target):
    if len(plants) >= target:
        return plants
    current = {name: 0 for name, *_ in PLANT_TIERS}
    for p in plants:
        current[p.tier] = current.get(p.tier, 0) + 1
    targets = _tier_targets(target)
    deficit = target - len(plants)
    added   = 0
    for name, tgt in targets.items():
        need  = tgt - current.get(name, 0)
        if need <= 0:
            continue
        batch = max(1, int(need // (100 / HATCH_PERCENT)))
        for _ in range(min(batch, deficit - added)):
            plants.append(Plant(tier=name)); added += 1
            if added >= deficit: break
        if added >= deficit: break
    return plants

# ── 개체 ─────────────────────────────────────────────────
class Organism:
    __slots__ = ('x','y','hue','size','speed','diet_gene','max_energy','energy',
                 'angle','alive','age','kill_count','eat_count',
                 '_body_color','_surf_cache_key','_cached_surf','splitcool','maxage')

    def __init__(self, x=None, y=None, speed=None, size=None, hue=None, diet_gene=None):
        self.x    = x    if x    is not None else rnd(20, MAP_W - 20)
        self.y    = y    if y    is not None else rnd(20, MAP_H - 20)
        self.hue  = hue  if hue  is not None else rnd(0, 360)
        if size is not None:
            self.size = clamp(size, MIN_SIZE, MAX_SIZE)
        else:
            base      = hue_to_base_size(self.hue)
            self.size = clamp(base + rnd(-3.0, 3.0), MIN_SIZE, MAX_SIZE)
        self.speed      = clamp(speed if speed is not None else rnd(1.0, 4.5), 0.3, MAX_SPEED)
        self.diet_gene  = diet_gene if diet_gene is not None else rnd(0.0, 1)
        self.max_energy = 2 * self.size ** 2
        self.energy     = self.max_energy * 0.5
        self.angle      = rnd(0, math.pi * 2)
        self.alive      = True
        self.age        = 0
        self.maxage     = 3000 * self.size ** 0.5 * (1 / self.speed)
        self.splitcool  = 0
        self.kill_count = 0
        self.eat_count  = 0
        self._body_color     = None
        self._surf_cache_key = None
        self._cached_surf    = None

    @property
    def is_carnivore(self): return self.diet_gene >= 0.6
    @property
    def is_herbivore(self): return self.diet_gene <= 0.4

    @property
    def diet_label(self):
        dg = self.diet_gene
        if dg >= 0.75: return "육식"
        if dg >= 0.55: return "준육식"
        if dg >= 0.45: return "잡식"
        if dg >= 0.25: return "준초식"
        return "초식"

    def hunt_efficiency(self): return 0.3 + 0.7 * self.diet_gene
    def herb_efficiency(self): return 0.3 + 0.7 * (1.0 - self.diet_gene)

    @property
    def split_threshold(self): return SPLIT_BASE * self.max_energy
    @property
    def sight_range(self):     return 80 + self.size * 20
    @property
    def hunt_range(self):      return 10 + self.size
    @property
    def flee_range(self):      return self.sight_range * 0.3

    def _drain(self): return self.speed * (self.size ** 1.5) * 0.003
    def can_hunt(self, other): return self.size >= other.size * HUNT_RATIO

    def update(self, plant_tree: KDTreeGrid, org_tree: KDTreeGrid):
        if self.splitcool > 0:
            self.splitcool -= 1
        self.age    += 1
        self.energy -= self._drain()

        sr = self.sight_range; fr = self.flee_range
        neighbors = org_tree.query(self.x, self.y, sr)

        threat  = None; threat_d = fr
        prey    = None; prey_d   = sr * (0.2 + 0.8 * self.diet_gene)

        for o in neighbors:
            if o is self or not o.alive: continue
            d = fast_hypot(o.x - self.x, o.y - self.y)
            if o.can_hunt(self) and d < threat_d and self.energy >= self.max_energy * 0.2:
                threat_d, threat = d, o
            if self.can_hunt(o) and d < prey_d and (self.energy >= self.max_energy * 0.2 and self.diet_gene < 0.9):
                prey_d, prey = d, o

        if threat:
            self.angle = fast_atan2(self.y - threat.y, self.x - threat.x)
        elif prey:
            self.angle = fast_atan2(prey.y - self.y, prey.x - self.x)
        elif self.diet_gene < 0.9:
            near_plants = plant_tree.query(self.x, self.y, sr)
            nearest = None; near_d = 99999
            for p in near_plants:
                if not p.alive or self.size <= EAT_RATIO * p.size: continue
                d = fast_hypot(p.x - self.x, p.y - self.y)
                if d < near_d: nearest, near_d = p, d
            if nearest:
                self.angle = fast_atan2(nearest.y - self.y, nearest.x - self.x)

        self.x = clamp(self.x + math.cos(self.angle) * self.speed, 8, MAP_W - 8)
        self.y = clamp(self.y + math.sin(self.angle) * self.speed, 8, MAP_H - 8)

        # 식물 섭취
        if self.diet_gene < 0.9:
            nearby_p = plant_tree.query(self.x, self.y, self.size + 30)
            for p in nearby_p:
                if not p.alive: continue
                if fast_hypot(p.x - self.x, p.y - self.y) < self.size + p.size * 0.5 + 4 \
                   and self.size >= EAT_RATIO * p.size:
                    self.energy = min(self.max_energy, self.energy + p.energy_value() * self.herb_efficiency())
                    p.alive = False
                    self.eat_count += 1

        # 포식
        if prey and self.can_hunt(prey) and prey_d < self.hunt_range + self.size:
            prey.alive = False
            self.energy = min(self.max_energy, self.energy + 5 * prey.size ** 2 * self.hunt_efficiency())
            self.kill_count += 1

        if self.energy <= 0 or self.age >= self.maxage:
            self.alive = False

    def try_split(self, mut_rate=9):
        if self.energy >= self.split_threshold and self.splitcool == 0:
            self.splitcool = int(3 * self.size ** 2)
            self.energy    = 0.5 * self.max_energy
            base_pct       = 0.20 * (mut_rate / 9.0)

            def mutate_pct(val, lo, hi):
                return clamp(val * rnd(1.0 - base_pct, 1.0 + base_pct), lo, hi)

            new_size  = mutate_pct(self.size,  MIN_SIZE, MAX_SIZE)
            new_speed = mutate_pct(self.speed, 0.3,      MAX_SPEED)
            new_diet  = clamp(self.diet_gene + rnd(-base_pct * 0.6, base_pct * 0.6), 0.0, 1.0)
            new_hue   = (self.hue + rnd(-base_pct * 60, base_pct * 60)) % 360

            child = Organism(
                x=clamp(self.x + rnd(-5*self.size, 5*self.size), 8, MAP_W - 8),
                y=clamp(self.y + rnd(-5*self.size, 5*self.size), 8, MAP_H - 8),
                speed=new_speed, size=new_size, hue=new_hue, diet_gene=new_diet,
            )
            child.energy = self.energy
            return child
        return None

    def get_body_color(self):
        if self._body_color is None:
            self._body_color = hsl_to_rgb(self.hue, 0.72, 0.45)
        return self._body_color
