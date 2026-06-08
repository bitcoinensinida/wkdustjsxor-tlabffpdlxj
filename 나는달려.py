"""
자연선택 시뮬레이션 v9 (뭐 이상한거 많이 패치함)
pip install pygame numpy scipy numba
"""

import pygame, random, math, sys, os, time
import numpy as np
from scipy.spatial import cKDTree
from numba import njit

# ── 상수 ─────────────────────────────────────────────────
MAP_W, MAP_H   = 4000, 2000

WIN_W, WIN_H   = 1100, 560
CHART_H        = 160
UI_H           = 56
TOTAL_H        = WIN_H + CHART_H + UI_H

FPS            = 60
SPLIT_BASE     = 0.8
HUNT_RATIO     = 1.30
EAT_RATIO      = 1
MAX_SPEED      = 20.0
MAX_SIZE       = 40.0
MIN_SIZE       = 0.0

MAX_ORGANISMS  = 9999999
RESPAWN_TICK   = 30
HATCH_PERCENT  = 20

BG       = (245, 243, 236)
CHART_BG = (22, 22, 28)
UI_BG    = (40, 39, 37)
GRID_COL = (215, 213, 206)
WHITE    = (235, 233, 228)
RED      = (220, 60, 40)
GOLD     = (255, 210, 50)
BLUE     = (80, 140, 220)
GREEN    = (60, 200, 80)

# ── 가속화된 수학 함수 (Numba JIT) ───────────────────────
@njit(fastmath=True)
def fast_hypot(dx, dy):#피타고라스
    return math.sqrt(dx*dx + dy*dy)

@njit(fastmath=True)
def fast_atan2(dy, dx):#각도 계산
    return math.atan2(dy, dx)

# ── 유틸 ─────────────────────────────────────────────────
def clamp(v, lo, hi): return max(lo, min(hi, v))
def rnd(a, b):        return random.uniform(a, b)#a,b사이 값 만듬

def hsl_to_rgb(h, s, l):#색상 채도 명도를 rgb로 변환해주는 코드
    h /= 360
    if s == 0:
        v = int(l * 255); return (v, v, v)
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

def draw_star(surface, color, cx, cy, r, width=0):#다각형그리기를 이용해서 별을 그려주는 코드
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

    def query(self, x, y, radius):#(x,y)좌표에서 radius거리 내에 있는 것들을 구함
        if not self.tree: return []
        indices = self.tree.query_ball_point((x, y), radius)
        return [self.objects[i] for i in indices]

# ── 식물 ─────────────────────────────────────────────────
PLANT_TIERS = [
    ('small',  0.90, (0,   3.0), (50,  100, 40)),
    ('medium', 0.07, (3,  5.0), (70,  150, 55)),
    ('large',  0.02, (5,  10.0), (40,  120, 35)),
    ('giant',  0.01, (10.0,15.0), (25,   80, 20)),
]

class Plant:
    __slots__ = ('x','y','tier','size','base_color','color','alive')
    def __init__(self, tier=None):
        self.x, self.y = rnd(10, MAP_W - 10), rnd(10, MAP_H - 10)#식물의 위치 설정
        self.alive = True
        if tier is None:#tier을 안주면 확률에 맞게 tier을 정함
            r = random.random(); cum = 0
            for name, prob, _, _ in PLANT_TIERS:
                cum += prob
                if r <= cum: tier = name; break
            else: tier = 'medium'
        for name, prob, size_rng, col in PLANT_TIERS:#위에서 정한 tier값을 가지고 크기,색을 정함
            if name == tier:
                self.tier       = name
                self.size       = rnd(*size_rng)
                self.color = col; break

    def energy_value(self): return self.size ** 2#식물의 에너지를 반환
    
    def draw(self, surf, ox, oy, zoom, sim_w, sim_h):#식물을 그림
        sx = int((self.x - ox) * zoom)
        sy = int((self.y - oy) * zoom)
        r  = max(1, int(self.size * 0.75 * zoom))
        
        if sx < -r or sy < -r or sx > sim_w + r or sy > sim_h + r:#식물이 시야밖에 있으면 안그림
            return
            
        if self.size >= 11.0:#크기가 11이상이면 별로 그림
            draw_star(surf, self.color, sx, sy, r)
        elif self.size >= 6.5:#크기가 6.5이상이면 테두리,속을 따로 만듬
            pygame.draw.circle(surf, self.color, (sx, sy), r)
            pygame.draw.circle(surf,
                (min(self.color[0]+40, 255), min(self.color[1]+40, 255), self.color[2]),
                (sx, sy), max(1, r - 2))
        else:#아니면 그냥 원(거의 점이니까)
            pygame.draw.circle(surf, self.color, (sx, sy), r)

def _tier_targets(total):#total개에서 PLANT_TIERS에서 정한 비율만큼 식물 종류,개수 전달
    targets   = {}; remaining = total
    for i, (name, prob, _, _) in enumerate(PLANT_TIERS):
        n = round(total * prob) if i < len(PLANT_TIERS) - 1 else remaining
        targets[name] = max(0, n); remaining -= targets[name]
    return targets

def make_plants_by_ratio(total):#전달받은 종류,개수가지고 식물 만듬
    plants = []
    for name, n in _tier_targets(total).items():
        for _ in range(n): plants.append(Plant(tier=name))
    return plants

def replenish_plants_by_ratio(plants, target):#식물 재생성 함수
    if len(plants) >= target: return plants#식물수가 목표수보다 많으면 그대로 보냄
    current = {name: 0 for name, *_ in PLANT_TIERS}#current딕셔너리에 4가지 종류넣음
    for p in plants: current[p.tier] = current.get(p.tier, 0) + 1#실제 식물을 가지고 current딕셔너리에 현재 식물 개수적음
    targets = _tier_targets(target)#함수로 목표치 가져옴
    deficit = target - len(plants)#부족한 개수
    added   = 0#확인용 변수
    for name, tgt in targets.items():
        need  = tgt - current.get(name, 0)
        if need <= 0: continue
        batch = max(1, int(need // (100/HATCH_PERCENT)))
        for _ in range(min(batch, deficit - added)):#부족한 개수에서 HATCH_PERCENT%만큼 추가함
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
        self.speed     = clamp(speed if speed is not None else rnd(1.0, 4.5), 0.3, MAX_SPEED)
        self.diet_gene = diet_gene if diet_gene is not None else rnd(0.0, 1)
        self.max_energy = 2 * self.size ** 2#최대에너지 설정
        self.energy     = self.max_energy * 0.5#생성됬을때 에너지
        self.angle      = rnd(0, math.pi * 2)#랜덤으로 방향 설정
        self.alive      = True#이게 살아있는지 아닌지 알려주는 거임
        self.age        = 0#나이, 이거가지고 통계값이나 자연사계산할때 씀
        self.maxage     = 3000 * self.size ** 0.5 * (1/self.speed)#수명 구하기
        self.splitcool  = 0#이거가지고 분열 쿨타임 만듬
        self.kill_count = 0#포식 통게시 사용
        self.eat_count  = 0#식물 먹은 통계시 사용
        self._body_color     = None
        self._surf_cache_key = None
        self._cached_surf    = None

    @property
    def is_carnivore(self): return self.diet_gene >= 0.6#육식 계산시 사용
    @property
    def is_herbivore(self): return self.diet_gene <= 0.4#초식 계산시 사용

    @property
    def diet_label(self):#통계낼때 식성가지고 육식~초식 구분
        dg = self.diet_gene
        if dg >= 0.75: return "육식"
        if dg >= 0.55: return "준육식"
        if dg >= 0.45: return "잡식"
        if dg >= 0.25: return "준초식"
        return "초식"

    def hunt_efficiency(self): return 0.3 + 0.7 * self.diet_gene #식성에 따른 에너지 효율
    def herb_efficiency(self): return 0.3 + 0.7 * (1.0 - self.diet_gene)

    @property
    def split_threshold(self): return SPLIT_BASE * self.max_energy #분열시 필요한 에너지량
    @property
    def sight_range(self): return 80 + self.size * 20 #시야범위
    @property
    def hunt_range(self):  return 10 + self.size #사냥범위
    @property
    def flee_range(self):  return self.sight_range * 0.3 #도망범위

    def _drain(self): return self.speed * (self.size ** 1.5) * 0.003 #개체의 에너지 소모량
    def can_hunt(self, other): return self.size >= other.size * HUNT_RATIO #개체의 크기를 가지고 먹을수있는지 구함

    def update(self, plant_tree: KDTreeGrid, org_tree: KDTreeGrid):
        if self.splitcool > 0: self.splitcool -= 1 #분열쿨이 0보다 크다면 1감소하고 나이 1먹고 에너지를 소모량만큼 줄임
        self.age    += 1
        self.energy -= self._drain()

        sr = self.sight_range; fr = self.flee_range
        neighbors = org_tree.query(self.x, self.y, sr)#시야범위 내에 있는 생명체

        threat   = None; threat_d = fr
        prey     = None; prey_d   = sr * (0.2 + 0.8 * self.diet_gene)#육식성이 높을 수록 시야범위가 큼

        for o in neighbors: #개체 탐색해서 먹이,포식자로 분류함
            if o is self or not o.alive: continue#개체가 자신이거나 죽었으면 넘김
            d = fast_hypot(o.x - self.x, o.y - self.y)#피타고라스로 거리 잼
            if o.can_hunt(self) and d < threat_d and self.energy >= self.max_energy * 0.2:#개체가 자신을 사냥할수있고 에너지가 20%보다 높으면 도망침
                threat_d, threat = d, o
            if self.can_hunt(o) and d < prey_d and (self.energy >= self.max_energy * 0.2 and self.diet_gene < 0.9):#자신이 개체를 사냥할수있고 에너지가 20%보다 높으면 쫓아감
                prey_d, prey = d, o

        if threat:#포식자 있으면 도주
            self.angle = fast_atan2(self.y - threat.y, self.x - threat.x)
        elif prey:#먹이 있으면 추적
            self.angle = fast_atan2(prey.y - self.y, prey.x - self.x)
        elif self.diet_gene < 0.9:#둘다없고 극성 육식이 아니면 주변 식물로 향함
            near_plants = plant_tree.query(self.x, self.y, sr)#시야 범위 내에 있는 식물을 찾음
            nearest = None; near_d = 99999
            for p in near_plants:
                if not p.alive or self.size <= EAT_RATIO * p.size: continue#식물이 살아있지 않거나 먹을수없는 사이즈면 넘김
                d = fast_hypot(p.x - self.x, p.y - self.y)
                if d < near_d: nearest, near_d = p, d
            if nearest:
                self.angle = fast_atan2(nearest.y - self.y, nearest.x - self.x)

        spd = self.speed

        self.x = clamp(self.x + math.cos(self.angle) * spd, 8, MAP_W - 8)#앞에서 구한 방향,속도가지고 이동
        self.y = clamp(self.y + math.sin(self.angle) * spd, 8, MAP_H - 8)

        # 식물 섭취
        if self.diet_gene < 0.9:#극성 육식이 아니면 발동 식물이 범위내에 있으면 먹음
            nearby_p = plant_tree.query(self.x, self.y, self.size + 30)
            for p in nearby_p:
                if not p.alive: continue
                if fast_hypot(p.x - self.x, p.y - self.y) < self.size + p.size * 0.5 + 4 \
                   and self.size >= EAT_RATIO * p.size:
                    self.energy = min(self.max_energy, self.energy + p.energy_value() * self.herb_efficiency())#만약에 식물먹은뒤 에너지가 최대에너지보다 높으면 최대에너지로 정함
                    p.alive = False 
                    self.eat_count += 1

        # 포식
        if prey and self.can_hunt(prey) and prey_d < self.hunt_range + self.size:#자신이 해당개체를 먹을수있고 범위내에 있으면 먹음
            prey.alive = False
            self.energy = min(self.max_energy, self.energy + 5 * prey.size ** 2 * self.hunt_efficiency())#포식자를 버프하기 위해서 동물은 같은 크기의 식물보다 5배의 에너지량을 가짐
            self.kill_count += 1

        if self.energy <= 0 or self.age >= self.maxage: self.alive = False#에너지량이 0보다 작거나 나이가 수명보다 많으면 죽음

    def try_split(self, mut_rate=9):#분열시도!
        if self.energy >= self.split_threshold and self.splitcool == 0:#에너지량이 분열시 요구 에너지량보다 높고, 분열 쿨타임보다 길면 발동
            self.splitcool = int(3 * self.size ** 2)
            self.energy = 0.5 * self.max_energy#분열후 자신의 에너지는 생성시 에너지와 같음
            base_pct = 0.20 * (mut_rate / 9.0)#돌연변이 율이라는게 그 확률에 걸리면 돌연변이가 발생하는게 아니라 돌연변이율이 클수록 자식이 부모와 크게 다른거임

            def mutate_pct(val, lo, hi):#돌연변이 시키고 최대최소안에 없으면 최대나 최소로 만듬
                return clamp(val * rnd(1.0 - base_pct, 1.0 + base_pct), lo, hi)

            new_size  = mutate_pct(self.size,  MIN_SIZE, MAX_SIZE)
            new_speed = mutate_pct(self.speed, 0.3,      MAX_SPEED)
            new_diet  = clamp(self.diet_gene + rnd(-base_pct * 0.6, base_pct * 0.6), 0.0, 1.0)#식성은 다른것보다 비교적 적게 변함
            new_hue   = (self.hue + rnd(-base_pct * 60, base_pct * 60)) % 360#랜덤하게 색상을 변경함

            child = Organism(
                x=clamp(self.x + rnd(-5*self.size, 5*self.size), 8, MAP_W - 8),
                y=clamp(self.y + rnd(-5*self.size, 5*self.size), 8, MAP_H - 8),
                speed=new_speed, size=new_size, hue=new_hue, diet_gene=new_diet,
            )#자식 개체 생성
            child.energy = self.energy#자식의 에너지는 부모의 분열후 에너지와 같음>>이것때문에 돌연변이율이 너무높으면 무한 분열하는 문제가 생김
            return child
        return None

    def get_body_color(self): #hsl 가지고 rgb값 만들어서 반환
        if self._body_color is None:
            self._body_color = hsl_to_rgb(self.hue, 0.72, 0.45)
        return self._body_color

# ── 패널 & UI ────────────────────────────────────────────────
class InfoPanel:#개체 통계보여주는 창
    def __init__(self): self.target = None
    def set_target(self, org): self.target = org
    def clear(self):           self.target = None
    
    def draw(self, screen, font, font_s, sw, sim_h):#정보들 그려줌
        o = self.target
        if o is None or not o.alive: self.target = None; return
        
        pw, ph = 240, 220; px = sw - pw - 10; py = 10
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(panel, (18, 16, 28, 220), (0, 0, pw, ph), border_radius=10)
        pygame.draw.rect(panel, (*GOLD, 180), (0, 0, pw, ph), 1, border_radius=10)
        screen.blit(panel, (px, py))
        
        col = o.get_body_color(); cr = max(5, int(o.size * 0.9))
        pygame.draw.circle(screen, col, (px + 20, py + 20), cr)
        
        if o.diet_gene >= 0.65: pygame.draw.circle(screen, RED, (px+20, py+20), cr, 2)
        elif o.diet_gene <= 0.35: pygame.draw.circle(screen, GREEN, (px+20, py+20), cr, 2)
        
        lines = [
            ("▶ 추적 중", GOLD),
            (f"크기:   {o.size:.1f}", WHITE),
            (f"속도:   {o.speed:.2f}", WHITE),
            (f"식성:   {o.diet_label} ({o.diet_gene:.2f})", WHITE),
            (f"에너지: {o.energy:.1f} / {o.max_energy:.0f}", WHITE),
            (f"포식:   {o.kill_count}회  채식: {o.eat_count}회", WHITE),
            (f"수명:   {o.age} / {o.maxage:.0f}틱", WHITE),
            (f"분열 쿨타임: {o.splitcool}틱", WHITE),
            ("[클릭] 추적 해제", (100, 98, 90)),
        ]
        
        for i, (txt, color) in enumerate(lines):#위에 lines를 가지고 통계를 적음
            t = font_s.render(txt, True, color)
            screen.blit(t, (px + 36 if i == 0 else px + 8, py + 6 + i * 22))

class UISlider:#돌연변이 슬라이더
    def __init__(self, label, val, minv, maxv, x, y, w, is_float=False):
        self.label   = label; self.val  = val
        self.minv    = minv;  self.maxv = maxv
        self.x       = x;     self.y    = y; self.w = w
        self.is_float = is_float; self.dragging = False

    def draw(self, surf, font_s):#슬라이더 그려줌
        pygame.draw.rect(surf, (60, 58, 55), (self.x, self.y + 10, self.w, 5), border_radius=2)
        t  = (self.val - self.minv) / (self.maxv - self.minv)
        kx = int(self.x + t * self.w)
        pygame.draw.circle(surf, GOLD, (kx, self.y + 12), 7)
        pct  = 0.20 * (self.val / 9.0) * 100
        vstr = f"{self.val:.1f}(±{pct:.0f}%)" if self.is_float else str(int(self.val))
        lbl  = font_s.render(f"{self.label}:{vstr}", True, WHITE)
        surf.blit(lbl, (self.x, self.y - 1))

    def handle(self, event, offset_y=0):#마우스로 슬라이더 움직일수있게해줌
        pos = (event.pos[0], event.pos[1] - offset_y) if hasattr(event, 'pos') else None
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, 'button', None) == 1 and pos:
            ax, ay = pos
            if abs(ay - (self.y + 12)) < 12:
                t = clamp((ax - self.x) / self.w, 0, 1)
                self.val = self.minv + t * (self.maxv - self.minv)
                if not self.is_float: self.val = round(self.val)
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP: self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging and pos:
            t = clamp((pos[0] - self.x) / self.w, 0, 1)
            self.val = self.minv + t * (self.maxv - self.minv)
            if not self.is_float: self.val = round(self.val)

class StartScreen:#시작화면
    def __init__(self, screen, font_b, font, font_s):
        self.screen = screen; self.font_b = font_b
        self.font   = font;   self.font_s = font_s
        self.init_pop = 45; self.init_mut = 9
        pygame.display.set_caption("자연선택 시뮬레이션")
        
        # 텍스트 입력 변수들
        self.food_text = "80"
        self.map_w_text = "4000"
        self.map_h_text = "2000"
        
        self.active_input = None 
        self.input_rects = {}
        
        self.dragging = None

    def _slider(self, surf, label, val, minv, maxv, x, y, w, is_float=True):#슬라이더를 만들어내는 함수
        pygame.draw.rect(surf, (60, 58, 55), (x, y + 12, w, 6), border_radius=3)
        tv = (val - minv) / (maxv - minv); kx = int(x + tv * w)
        pygame.draw.circle(surf, GOLD, (kx, y + 15), 9)
        vstr = f"{val:.1f}" if is_float else str(int(val))
        lbl  = self.font.render(f"{label}: {vstr}", True, WHITE)
        surf.blit(lbl, (x, y - 2))
        return pygame.Rect(x - 4, y + 6, w + 8, 20), kx

    def _draw_input_box(self, label, text, is_active, x, y, w):#입력창 만드는 함수
        rect = pygame.Rect(x, y, w, 32)
        bg_col = (60, 60, 65) if is_active else (40, 40, 45)
        border_col = GOLD if is_active else (100, 100, 100)
        
        pygame.draw.rect(self.screen, bg_col, rect, border_radius=4)
        pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=4)
        
        lbl = self.font.render(f"{label} : {text}", True, WHITE)
        self.screen.blit(lbl, (rect.x + 10, rect.y + 6))
        
        # 깜빡이는 커서
        if is_active and time.time() % 1 > 0.5:
            cx = rect.x + 12 + lbl.get_width()
            pygame.draw.line(self.screen, WHITE, (cx, rect.y + 6), (cx, rect.y + 26), 2)
            
        return rect

    def run(self):#시작화면 실행 코드
        clock = pygame.time.Clock()
        particles = [(rnd(0, WIN_W), rnd(0, TOTAL_H), rnd(0.2, 1.0), rnd(0, 360)) for _ in range(60)]
        while True:
            sw, sh = self.screen.get_size()
            self.screen.fill((18, 17, 22))
            for i, (px, py, spd, hue) in enumerate(particles):
                py = (py - spd) % sh; particles[i] = (px, py, spd, hue)
                pygame.draw.circle(self.screen, hsl_to_rgb(hue, 0.7, 0.5), (int(px), int(py)), max(2, int(spd * 3)))

            title = self.font_b.render("자연선택 시뮬레이션", True, (220, 215, 160))
            self.screen.blit(title, (sw // 2 - title.get_width() // 2, 26))

            panel_w, panel_h = 460, 370
            panel_x = sw // 2 - panel_w // 2; panel_y = 70
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pygame.draw.rect(panel, (30, 28, 36, 220), (0, 0, panel_w, panel_h), border_radius=14)
            pygame.draw.rect(panel, (80, 78, 70, 160), (0, 0, panel_w, panel_h), 2, border_radius=14)
            self.screen.blit(panel, (panel_x, panel_y))

            inner_x = panel_x + 30; sl_w = panel_w - 60
            
            # 슬라이더
            r_pop, _ = self._slider(self.screen, "시작 개체수", self.init_pop, 10, 200, inner_x, panel_y + 24, sl_w, is_float=False)
            r_mut, _ = self._slider(self.screen, f"돌연변이율 (±{(0.20*(self.init_mut/9.0))*100:.0f}%)", self.init_mut, 1, 30, inner_x, panel_y + 82, sl_w)
            
            # 텍스트 박스들
            self.input_rects['food'] = self._draw_input_box("초기 식물량", self.food_text, self.active_input == 'food', inner_x, panel_y + 140, sl_w)
            self.input_rects['map_w'] = self._draw_input_box("맵 가로 크기", self.map_w_text, self.active_input == 'map_w', inner_x, panel_y + 198, sl_w)
            self.input_rects['map_h'] = self._draw_input_box("맵 세로 크기", self.map_h_text, self.active_input == 'map_h', inner_x, panel_y + 256, sl_w)

            btn_w  = 200; btn_h = 44; btn_x = sw // 2 - btn_w // 2; btn_y = panel_y + panel_h + 30
            mx, my = pygame.mouse.get_pos()
            hover  = btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h
            pygame.draw.rect(self.screen, (100, 200, 100) if hover else (60, 150, 70), (btn_x, btn_y, btn_w, btn_h), border_radius=10)
            btn_lbl = self.font_b.render("▶  시뮬레이션 시작", True, (10, 10, 10))
            self.screen.blit(btn_lbl, (btn_x + btn_w // 2 - btn_lbl.get_width() // 2, btn_y + btn_h // 2 - btn_lbl.get_height() // 2))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                # 키보드 텍스트 입력
                if event.type == pygame.KEYDOWN:
                    if self.active_input:
                        if event.key == pygame.K_BACKSPACE:
                            if self.active_input == 'food': self.food_text = self.food_text[:-1]
                            elif self.active_input == 'map_w': self.map_w_text = self.map_w_text[:-1]
                            elif self.active_input == 'map_h': self.map_h_text = self.map_h_text[:-1]
                        elif event.unicode.isnumeric():
                            if self.active_input == 'food': self.food_text += event.unicode
                            elif self.active_input == 'map_w': self.map_w_text += event.unicode
                            elif self.active_input == 'map_h': self.map_h_text += event.unicode
                    
                    if event.key == pygame.K_RETURN:
                        final_food = int(self.food_text) if self.food_text.strip() else 80
                        final_w = int(self.map_w_text) if self.map_w_text.strip() else 4000
                        final_h = int(self.map_h_text) if self.map_h_text.strip() else 2000
                        return self.init_pop, self.init_mut, final_food, final_w, final_h, []
                
                # 마우스 클릭 처리
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 입력 박스 클릭 판별
                    hit_any_box = False
                    for key, rect in self.input_rects.items():
                        if rect.collidepoint(event.pos):
                            self.active_input = key
                            hit_any_box = True
                            break
                    if not hit_any_box:
                        self.active_input = None
                        
                    # 시작 버튼
                    if btn_x <= event.pos[0] <= btn_x + btn_w and btn_y <= event.pos[1] <= btn_y + btn_h:
                        final_food = int(self.food_text) if self.food_text.strip() else 80
                        final_w = int(self.map_w_text) if self.map_w_text.strip() else 4000
                        final_h = int(self.map_h_text) if self.map_h_text.strip() else 2000
                        return self.init_pop, self.init_mut, final_food, final_w, final_h, []
                    
                    # 슬라이더
                    if r_pop.collidepoint(event.pos): self.dragging = ('pop', inner_x, sl_w)
                    elif r_mut.collidepoint(event.pos): self.dragging = ('mut', inner_x, sl_w)
                
                if event.type == pygame.MOUSEBUTTONUP: self.dragging = None
                if event.type == pygame.MOUSEMOTION and self.dragging:
                    key, sx0, sw2 = self.dragging; t = clamp((event.pos[0] - sx0) / sw2, 0, 1)
                    if key == 'pop': self.init_pop = int(10 + t * 190)
                    elif key == 'mut': self.init_mut = round(1 + t * 29, 1)
            clock.tick(60)

class Ending:
    def __init__(self,tick):
        self.screen = pygame.display.set_mode((WIN_W, TOTAL_H), pygame.RESIZABLE)
        pygame.display.set_caption("엔딩")
        self.clock = pygame.time.Clock()
        font = pygame.font.Font(None,40)
        self.text_surface = font.render(f"Ending:You played {tick}ticks",True,(255,255,255))
    
    def run(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "r"
                if event.key == pygame.K_ESCAPE:
                    return "esc"
        self.screen.blit(self.text_surface,(WIN_W/2,TOTAL_H/2))
        pygame.display.flip()
        self.clock.tick(FPS)
# ── 시뮬레이션 ───────────────────────────────────────────
class Simulation:#시뮬레이터를 총괄하는 클래스
    SPEED_MULTS = [1, 2, 5, 10]#배속

    def __init__(self, init_pop, init_mut, init_food, custom_orgs=None):
        self.init_pop    = init_pop;   self.init_mut  = init_mut
        self.init_food   = init_food;  
        self.custom_orgs = custom_orgs or []
        self.fullscreen  = False
        self.screen = pygame.display.set_mode((WIN_W, TOTAL_H), pygame.RESIZABLE)
        pygame.display.set_caption("자연선택 시뮬레이션")
        self.clock = pygame.time.Clock()

        for fname in ("malgun gothic", "AppleGothic", "NanumGothic", "arial", None):
            try:
                self.font   = pygame.font.SysFont(fname, 14)
                self.font_b = pygame.font.SysFont(fname, 16, bold=True)
                self.font_s = pygame.font.SysFont(fname, 12)
                break
            except: continue

        self.cam_x = 0; self.cam_y = 0; self.zoom = 1.0
        self.speed_idx = 0; self.paused = False
        self.info_panel  = InfoPanel()
        self.sl_mut      = UISlider("돌연변이%", init_mut, 1, 30, 10, 32, 155, is_float=True)
        self._scatter_dots = []
        self._fps_history = []
        self._tick_ms = 0.0

        self.reset()

    def reset(self):#전체를 리셋함
        self.tick = 0
        self.organisms = [Organism() for _ in range(self.init_pop)]
        self.food_target = self.init_food 
        self.plants    = make_plants_by_ratio(self.food_target)
        self.ending    = False; self.paused = False
        self._scatter_dots = []
        self.info_panel.clear()

    def _screen_to_world(self, sx, sy):
        return sx / self.zoom + self.cam_x, sy / self.zoom + self.cam_y

    def _click_organism_sim(self, sx, sy, sim_h):
        if sy >= sim_h: return None
        wx, wy = self._screen_to_world(sx, sy)
        best = None; best_d = 30
        for o in self.organisms:
            if not o.alive: continue
            d = fast_hypot(o.x - wx, o.y - wy)
            if d < max(o.size, 8) and d < best_d: best_d = d; best = o
        return best

    def _click_organism_scatter(self, sx, sy, chart_y_offset):#10거리내에서 마우스에서 가장가까운 생명체 반환
        best = None; best_d = 10
        for (px, py, org) in self._scatter_dots:
            d = fast_hypot(sx - px, (sy - chart_y_offset) - py)
            if d < best_d: best_d = d; best = org
        return best

    def move_camera(self, keys, sim_w, sim_h):
        s = 12 / self.zoom
        max_cx = max(0, MAP_W - sim_w / self.zoom)
        max_cy = max(0, MAP_H - sim_h / self.zoom)
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: self.cam_x = max(0, self.cam_x - s)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.cam_x = min(max_cx, self.cam_x + s)
        if keys[pygame.K_UP]    or keys[pygame.K_w]: self.cam_y = max(0, self.cam_y - s)
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: self.cam_y = min(max_cy, self.cam_y + s)

    def adjust_zoom(self, delta, sim_w, sim_h):
        old_zoom = self.zoom; self.zoom = clamp(self.zoom + delta, 0.1, 4.0)
        cx = self.cam_x + sim_w / old_zoom / 2
        cy = self.cam_y + sim_h / old_zoom / 2
        self.cam_x = clamp(cx - sim_w / self.zoom / 2, 0, max(0, MAP_W - sim_w / self.zoom))
        self.cam_y = clamp(cy - sim_h / self.zoom / 2, 0, max(0, MAP_H - sim_h / self.zoom))

    def _auto_follow(self, sim_w, sim_h):#생명체 따라가는 함수
        o = self.info_panel.target
        if o and o.alive:
            self.cam_x = clamp(o.x - sim_w / self.zoom / 2, 0, max(0, MAP_W - sim_w / self.zoom))
            self.cam_y = clamp(o.y - sim_h / self.zoom / 2, 0, max(0, MAP_H - sim_h / self.zoom))

    def _make_speed_btn_rects(self, ui_w):#배속버튼 네모 생성 함수
        return [pygame.Rect(175 + i * 52, 4, 46, 22) for i in range(len(self.SPEED_MULTS))]

    def handle_ui_click(self, event, ui_y, ui_w):
        pos = (event.pos[0], event.pos[1] - ui_y)
        for i, rect in enumerate(self._make_speed_btn_rects(ui_w)):
            if rect.collidepoint(pos): self.speed_idx = i; return

    def handle_ui_event(self, event, ui_y):#돌연변이율 슬라이더 조절
        self.sl_mut.handle(event, ui_y)

    def update(self):
        self.tick += 1
        
        # 리스폰 틱마다 식물을 재생성
        if self.tick % RESPAWN_TICK == 0:
            self.plants = replenish_plants_by_ratio(self.plants, self.food_target)
            
        mut = self.sl_mut.val

        t0 = time.perf_counter()
        
        plant_tree = KDTreeGrid([p for p in self.plants if p.alive])#식물이랑 생물의 KDTree만듬
        org_tree   = KDTreeGrid([o for o in self.organisms if o.alive])

        children = []
        for o in self.organisms:
            if not o.alive: continue
            o.update(plant_tree, org_tree)
            c = o.try_split(mut)#분열이 가능하면 자식을 받음
            if c: children.append(c)#자식을 children리스트에 추기함

        self.plants = [p for p in self.plants if getattr(p, 'alive', True)]
        self.organisms = [o for o in self.organisms if o.alive] + children

        self._tick_ms = (time.perf_counter() - t0) * 1000

        if len(self.organisms) > MAX_ORGANISMS:
            random.shuffle(self.organisms)
            self.organisms = self.organisms[:MAX_ORGANISMS]

        if not self.organisms:
            self.ending = True

    def draw_sim(self, surf, sim_w, sim_h):#식물이랑 동물그려주는 함수
        surf.fill(BG); ox, oy = self.cam_x, self.cam_y; z = self.zoom
        
        gx0 = int(ox // 80) * 80; gy0 = int(oy // 80) * 80
        for gx in range(gx0, int(ox + sim_w / z) + 80, 80):
            lx = int((gx - ox) * z); pygame.draw.line(surf, GRID_COL, (lx, 0), (lx, sim_h))
        for gy in range(gy0, int(oy + sim_h / z) + 80, 80):
            ly = int((gy - oy) * z); pygame.draw.line(surf, GRID_COL, (0, ly), (sim_w, ly))
        
        bx0 = int(-ox * z); by0 = int(-oy * z)
        pygame.draw.rect(surf, (180, 120, 80), (bx0, by0, int(MAP_W * z), int(MAP_H * z)), 3)

        for p in self.plants: p.draw(surf, ox, oy, z, sim_w, sim_h)

        draw_list = []
        tracked = self.info_panel.target
        for o in self.organisms:
            sx = int((o.x - ox) * z); sy = int((o.y - oy) * z)
            r  = max(2, int(o.size * z))
            if sx < -r-8 or sy < -r-8 or sx > sim_w+r+8 or sy > sim_h+r+8: continue

            color = o.get_body_color()
            alpha = clamp(int(o.energy / o.max_energy * 220) + 40, 60, 255)
            dg = o.diet_gene
            selected = (o is tracked and o.alive)

            cache_key = (r, color, int(dg * 10), selected)
            if o._surf_cache_key == cache_key and not selected:
                s = o._cached_surf
                cr = r + 7
            else:
                sz = r * 2 + 14; cr = r + 7
                s  = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (cr, cr), r)
                if dg >= 0.65:
                    bw = min(int(dg * 3), 3)
                    pygame.draw.circle(s, (*RED, 200), (cr, cr), r, bw)
                elif dg <= 0.35:
                    bw = min(int((1 - dg) * 3), 3)
                    pygame.draw.circle(s, (*GREEN, 200), (cr, cr), r, bw)
                if selected:
                    pygame.draw.circle(s, (*GOLD, 230), (cr, cr), r + 4, 2)
                if not selected:
                    o._surf_cache_key = cache_key
                    o._cached_surf    = s
            
            draw_list.append((s, (sx - cr, sy - cr)))
        
        surf.blits(draw_list)

        if self.paused:
            pt = self.font_b.render("[ 일시정지 ]  SPACE 로 재개", True, GOLD)
            surf.blit(pt, (sim_w // 2 - pt.get_width() // 2, sim_h // 2 - 12))
        zt = self.font_s.render(f"줌 x{self.zoom:.1f}", True, (140, 138, 130))
        surf.blit(zt, (sim_w - zt.get_width() - 8, 4))

    def draw_scatter(self, surf, sw, sim_h_area):#그래프 그려주는 함수
        surf.fill(CHART_BG); alive = self.organisms
        self._scatter_dots = [] 
        
        if not alive: return
        pad = 36; cw = sw - pad * 2; ch = CHART_H - 26
        tracked = self.info_panel.target
        for o in alive:
            px2 = pad + int((o.speed / MAX_SPEED) * cw)
            py2 = 10  + int((1 - o.size / MAX_SIZE) * ch)
            col = o.get_body_color()
            
            self._scatter_dots.append((px2, py2, o))
            
            is_tracked = (o is tracked and o.alive)
            if is_tracked: pygame.draw.circle(surf, GOLD, (px2, py2), 6, 2)
            pygame.draw.circle(surf, col, (px2, py2), 1)

        pygame.draw.line(surf, (70, 70, 80), (pad, 10),    (pad, 10 + ch))
        pygame.draw.line(surf, (70, 70, 80), (pad, 10+ch), (pad + cw, 10 + ch))
        
        perf = self.font_s.render(f"틱:{self._tick_ms:.1f}ms  엔진: cKDTree + Numba  맵:{MAP_W}x{MAP_H}", True, (80, 80, 100))
        surf.blit(perf, (pad, 2))

    def draw_ui(self, surf, ui_w):
        surf.fill(UI_BG); n = len(self.organisms)
        avg_sp  = sum(o.speed for o in self.organisms) / n if n else 0
        avg_sz  = sum(o.size  for o in self.organisms) / n if n else 0
        carn    = sum(1 for o in self.organisms if o.is_carnivore)
        herb    = sum(1 for o in self.organisms if o.is_herbivore)
        stats   = [f"개체수:{n}", f"육식:{carn}", f"초식:{herb}",
                   f"평균속도:{avg_sp:.2f}", f"평균크기:{avg_sz:.2f}",
                   f"식물:{len(self.plants)}/{self.food_target}", f"틱:{self.tick}"]
        x = 380
        for s in stats:
            t = self.font_b.render(s, True, WHITE); surf.blit(t, (x, 4)); x += t.get_width() + 12

        btn_rects = self._make_speed_btn_rects(ui_w)
        for i, (rect, m) in enumerate(zip(btn_rects, self.SPEED_MULTS)):#배속 버튼을 그려줌
            active = (i == self.speed_idx); col = GOLD if active else (70, 68, 65)
            pygame.draw.rect(surf, col, rect, border_radius=4)
            pygame.draw.rect(surf, (100, 98, 94), rect, 1, border_radius=4)
            lbl = self.font_b.render(f"x{m}", True, (20, 18, 16) if active else WHITE)
            surf.blit(lbl, (rect.x + rect.w//2 - lbl.get_width()//2,
                            rect.y + rect.h//2 - lbl.get_height()//2))

        self.sl_mut.draw(surf, self.font_s)#돌연변이 슬라이더를 만들어줌

        pause_col = (200, 180, 40) if self.paused else (60, 60, 60)
        pygame.draw.rect(surf, pause_col, (ui_w - 210, 4, 100, 20), border_radius=3)
        surf.blit(self.font_s.render("SPACE 일시정지" if not self.paused else "재개: SPACE", True, WHITE), (ui_w - 208, 7))

        fp_rect = pygame.Rect(ui_w - 104, 32, 96, 18)
        pygame.draw.rect(surf, (50, 70, 50), fp_rect, border_radius=3)
        surf.blit(self.font_s.render(f"식물[+/-]:{self.food_target}", True, WHITE), (fp_rect.x + 4, fp_rect.y + 2))

        surf.blit(self.font_s.render(
            "[R]초기화  [WASD]카메라  [휠/Q·E]줌  [SPACE]정지  [클릭]추적",
            True, (130, 128, 122)), (175, 36))

    def run(self):
        while True:
            sw, sh       = self.screen.get_size()
            sim_h_area   = max(100, sh - CHART_H - UI_H)
            ui_y         = sim_h_area + CHART_H
            scatter_y    = sim_h_area
            
            sim_surf     = pygame.Surface((sw, sim_h_area))
            scatter_surf = pygame.Surface((sw, CHART_H))
            ui_surf      = pygame.Surface((sw, UI_H))
            keys         = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: return 'menu'
                    elif event.key == pygame.K_r: self.reset() # R키 초기화
                    elif event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_f):
                        self.food_target = min(1000, self.food_target + 10)
                    elif event.key in (pygame.K_MINUS, pygame.K_v):
                        self.food_target = max(10, self.food_target - 10)
                        self.plants = self.plants[:self.food_target]
                    elif event.key == pygame.K_1: self.speed_idx = 0
                    elif event.key == pygame.K_2: self.speed_idx = 1
                    elif event.key == pygame.K_3: self.speed_idx = 2
                    elif event.key == pygame.K_4: self.speed_idx = 3
                    elif event.key == pygame.K_q: self.adjust_zoom(-0.1, sw, sim_h_area)
                    elif event.key == pygame.K_e: self.adjust_zoom(+0.1, sw, sim_h_area)

                if event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if my < sim_h_area: self.adjust_zoom(event.y * 0.12, sw, sim_h_area)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if my < sim_h_area:
                        clicked = self._click_organism_sim(mx, my, sim_h_area)
                        if clicked:
                            if self.info_panel.target is clicked: self.info_panel.clear()
                            else: self.info_panel.set_target(clicked)
                        else: self.info_panel.clear()
                    elif scatter_y <= my < ui_y:
                        clicked = self._click_organism_scatter(mx, my, scatter_y)
                        if clicked:
                            if self.info_panel.target is clicked: self.info_panel.clear()
                            else: self.info_panel.set_target(clicked)
                    elif my >= ui_y:
                        self.handle_ui_click(event, ui_y, sw)
                        self.handle_ui_event(event, ui_y)

                if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
                    self.handle_ui_event(event, ui_y)

            if self.info_panel.target and self.info_panel.target.alive:
                self._auto_follow(sw, sim_h_area)
            else:
                self.move_camera(keys, sw, sim_h_area)

            if not self.ending and not self.paused:
                for _ in range(self.SPEED_MULTS[self.speed_idx]):
                    self.update()
                    if self.ending: break

            self.draw_sim(sim_surf, sw, sim_h_area)
            self.draw_scatter(scatter_surf, sw, sim_h_area)
            self.draw_ui(ui_surf, sw)
            self.screen.blit(sim_surf,     (0, 0))
            self.screen.blit(scatter_surf, (0, sim_h_area))
            self.screen.blit(ui_surf,      (0, ui_y))
            
            self.info_panel.draw(self.screen, self.font, self.font_s, sw, sim_h_area)

            fps = self.clock.get_fps()
            self._fps_history.append(fps)
            if len(self._fps_history) > 60: self._fps_history.pop(0)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            fps_t   = self.font_s.render(f"FPS {avg_fps:.0f}", True, GREEN if avg_fps >= 50 else RED)
            self.screen.blit(fps_t, (4, 4))

            if self.organisms == []:
                End = Ending(self.tick)
                a = End.run()
                if a == "esc":
                    return "menu"
                elif a == "r":
                    self.reset()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
    icon = pygame.image.load("안녕.png") #아이콘 지정
    pygame.display.set_icon(icon)

# ── 메인 ─────────────────────────────────────────────────
def main():
    global MAP_W, MAP_H 
    
    pygame.init()#창키기
    screen = pygame.display.set_mode((WIN_W, TOTAL_H), pygame.RESIZABLE)#스크린 만들고 자유롭게 조정가능케 함
    font_b = pygame.font.SysFont("malgun gothic", 16, bold=True)
    font   = pygame.font.SysFont("malgun gothic", 14)
    font_s = pygame.font.SysFont("malgun gothic", 12)

    while True:
        ss = StartScreen(screen, font_b, font, font_s)
        init_pop, init_mut, init_food, map_w, map_h, custom_orgs = ss.run()#시작화면에서 변수를 받아옴
        
        MAP_W, MAP_H = map_w, map_h
        
        sim = Simulation(init_pop, init_mut, init_food, custom_orgs)#받은 변수를 가지고 게임을 실행함
        if sim.run() == 'menu': continue#만약 esc를 눌렀다면 다시 시작화면으로 돌아감

if __name__ == "__main__":
    try:#노래실행
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bgm_path = os.path.join(current_dir, "bgm.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(bgm_path)
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"음악 파일 로드 실패: {e}")
        
    main()#게임 전체 함수
    #thanks for https://pgtd.tistory.com/251