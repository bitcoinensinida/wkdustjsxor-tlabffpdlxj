import pygame, random, sys, os, time, math

from 선성준 import (
    MAP_W, MAP_H,
    MAX_ORGANISMS, RESPAWN_TICK,
    MAX_SPEED, MAX_SIZE,
    BG, CHART_BG, UI_BG, GRID_COL, WHITE, RED, GOLD, GREEN,
    clamp, rnd, hsl_to_rgb,
    fast_hypot, fast_atan2,
    KDTreeGrid,
    Organism, Plant,
    make_plants_by_ratio, replenish_plants_by_ratio,
)
from 박현선 import (
    WIN_W, WIN_H, CHART_H, UI_H, TOTAL_H,
    InfoPanel, UISlider, StartScreen, Ending,
)

FPS = 60

# ── 전역 맵 크기 (StartScreen 에서 덮어씀) ─────────────
import 선성준 as _ent


class Simulation:
    SPEED_MULTS = [1, 2, 5, 10]

    def __init__(self, init_pop, init_mut, init_food, custom_orgs=None):
        self.init_pop    = init_pop
        self.init_mut    = init_mut
        self.init_food   = init_food
        self.custom_orgs = custom_orgs or []

        self.screen = pygame.display.set_mode((WIN_W, TOTAL_H), pygame.RESIZABLE)
        pygame.display.set_caption("자연선택 시뮬레이션")
        self.clock = pygame.time.Clock()

        # 폰트
        for fname in ("malgun gothic", "AppleGothic", "NanumGothic", "arial", None):
            try:
                self.font   = pygame.font.SysFont(fname, 14)
                self.font_b = pygame.font.SysFont(fname, 16, bold=True)
                self.font_s = pygame.font.SysFont(fname, 12)
                break
            except:
                continue

        self.cam_x = 0; self.cam_y = 0; self.zoom = 1.0
        self.speed_idx = 0; self.paused = False
        self.info_panel     = InfoPanel()
        self.sl_mut         = UISlider("돌연변이%", init_mut, 1, 30, 10, 32, 155, is_float=True)
        self._scatter_dots  = []
        self._fps_history   = []
        self._tick_ms       = 0.0

        self.reset()

    # ── 리셋 ─────────────────────────────────────────────
    def reset(self):
        self.tick        = 0
        self.organisms   = [Organism() for _ in range(self.init_pop)]
        self.food_target = self.init_food
        self.plants      = make_plants_by_ratio(self.food_target)
        self.ending      = False
        self.paused      = False
        self._scatter_dots = []
        self.info_panel.clear()

    # ── 좌표 변환 ─────────────────────────────────────────
    def _screen_to_world(self, sx, sy):
        return sx / self.zoom + self.cam_x, sy / self.zoom + self.cam_y

    def _click_organism_sim(self, sx, sy, sim_h):
        if sy >= sim_h: return None
        wx, wy = self._screen_to_world(sx, sy)
        best = None; best_d = 30
        for o in self.organisms:
            if not o.alive: continue
            d = fast_hypot(o.x - wx, o.y - wy)
            if d < max(o.size, 8) and d < best_d:
                best_d = d; best = o
        return best

    def _click_organism_scatter(self, sx, sy, chart_y_offset):
        best = None; best_d = 10
        for (px, py, org) in self._scatter_dots:
            d = fast_hypot(sx - px, (sy - chart_y_offset) - py)
            if d < best_d:
                best_d = d; best = org
        return best

    # ── 카메라 ────────────────────────────────────────────
    def move_camera(self, keys, sim_w, sim_h):
        s = 12 / self.zoom
        max_cx = max(0, _ent.MAP_W - sim_w / self.zoom)
        max_cy = max(0, _ent.MAP_H - sim_h / self.zoom)
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: self.cam_x = max(0, self.cam_x - s)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.cam_x = min(max_cx, self.cam_x + s)
        if keys[pygame.K_UP]    or keys[pygame.K_w]: self.cam_y = max(0, self.cam_y - s)
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: self.cam_y = min(max_cy, self.cam_y + s)

    def adjust_zoom(self, delta, sim_w, sim_h):
        old_zoom  = self.zoom
        self.zoom = clamp(self.zoom + delta, 0.1, 4.0)
        cx = self.cam_x + sim_w / old_zoom / 2
        cy = self.cam_y + sim_h / old_zoom / 2
        self.cam_x = clamp(cx - sim_w / self.zoom / 2, 0, max(0, _ent.MAP_W - sim_w / self.zoom))
        self.cam_y = clamp(cy - sim_h / self.zoom / 2, 0, max(0, _ent.MAP_H - sim_h / self.zoom))

    def _auto_follow(self, sim_w, sim_h):
        o = self.info_panel.target
        if o and o.alive:
            self.cam_x = clamp(o.x - sim_w / self.zoom / 2, 0, max(0, _ent.MAP_W - sim_w / self.zoom))
            self.cam_y = clamp(o.y - sim_h / self.zoom / 2, 0, max(0, _ent.MAP_H - sim_h / self.zoom))

    # ── UI 버튼 ───────────────────────────────────────────
    def _make_speed_btn_rects(self, ui_w):
        return [pygame.Rect(175 + i * 52, 4, 46, 22) for i in range(len(self.SPEED_MULTS))]

    def handle_ui_click(self, event, ui_y, ui_w):
        pos = (event.pos[0], event.pos[1] - ui_y)
        for i, rect in enumerate(self._make_speed_btn_rects(ui_w)):
            if rect.collidepoint(pos):
                self.speed_idx = i; return

    def handle_ui_event(self, event, ui_y):
        self.sl_mut.handle(event, ui_y)

    # ── 업데이트 ──────────────────────────────────────────
    def update(self):
        self.tick += 1
        if self.tick % RESPAWN_TICK == 0:
            self.plants = replenish_plants_by_ratio(self.plants, self.food_target)

        mut = self.sl_mut.val
        t0  = time.perf_counter()

        plant_tree = KDTreeGrid([p for p in self.plants    if p.alive])
        org_tree   = KDTreeGrid([o for o in self.organisms if o.alive])

        children = []
        for o in self.organisms:
            if not o.alive: continue
            o.update(plant_tree, org_tree)
            c = o.try_split(mut)
            if c: children.append(c)

        self.plants    = [p for p in self.plants    if getattr(p, 'alive', True)]
        self.organisms = [o for o in self.organisms if o.alive] + children
        self._tick_ms  = (time.perf_counter() - t0) * 1000

        if len(self.organisms) > MAX_ORGANISMS:
            random.shuffle(self.organisms)
            self.organisms = self.organisms[:MAX_ORGANISMS]

        if not self.organisms:
            self.ending = True

    # ── 그리기: 시뮬레이션 뷰 ────────────────────────────
    def draw_sim(self, surf, sim_w, sim_h):
        surf.fill(BG)
        ox, oy, z = self.cam_x, self.cam_y, self.zoom

        # 그리드
        gx0 = int(ox // 80) * 80; gy0 = int(oy // 80) * 80
        for gx in range(gx0, int(ox + sim_w / z) + 80, 80):
            lx = int((gx - ox) * z)
            pygame.draw.line(surf, GRID_COL, (lx, 0), (lx, sim_h))
        for gy in range(gy0, int(oy + sim_h / z) + 80, 80):
            ly = int((gy - oy) * z)
            pygame.draw.line(surf, GRID_COL, (0, ly), (sim_w, ly))

        # 맵 테두리
        bx0 = int(-ox * z); by0 = int(-oy * z)
        pygame.draw.rect(surf, (180, 120, 80),
                         (bx0, by0, int(_ent.MAP_W * z), int(_ent.MAP_H * z)), 3)

        for p in self.plants:
            p.draw(surf, ox, oy, z, sim_w, sim_h)

        draw_list = []
        tracked   = self.info_panel.target
        for o in self.organisms:
            sx = int((o.x - ox) * z); sy = int((o.y - oy) * z)
            r  = max(2, int(o.size * z))
            if sx < -r-8 or sy < -r-8 or sx > sim_w+r+8 or sy > sim_h+r+8: continue

            color    = o.get_body_color()
            alpha    = clamp(int(o.energy / o.max_energy * 220) + 40, 60, 255)
            dg       = o.diet_gene
            selected = (o is tracked and o.alive)

            cache_key = (r, color, int(dg * 10), selected)
            if o._surf_cache_key == cache_key and not selected:
                s  = o._cached_surf
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

    # ── 그리기: 산점도 ────────────────────────────────────
    def draw_scatter(self, surf, sw, sim_h_area):
        surf.fill(CHART_BG)
        alive = self.organisms
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
            if is_tracked:
                pygame.draw.circle(surf, GOLD, (px2, py2), 6, 2)
            pygame.draw.circle(surf, col, (px2, py2), 1)

        pygame.draw.line(surf, (70, 70, 80), (pad, 10),       (pad, 10 + ch))
        pygame.draw.line(surf, (70, 70, 80), (pad, 10 + ch),  (pad + cw, 10 + ch))

        perf = self.font_s.render(
            f"틱:{self._tick_ms:.1f}ms  엔진: cKDTree + Numba  맵:{_ent.MAP_W}x{_ent.MAP_H}",
            True, (80, 80, 100))
        surf.blit(perf, (pad, 2))

    # ── 그리기: UI 바 ─────────────────────────────────────
    def draw_ui(self, surf, ui_w):
        surf.fill(UI_BG)
        n      = len(self.organisms)
        avg_sp = sum(o.speed for o in self.organisms) / n if n else 0
        avg_sz = sum(o.size  for o in self.organisms) / n if n else 0
        carn   = sum(1 for o in self.organisms if o.is_carnivore)
        herb   = sum(1 for o in self.organisms if o.is_herbivore)
        stats  = [f"개체수:{n}", f"육식:{carn}", f"초식:{herb}",
                  f"평균속도:{avg_sp:.2f}", f"평균크기:{avg_sz:.2f}",
                  f"식물:{len(self.plants)}/{self.food_target}", f"틱:{self.tick}"]
        x = 380
        for s in stats:
            t = self.font_b.render(s, True, WHITE)
            surf.blit(t, (x, 4)); x += t.get_width() + 12

        btn_rects = self._make_speed_btn_rects(ui_w)
        for i, (rect, m) in enumerate(zip(btn_rects, self.SPEED_MULTS)):
            active = (i == self.speed_idx)
            col    = GOLD if active else (70, 68, 65)
            pygame.draw.rect(surf, col, rect, border_radius=4)
            pygame.draw.rect(surf, (100, 98, 94), rect, 1, border_radius=4)
            lbl = self.font_b.render(f"x{m}", True, (20, 18, 16) if active else WHITE)
            surf.blit(lbl, (rect.x + rect.w//2 - lbl.get_width()//2,
                            rect.y + rect.h//2 - lbl.get_height()//2))

        self.sl_mut.draw(surf, self.font_s)

        pause_col = (200, 180, 40) if self.paused else (60, 60, 60)
        pygame.draw.rect(surf, pause_col, (ui_w - 210, 4, 100, 20), border_radius=3)
        surf.blit(self.font_s.render(
            "SPACE 일시정지" if not self.paused else "재개: SPACE",
            True, WHITE), (ui_w - 208, 7))

        fp_rect = pygame.Rect(ui_w - 104, 32, 96, 18)
        pygame.draw.rect(surf, (50, 70, 50), fp_rect, border_radius=3)
        surf.blit(self.font_s.render(f"식물[+/-]:{self.food_target}", True, WHITE),
                  (fp_rect.x + 4, fp_rect.y + 2))

        surf.blit(self.font_s.render(
            "[R]초기화  [WASD]카메라  [휠/Q·E]줌  [SPACE]정지  [클릭]추적",
            True, (130, 128, 122)), (175, 36))

    # ── 메인 루프 ─────────────────────────────────────────
    def run(self):
        while True:
            sw, sh        = self.screen.get_size()
            sim_h_area    = max(100, sh - CHART_H - UI_H)
            ui_y          = sim_h_area + CHART_H
            scatter_y     = sim_h_area

            sim_surf      = pygame.Surface((sw, sim_h_area))
            scatter_surf  = pygame.Surface((sw, CHART_H))
            ui_surf       = pygame.Surface((sw, UI_H))
            keys          = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN:
                    if   event.key == pygame.K_ESCAPE: return 'menu'
                    elif event.key == pygame.K_r:      self.reset()
                    elif event.key == pygame.K_SPACE:  self.paused = not self.paused
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
                    if my < sim_h_area:
                        self.adjust_zoom(event.y * 0.12, sw, sim_h_area)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if my < sim_h_area:
                        clicked = self._click_organism_sim(mx, my, sim_h_area)
                        if clicked:
                            if self.info_panel.target is clicked: self.info_panel.clear()
                            else: self.info_panel.set_target(clicked)
                        else:
                            self.info_panel.clear()
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

            self.draw_sim(sim_surf,     sw, sim_h_area)
            self.draw_scatter(scatter_surf, sw, sim_h_area)
            self.draw_ui(ui_surf, sw)
            self.screen.blit(sim_surf,     (0, 0))
            self.screen.blit(scatter_surf, (0, sim_h_area))
            self.screen.blit(ui_surf,      (0, ui_y))
            self.info_panel.draw(self.screen, self.font, self.font_s, sw, sim_h_area)

            # FPS 표시
            fps = self.clock.get_fps()
            self._fps_history.append(fps)
            if len(self._fps_history) > 60: self._fps_history.pop(0)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            fps_t   = self.font_s.render(f"FPS {avg_fps:.0f}", True,
                                          GREEN if avg_fps >= 50 else RED)
            self.screen.blit(fps_t, (4, 4))

            # 엔딩 처리
            if not self.organisms:
                end = Ending(self.screen, self.tick)
                result = end.run()
                if result == "esc": return "menu"
                elif result == "r": self.reset()

            pygame.display.flip()
            self.clock.tick(FPS)


# ── 메인 ─────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, TOTAL_H), pygame.RESIZABLE)

    # 아이콘 (파일 없으면 무시)
    try:
        icon = pygame.image.load("안녕.png")
        pygame.display.set_icon(icon)
    except Exception:
        pass

    font_b = pygame.font.SysFont("malgun gothic", 16, bold=True)
    font   = pygame.font.SysFont("malgun gothic", 14)
    font_s = pygame.font.SysFont("malgun gothic", 12)

    while True:
        ss = StartScreen(screen, font_b, font, font_s)
        init_pop, init_mut, init_food, map_w, map_h, custom_orgs = ss.run()

        # 맵 크기를 entities 모듈에 반영
        _ent.MAP_W = map_w
        _ent.MAP_H = map_h

        sim = Simulation(init_pop, init_mut, init_food, custom_orgs)
        if sim.run() == 'menu':
            continue


if __name__ == "__main__":
    # BGM
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bgm_path    = os.path.join(current_dir, "bgm.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(bgm_path)
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"음악 파일 로드 실패: {e}")

    main()
