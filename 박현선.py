import pygame, sys, time, math
from 선성준 import (
    clamp, rnd, hsl_to_rgb,
    WHITE, RED, GOLD, GREEN, BLUE,
    MAX_SPEED, MAX_SIZE,
)

# layout 상수
WIN_W  = 1100
WIN_H  = 560
CHART_H = 160
UI_H   = 56
TOTAL_H = WIN_H + CHART_H + UI_H


class InfoPanel:
    """선택된 개체의 상세 정보를 패널로 표시."""

    def __init__(self):
        self.target = None

    def set_target(self, org):
        self.target = org

    def clear(self):
        self.target = None

    def draw(self, screen, font, font_s, sw, sim_h):
        o = self.target
        if o is None or not o.alive:
            self.target = None
            return

        pw, ph = 240, 220
        px = sw - pw - 10
        py = 10
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(panel, (18, 16, 28, 220), (0, 0, pw, ph), border_radius=10)
        pygame.draw.rect(panel, (*GOLD, 180),       (0, 0, pw, ph), 1, border_radius=10)
        screen.blit(panel, (px, py))

        col = o.get_body_color()
        cr  = max(5, int(o.size * 0.9))
        pygame.draw.circle(screen, col, (px + 20, py + 20), cr)

        if   o.diet_gene >= 0.65: pygame.draw.circle(screen, RED,   (px+20, py+20), cr, 2)
        elif o.diet_gene <= 0.35: pygame.draw.circle(screen, GREEN, (px+20, py+20), cr, 2)

        lines = [
            ("▶ 추적 중",                                      GOLD),
            (f"크기:   {o.size:.1f}",                          WHITE),
            (f"속도:   {o.speed:.2f}",                         WHITE),
            (f"식성:   {o.diet_label} ({o.diet_gene:.2f})",    WHITE),
            (f"에너지: {o.energy:.1f} / {o.max_energy:.0f}",   WHITE),
            (f"포식:   {o.kill_count}회  채식: {o.eat_count}회", WHITE),
            (f"수명:   {o.age} / {o.maxage:.0f}틱",            WHITE),
            (f"분열 쿨타임: {o.splitcool}틱",                  WHITE),
            ("[클릭] 추적 해제",                               (100, 98, 90)),
        ]
        for i, (txt, color) in enumerate(lines):
            t = font_s.render(txt, True, color)
            screen.blit(t, (px + 36 if i == 0 else px + 8, py + 6 + i * 22))


class UISlider:
    """돌연변이율 등 단순 수치를 조절하는 슬라이더."""

    def __init__(self, label, val, minv, maxv, x, y, w, is_float=False):
        self.label    = label
        self.val      = val
        self.minv     = minv
        self.maxv     = maxv
        self.x        = x
        self.y        = y
        self.w        = w
        self.is_float = is_float
        self.dragging = False

    def draw(self, surf, font_s):
        pygame.draw.rect(surf, (60, 58, 55), (self.x, self.y + 10, self.w, 5), border_radius=2)
        t  = (self.val - self.minv) / (self.maxv - self.minv)
        kx = int(self.x + t * self.w)
        pygame.draw.circle(surf, GOLD, (kx, self.y + 12), 7)
        pct  = 0.20 * (self.val / 9.0) * 100
        vstr = f"{self.val:.1f}(±{pct:.0f}%)" if self.is_float else str(int(self.val))
        lbl  = font_s.render(f"{self.label}:{vstr}", True, WHITE)
        surf.blit(lbl, (self.x, self.y - 1))

    def handle(self, event, offset_y=0):
        pos = (event.pos[0], event.pos[1] - offset_y) if hasattr(event, 'pos') else None
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, 'button', None) == 1 and pos:
            ax, ay = pos
            if abs(ay - (self.y + 12)) < 12:
                t = clamp((ax - self.x) / self.w, 0, 1)
                self.val = self.minv + t * (self.maxv - self.minv)
                if not self.is_float: self.val = round(self.val)
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging and pos:
            t = clamp((pos[0] - self.x) / self.w, 0, 1)
            self.val = self.minv + t * (self.maxv - self.minv)
            if not self.is_float: self.val = round(self.val)


class StartScreen:
    """시뮬레이션 시작 전 설정 화면."""

    def __init__(self, screen, font_b, font, font_s):
        self.screen = screen
        self.font_b = font_b
        self.font   = font
        self.font_s = font_s
        self.init_pop = 45
        self.init_mut = 9

        self.food_text  = "80"
        self.map_w_text = "4000"
        self.map_h_text = "2000"

        self.active_input = None
        self.input_rects  = {}
        self.dragging     = None

        pygame.display.set_caption("자연선택 시뮬레이션")

    # ── 내부 그리기 헬퍼 ────────────────────────────────
    def _slider(self, surf, label, val, minv, maxv, x, y, w, is_float=True):
        pygame.draw.rect(surf, (60, 58, 55), (x, y + 12, w, 6), border_radius=3)
        tv = (val - minv) / (maxv - minv)
        kx = int(x + tv * w)
        pygame.draw.circle(surf, GOLD, (kx, y + 15), 9)
        vstr = f"{val:.1f}" if is_float else str(int(val))
        lbl  = self.font.render(f"{label}: {vstr}", True, WHITE)
        surf.blit(lbl, (x, y - 2))
        return pygame.Rect(x - 4, y + 6, w + 8, 20), kx

    def _draw_input_box(self, label, text, is_active, x, y, w):
        rect       = pygame.Rect(x, y, w, 32)
        bg_col     = (60, 60, 65)    if is_active else (40, 40, 45)
        border_col = GOLD            if is_active else (100, 100, 100)
        pygame.draw.rect(self.screen, bg_col,     rect, border_radius=4)
        pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=4)
        lbl = self.font.render(f"{label} : {text}", True, WHITE)
        self.screen.blit(lbl, (rect.x + 10, rect.y + 6))
        if is_active and time.time() % 1 > 0.5:
            cx = rect.x + 12 + lbl.get_width()
            pygame.draw.line(self.screen, WHITE, (cx, rect.y + 6), (cx, rect.y + 26), 2)
        return rect

    # ── 메인 루프 ────────────────────────────────────────
    def run(self):
        clock     = pygame.time.Clock()
        particles = [(rnd(0, WIN_W), rnd(0, TOTAL_H), rnd(0.2, 1.0), rnd(0, 360))
                     for _ in range(60)]

        while True:
            sw, sh = self.screen.get_size()
            self.screen.fill((18, 17, 22))

            # 배경 파티클
            for i, (px, py, spd, hue) in enumerate(particles):
                py = (py - spd) % sh
                particles[i] = (px, py, spd, hue)
                pygame.draw.circle(self.screen, hsl_to_rgb(hue, 0.7, 0.5),
                                   (int(px), int(py)), max(2, int(spd * 3)))

            # 타이틀
            title = self.font_b.render("자연선택 시뮬레이션", True, (220, 215, 160))
            self.screen.blit(title, (sw // 2 - title.get_width() // 2, 26))

            # 패널
            panel_w, panel_h = 460, 370
            panel_x = sw // 2 - panel_w // 2
            panel_y = 70
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pygame.draw.rect(panel, (30, 28, 36, 220), (0, 0, panel_w, panel_h), border_radius=14)
            pygame.draw.rect(panel, (80, 78, 70, 160), (0, 0, panel_w, panel_h), 2, border_radius=14)
            self.screen.blit(panel, (panel_x, panel_y))

            inner_x = panel_x + 30
            sl_w    = panel_w - 60

            r_pop, _ = self._slider(self.screen, "시작 개체수",
                                    self.init_pop, 10, 200,
                                    inner_x, panel_y + 24, sl_w, is_float=False)
            r_mut, _ = self._slider(self.screen,
                                    f"돌연변이율 (±{(0.20*(self.init_mut/9.0))*100:.0f}%)",
                                    self.init_mut, 1, 30,
                                    inner_x, panel_y + 82, sl_w)

            self.input_rects['food']  = self._draw_input_box(
                "초기 식물량",  self.food_text,  self.active_input == 'food',
                inner_x, panel_y + 140, sl_w)
            self.input_rects['map_w'] = self._draw_input_box(
                "맵 가로 크기", self.map_w_text, self.active_input == 'map_w',
                inner_x, panel_y + 198, sl_w)
            self.input_rects['map_h'] = self._draw_input_box(
                "맵 세로 크기", self.map_h_text, self.active_input == 'map_h',
                inner_x, panel_y + 256, sl_w)

            # 시작 버튼
            btn_w = 200; btn_h = 44
            btn_x = sw // 2 - btn_w // 2
            btn_y = panel_y + panel_h + 30
            mx, my = pygame.mouse.get_pos()
            hover  = btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h
            pygame.draw.rect(self.screen,
                             (100, 200, 100) if hover else (60, 150, 70),
                             (btn_x, btn_y, btn_w, btn_h), border_radius=10)
            btn_lbl = self.font_b.render("▶  시뮬레이션 시작", True, (10, 10, 10))
            self.screen.blit(btn_lbl,
                             (btn_x + btn_w//2 - btn_lbl.get_width()//2,
                              btn_y + btn_h//2 - btn_lbl.get_height()//2))

            pygame.display.flip()

            # ── 이벤트 처리 ──────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN:
                    if self.active_input:
                        key = self.active_input
                        if event.key == pygame.K_BACKSPACE:
                            if key == 'food':  self.food_text  = self.food_text[:-1]
                            elif key == 'map_w': self.map_w_text = self.map_w_text[:-1]
                            elif key == 'map_h': self.map_h_text = self.map_h_text[:-1]
                        elif event.unicode.isnumeric():
                            if key == 'food':  self.food_text  += event.unicode
                            elif key == 'map_w': self.map_w_text += event.unicode
                            elif key == 'map_h': self.map_h_text += event.unicode
                    if event.key == pygame.K_RETURN:
                        return self._collect_result()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 입력 박스
                    hit = False
                    for k, rect in self.input_rects.items():
                        if rect.collidepoint(event.pos):
                            self.active_input = k; hit = True; break
                    if not hit:
                        self.active_input = None

                    # 시작 버튼
                    if btn_x <= event.pos[0] <= btn_x + btn_w and \
                       btn_y <= event.pos[1] <= btn_y + btn_h:
                        return self._collect_result()

                    # 슬라이더
                    if r_pop.collidepoint(event.pos):
                        self.dragging = ('pop', inner_x, sl_w)
                    elif r_mut.collidepoint(event.pos):
                        self.dragging = ('mut', inner_x, sl_w)

                if event.type == pygame.MOUSEBUTTONUP:
                    self.dragging = None

                if event.type == pygame.MOUSEMOTION and self.dragging:
                    key, sx0, sw2 = self.dragging
                    t = clamp((event.pos[0] - sx0) / sw2, 0, 1)
                    if key == 'pop': self.init_pop = int(10 + t * 190)
                    elif key == 'mut': self.init_mut = round(1 + t * 29, 1)

            clock.tick(60)

    def _collect_result(self):
        final_food = int(self.food_text)  if self.food_text.strip()  else 80
        final_w    = int(self.map_w_text) if self.map_w_text.strip() else 4000
        final_h    = int(self.map_h_text) if self.map_h_text.strip() else 2000
        return self.init_pop, self.init_mut, final_food, final_w, final_h, []


class Ending:
    """전멸 엔딩 화면."""

    def __init__(self, screen, tick):
        self.screen = screen
        self.clock  = pygame.time.Clock()
        font = pygame.font.Font(None, 40)
        self.text_surface = font.render(
            f"Ending: You played {tick} ticks", True, (255, 255, 255))
        pygame.display.set_caption("엔딩")

    def run(self):
        FPS = 60
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:     return "r"
                    if event.key == pygame.K_ESCAPE: return "esc"

            sw, sh = self.screen.get_size()
            self.screen.fill((10, 10, 15))
            self.screen.blit(self.text_surface,
                             (sw // 2 - self.text_surface.get_width() // 2,
                              sh // 2 - self.text_surface.get_height() // 2))
            pygame.display.flip()
            self.clock.tick(FPS)
