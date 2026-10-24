"""Interfaz Pygame de la Tarea B: pronóstico de demanda con LSTM.

Uso:
    python app.py
    python app.py --data data/sample_retail_sales.csv

Controles:
    - Botones de serie (izquierda): eligen la categoría a pronosticar.
    - Sliders: épocas, semanas de historia usadas y lookback (ventana de entrada).
    - "Entrenar": reentrena con los valores de los sliders.
    - "Media móvil": muestra/oculta el baseline.
    - "Exportar CSV": guarda las predicciones de la serie actual en outputs/.
"""
import argparse
from pathlib import Path

import pygame

import config
from data_prep import load_weekly_series
from forecasting import export_predictions, train_and_forecast

WIDTH, HEIGHT = 1200, 720
PANEL_W = 300

BG = (245, 246, 250)
PANEL = (255, 255, 255)
TEXT = (30, 33, 40)
MUTED = (110, 116, 128)
GRID = (225, 228, 235)
ACCENT = (37, 99, 235)
ACTUAL = (55, 65, 81)
LSTM_C = (234, 88, 12)
BASE_C = (22, 163, 74)
TEST_BG = (255, 247, 237)


class Button:
    def __init__(self, rect, label, toggle=False, active=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.toggle = toggle
        self.active = active

    def draw(self, surf, font):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.active:
            fill, fg = ACCENT, (255, 255, 255)
        else:
            fill, fg = ((232, 237, 250) if hover else (240, 242, 247)), TEXT
        pygame.draw.rect(surf, fill, self.rect, border_radius=6)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, step=1):
        self.track = pygame.Rect(x, y + 22, w, 6)
        self.label = label
        self.vmin, self.vmax, self.step = vmin, vmax, step
        self.value = value
        self.dragging = False

    def set_range(self, vmin, vmax):
        self.vmin, self.vmax = vmin, max(vmin, vmax)
        self.value = min(max(self.value, self.vmin), self.vmax)

    def _knob_x(self):
        span = self.vmax - self.vmin or 1
        return self.track.x + (self.value - self.vmin) / span * self.track.w

    def handle(self, event):
        """Devuelve True si el valor cambió."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.track.inflate(16, 24).collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            ratio = min(max((event.pos[0] - self.track.x) / self.track.w, 0), 1)
            raw = self.vmin + ratio * (self.vmax - self.vmin)
            new = int(round((raw - self.vmin) / self.step) * self.step + self.vmin)
            new = min(max(new, self.vmin), self.vmax)
            if new != self.value:
                self.value = new
                return True
        return False

    def draw(self, surf, font):
        surf.blit(font.render(f"{self.label}: {self.value}", True, TEXT),
                  (self.track.x, self.track.y - 22))
        pygame.draw.rect(surf, GRID, self.track, border_radius=3)
        filled = self.track.copy()
        filled.w = self._knob_x() - self.track.x
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        pygame.draw.circle(surf, ACCENT, (int(self._knob_x()), self.track.centery), 9)


def fmt_money(v):
    return f"${v:,.0f}"


class ForecastApp:
    def __init__(self, data_path):
        pygame.init()
        pygame.display.set_caption("UrbanStyle - Tarea B: Pronóstico de demanda (LSTM)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 15)
        self.small = pygame.font.SysFont("arial", 12)
        self.title_font = pygame.font.SysFont("arial", 20, bold=True)

        self.weekly = load_weekly_series(data_path)
        self.n_weeks = len(self.weekly)
        self.series_names = list(self.weekly.columns)
        self.selected = 0
        self.result = None
        self.status = ""
        self.dirty = False
        self.show_baseline = True

        x, w = 20, PANEL_W - 40
        y = 70
        self.series_buttons = []
        for i, name in enumerate(self.series_names):
            self.series_buttons.append(Button((x, y, w, 30), name, active=(i == 0)))
            y += 36
        y += 10
        self.epochs = Slider(x, y, w, "Épocas", 50, 1000, config.EPOCHS, step=50)
        y += 55
        self.lookback = Slider(x, y, w, "Lookback (semanas de entrada)", 2, 8, config.LOOKBACK)
        y += 55
        self.history = Slider(x, y, w, "Semanas de historia", 0, self.n_weeks, self.n_weeks)
        self._update_history_range()
        y += 65
        self.train_btn = Button((x, y, w, 36), "Entrenar")
        y += 44
        self.baseline_btn = Button((x, y, w, 30), "Media móvil", toggle=True, active=True)
        y += 38
        self.export_btn = Button((x, y, w, 30), "Exportar CSV")

        self.train()

    def _update_history_range(self):
        # Mínimo de semanas para tener al menos 8 muestras de entrenamiento
        min_hist = self.lookback.value + config.TEST_WEEKS + 8
        self.history.set_range(min(min_hist, self.n_weeks), self.n_weeks)

    def train(self):
        name = self.series_names[self.selected]
        self.status = f"Entrenando '{name}'..."
        self.draw()
        pygame.display.flip()
        try:
            self.result = train_and_forecast(
                self.weekly[name],
                lookback=self.lookback.value,
                epochs=self.epochs.value,
                history_weeks=self.history.value,
            )
            self.status = f"Listo: {name}"
        except ValueError as exc:
            self.result = None
            self.status = str(exc)
        self.dirty = False

    def export(self):
        if not self.result:
            return
        safe = self.result.series_name.replace(" - ", "_").replace(" ", "_")
        path = export_predictions([self.result],
                                  config.OUTPUT_DIR / f"predictions_{safe}.csv")
        self.status = f"Exportado: {path.name}"

    # ------------------------------------------------------------------ eventos
    def handle(self, event):
        for i, btn in enumerate(self.series_buttons):
            if btn.clicked(event) and i != self.selected:
                self.series_buttons[self.selected].active = False
                btn.active = True
                self.selected = i
                self.train()
                return
        for slider in (self.epochs, self.lookback, self.history):
            if slider.handle(event):
                self.dirty = True
                if slider is self.lookback:
                    self._update_history_range()
        if self.train_btn.clicked(event):
            self.train()
        elif self.baseline_btn.clicked(event):
            self.show_baseline = not self.show_baseline
            self.baseline_btn.active = self.show_baseline
        elif self.export_btn.clicked(event):
            self.export()

    # ------------------------------------------------------------------ dibujo
    def draw(self):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, PANEL, (0, 0, PANEL_W, HEIGHT))
        self.screen.blit(self.title_font.render("Tarea B - Pronóstico", True, TEXT), (20, 18))
        self.screen.blit(self.small.render(f"Objetivo: {config.TARGET_COLUMN} semanal",
                                           True, MUTED), (20, 44))
        for btn in self.series_buttons:
            btn.draw(self.screen, self.font)
        for slider in (self.epochs, self.lookback, self.history):
            slider.draw(self.screen, self.font)
        self.train_btn.label = "Entrenar *" if self.dirty else "Entrenar"
        self.train_btn.active = self.dirty
        for btn in (self.train_btn, self.baseline_btn, self.export_btn):
            btn.draw(self.screen, self.font)
        self.screen.blit(self.small.render(self.status[:48], True, MUTED), (20, HEIGHT - 28))

        plot = pygame.Rect(PANEL_W + 70, 40, WIDTH - PANEL_W - 100, 420)
        self.draw_forecast(plot)
        self.draw_metrics(pygame.Rect(PANEL_W + 30, 500, 520, 190))
        self.draw_loss(pygame.Rect(PANEL_W + 590, 520, WIDTH - PANEL_W - 620, 150))

    def draw_forecast(self, area):
        r = self.result
        title = f"{r.series_name}: real vs pronóstico" if r else "Sin resultado"
        self.screen.blit(self.title_font.render(title, True, TEXT), (area.x - 40, 8))
        if not r:
            return

        n = len(r.actual)
        xs_total = n + 1  # +1 para la semana pronosticada
        values = list(r.actual) + list(r.test_pred) + [r.next_pred]
        if self.show_baseline:
            values += list(r.test_baseline) + [r.next_baseline]
        vmax = max(values) * 1.1 or 1.0

        def px(i):
            return area.x + i / (xs_total - 1) * area.w

        def py(v):
            return area.bottom - v / vmax * area.h

        # Zona de prueba sombreada
        split = n - len(r.test_actual)
        pygame.draw.rect(self.screen, TEST_BG,
                         (px(split), area.y, area.right - px(split), area.h))
        self.screen.blit(self.small.render("prueba", True, LSTM_C), (px(split) + 4, area.y + 4))

        # Rejilla y eje Y
        for k in range(5):
            v = vmax * k / 4
            y = py(v)
            pygame.draw.line(self.screen, GRID, (area.x, y), (area.right, y))
            lbl = self.small.render(fmt_money(v), True, MUTED)
            self.screen.blit(lbl, (area.x - lbl.get_width() - 6, y - 7))
        # Eje X: algunas fechas
        step = max(1, n // 6)
        for i in range(0, n, step):
            lbl = self.small.render(f"{r.weeks[i]:%d %b}", True, MUTED)
            self.screen.blit(lbl, (px(i) - lbl.get_width() / 2, area.bottom + 6))
        nxt_lbl = self.small.render(f"{r.next_week:%d %b}", True, LSTM_C)
        self.screen.blit(nxt_lbl, (px(n) - nxt_lbl.get_width() / 2, area.bottom + 6))

        # Serie real
        pts = [(px(i), py(v)) for i, v in enumerate(r.actual)]
        pygame.draw.lines(self.screen, ACTUAL, False, pts, 2)

        # Baseline (media móvil)
        if self.show_baseline:
            bpts = [(px(split + i), py(v)) for i, v in enumerate(r.test_baseline)]
            bpts.append((px(n), py(r.next_baseline)))
            pygame.draw.lines(self.screen, BASE_C, False, bpts, 2)

        # Predicciones LSTM + próxima semana
        lpts = [(px(split + i), py(v)) for i, v in enumerate(r.test_pred)]
        lpts.append((px(n), py(r.next_pred)))
        pygame.draw.lines(self.screen, LSTM_C, False, lpts, 3)
        for p in lpts[:-1]:
            pygame.draw.circle(self.screen, LSTM_C, (int(p[0]), int(p[1])), 3)
        nx, ny = lpts[-1]
        pygame.draw.circle(self.screen, LSTM_C, (int(nx), int(ny)), 8)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(nx), int(ny)), 4)

        # Leyenda
        lx, ly = area.x + 10, area.y + 10
        items = [(ACTUAL, "Real"), (LSTM_C, "LSTM")]
        if self.show_baseline:
            items.append((BASE_C, f"Media móvil ({config.MOVING_AVG_WINDOW} sem)"))
        for color, label in items:
            pygame.draw.line(self.screen, color, (lx, ly + 8), (lx + 22, ly + 8), 3)
            self.screen.blit(self.small.render(label, True, TEXT), (lx + 28, ly + 1))
            ly += 18

    def draw_metrics(self, area):
        pygame.draw.rect(self.screen, PANEL, area, border_radius=8)
        r = self.result
        if not r:
            return
        x, y = area.x + 16, area.y + 12
        self.screen.blit(self.title_font.render("Métricas (periodo de prueba)", True, TEXT), (x, y))
        y += 34
        cols = [x, x + 170, x + 300]
        for cx, h in zip(cols, ["Modelo", "MAE", "RMSE"]):
            self.screen.blit(self.font.render(h, True, MUTED), (cx, y))
        y += 24
        rows = [("LSTM", r.mae, r.rmse, LSTM_C),
                ("Media móvil", r.baseline_mae, r.baseline_rmse, BASE_C)]
        for name, m, s, color in rows:
            self.screen.blit(self.font.render(name, True, color), (cols[0], y))
            self.screen.blit(self.font.render(fmt_money(m), True, TEXT), (cols[1], y))
            self.screen.blit(self.font.render(fmt_money(s), True, TEXT), (cols[2], y))
            y += 24
        y += 10
        winner = "LSTM" if r.mae < r.baseline_mae else "Media móvil"
        self.screen.blit(self.font.render(
            f"Pronóstico semana {r.next_week:%d-%b-%Y}:  LSTM {fmt_money(r.next_pred)}"
            f"  |  MA {fmt_money(r.next_baseline)}", True, TEXT), (x, y))
        y += 24
        self.screen.blit(self.small.render(f"Menor MAE: {winner}", True, MUTED), (x, y))

    def draw_loss(self, area):
        self.screen.blit(self.font.render("Pérdida de entrenamiento (MSE)", True, TEXT),
                         (area.x, area.y - 22))
        pygame.draw.rect(self.screen, PANEL, area, border_radius=6)
        if not self.result or len(self.result.loss_history) < 2:
            return
        loss = self.result.loss_history
        lmax = max(loss) or 1.0
        pts = [(area.x + 8 + i / (len(loss) - 1) * (area.w - 16),
                area.bottom - 8 - l / lmax * (area.h - 16)) for i, l in enumerate(loss)]
        pygame.draw.lines(self.screen, ACCENT, False, pts, 2)
        self.screen.blit(self.small.render(f"final: {loss[-1]:.4f}", True, MUTED),
                         (area.right - 80, area.y + 6))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                else:
                    self.handle(event)
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Tarea B - App Pygame")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    args = parser.parse_args()
    ForecastApp(args.data).run()


if __name__ == "__main__":
    main()
