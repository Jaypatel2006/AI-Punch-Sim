import pygame
import sys
import random
import socket
import threading

pygame.init()

WIDTH, HEIGHT = 900, 500
FPS = 60
GROUND_Y = 380

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PIXEL FIST — Q-Learning MDP AI")
clock = pygame.time.Clock()

font_big   = pygame.font.SysFont("couriernew", 52, bold=True)
font_med   = pygame.font.SysFont("couriernew", 28, bold=True)
font_small = pygame.font.SysFont("couriernew", 20)
font_tiny  = pygame.font.SysFont("couriernew", 16)

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (220, 50, 50)
BLUE   = (50, 120, 220)
YELLOW = (255, 220, 50)
ORANGE = (255, 140, 0)
GRAY   = (130, 130, 130)

# ── Shared state ──────────────────────────────────────────────────────────────
command      = None
command_lock = threading.Lock()
cv_connected = False


def socket_server():
    global command, cv_connected
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 5555))
    srv.listen(1)
    print("[GAME] Waiting for OpenCV on port 5555...")
    while True:
        conn, addr = srv.accept()
        cv_connected = True
        print(f"[GAME] OpenCV connected: {addr}")
        buf = ""
        try:
            while True:
                data = conn.recv(1024).decode(errors="ignore")
                if not data:
                    break
                buf += data
                # consume all complete command tokens
                changed = True
                while changed:
                    changed = False
                    for token in ("PUNCH", "BLOCK"):
                        idx = buf.find(token)
                        if idx != -1:
                            with command_lock:
                                command = token
                            buf = buf[idx + len(token):]
                            changed = True
                            break
                if len(buf) > 512:
                    buf = buf[-64:]
        except Exception as e:
            print(f"[GAME] Connection error: {e}")
        finally:
            cv_connected = False
            conn.close()
            print("[GAME] Disconnected. Waiting for next connection...")


# Start socket server BEFORE the game loop
threading.Thread(target=socket_server, daemon=True).start()


# ── Fighter ───────────────────────────────────────────────────────────────────
class Fighter:
    W = 50
    H = 90

    def __init__(self, x, color, name, keys, is_ai=False):
        self.start_x      = float(x)
        self.x            = self.start_x
        self.y            = float(GROUND_Y - self.H)
        self.color        = color
        self.name         = name
        self.keys         = keys
        self.hp           = 100
        self.facing       = 1
        self.state        = "idle"
        self.state_timer  = 0
        self.punch_landed = False

        self.is_ai = is_ai
        if self.is_ai:
            self.q_table     = {}
            self.alpha       = 0.15
            self.gamma       = 0.90
            self.epsilon     = 1.0
            self.last_state  = None
            self.last_action = 0
            self.last_my_hp  = 100
            self.last_opp_hp = 100

    def reset_round(self):
        self.hp           = 100
        self.x            = self.start_x
        self.state        = "idle"
        self.state_timer  = 0
        self.punch_landed = False
        if self.is_ai:
            self.last_state  = None
            self.last_my_hp  = 100
            self.last_opp_hp = 100

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.W, self.H)

    def fist_rect(self):
        if self.facing == 1:
            return pygame.Rect(int(self.x) + self.W, int(self.y) + 20, 46, 30)
        else:
            return pygame.Rect(int(self.x) - 46, int(self.y) + 20, 46, 30)

    def handle_input(self, kp):
        global command
        if self.state in ("hurt", "ko") or self.is_ai:
            return

        # ── Socket command (from OpenCV) ──────────────────────────────────────
        with command_lock:
            cmd     = command
            command = None       # consume

        if cmd == "PUNCH" and self.state not in ("punching", "blocking", "hurt"):
            self.state        = "punching"
            self.state_timer  = 22
            self.punch_landed = False
            print("[GAME] PUNCH executed")

        elif cmd == "BLOCK" and self.state not in ("punching", "hurt"):
            self.state       = "blocking"
            self.state_timer = 40
            print("[GAME] BLOCK executed")

        # ── Keyboard fallback ─────────────────────────────────────────────────
        k = self.keys
        if kp[k["punch"]] and self.state not in ("punching", "blocking", "hurt"):
            self.state        = "punching"
            self.state_timer  = 22
            self.punch_landed = False

        if kp[k["block"]] and self.state not in ("punching", "hurt"):
            self.state       = "blocking"
            self.state_timer = 20
        elif not kp[k["block"]] and self.state == "blocking" and cmd != "BLOCK":
            self.state       = "idle"
            self.state_timer = 0

    def get_mdp_state(self, opponent):
        opp_map = {"idle": 0, "punching": 1, "blocking": 2, "hurt": 3, "ko": 4}
        s_opp   = opp_map.get(opponent.state, 0)
        hp_diff = self.hp - opponent.hp
        s_hp    = 1 if hp_diff > 15 else (-1 if hp_diff < -15 else 0)
        return (s_opp, s_hp)

    def ai_update(self, opponent):
        if not self.is_ai or self.state == "ko" or opponent.state == "ko":
            return

        if self.state in ("punching", "hurt", "blocking", "waiting"):
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.state = "idle"
            return

        current_state = self.get_mdp_state(opponent)

        if self.last_state is not None:
            damage_dealt = self.last_opp_hp - opponent.hp
            damage_taken = self.last_my_hp  - self.hp
            reward       = (damage_dealt * 1.5) - (damage_taken * 1.0)
            if self.last_action == 0:
                reward -= 0.5
            q_next  = self.q_table.get(current_state, [0.0, 0.0, 0.0])
            old_q   = self.q_table[self.last_state][self.last_action]
            self.q_table[self.last_state][self.last_action] = (
                old_q + self.alpha * (reward + self.gamma * max(q_next) - old_q)
            )

        if current_state not in self.q_table:
            self.q_table[current_state] = [0.0, 0.0, 0.0]

        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            q_vals       = self.q_table[current_state]
            max_val      = max(q_vals)
            best_actions = [i for i, v in enumerate(q_vals) if v == max_val]
            action       = random.choice(best_actions)

        self.last_state  = current_state
        self.last_action = action
        self.last_my_hp  = self.hp
        self.last_opp_hp = opponent.hp

        if action == 0:
            self.state       = "waiting"
            self.state_timer = 10
        elif action == 1:
            self.state        = "punching"
            self.state_timer  = 22
            self.punch_landed = False
        elif action == 2:
            self.state       = "blocking"
            self.state_timer = 25

    def update(self, opponent=None):
        if self.is_ai:
            self.ai_update(opponent)
        else:
            if self.state in ("punching", "hurt"):
                self.state_timer -= 1
                if self.state_timer <= 0:
                    self.state = "idle"

    def take_hit(self, dmg):
        if self.state == "ko":
            return
        if self.state == "blocking":
            self.hp -= 1
        else:
            self.hp -= dmg//2
            self.state       = "hurt"
            self.state_timer = 18
        if self.hp <= 0:
            self.hp    = 0
            self.state = "ko"

    def draw(self, surf):
        x   = int(self.x)
        y   = int(self.y)
        f   = self.facing
        col = self.color
        if self.state == "hurt":
            col = YELLOW
        elif self.state == "blocking":
            col = tuple(min(255, c + 70) for c in self.color)

        pygame.draw.ellipse(surf, (20, 10, 35), (x + 5, GROUND_Y + 4, self.W - 10, 10))

        if self.state == "ko":
            pygame.draw.ellipse(surf, GRAY, (x - 15, GROUND_Y - 16, self.W + 30, 18))
            pygame.draw.circle(surf, GRAY, (x + self.W // 2 + f * 30, GROUND_Y - 10), 14)
            return

        pygame.draw.rect(surf, col, (x + 8,  y + 60, 14, 30))
        pygame.draw.rect(surf, col, (x + 28, y + 60, 14, 30))
        pygame.draw.rect(surf, col, (x + 5,  y + 28, self.W - 10, 36))
        pygame.draw.ellipse(surf, col, (x + 8, y, 34, 30))

        ex = x + 22 + f * 6
        ey = y + 12
        pygame.draw.circle(surf, WHITE, (ex, ey), 5)
        pygame.draw.circle(surf, BLACK, (ex + f * 2, ey), 3)

        if self.state == "punching":
            arm = 46
            if f == 1:
                pygame.draw.rect(surf, col, (x + self.W, y + 30, arm, 13))
                pygame.draw.circle(surf, ORANGE, (x + self.W + arm, y + 37), 12)
            else:
                pygame.draw.rect(surf, col, (x - arm, y + 30, arm, 13))
                pygame.draw.circle(surf, ORANGE, (x - arm, y + 37), 12)
        else:
            pygame.draw.rect(surf, col, (x - 5,          y + 30, 12, 28))
            pygame.draw.rect(surf, col, (x + self.W - 7, y + 30, 12, 28))

        if self.state == "blocking":
            sx = x + self.W + 4 if f == 1 else x - 22
            sh = pygame.Surface((20, 52), pygame.SRCALPHA)
            pygame.draw.rect(sh, (180, 210, 255, 200), (0, 0, 20, 52), border_radius=5)
            surf.blit(sh, (sx, y + 14))

        lbl = font_tiny.render(self.name, True, WHITE)
        surf.blit(lbl, (x + self.W // 2 - lbl.get_width() // 2, y - 20))


def draw_bg():
    for row in range(HEIGHT):
        t = row / HEIGHT
        r = int(12 + t * 33)
        g = int(8  + t * 17)
        b = int(28 + t * 52)
        pygame.draw.line(screen, (r, g, b), (0, row), (WIDTH, row))
    pygame.draw.rect(screen, (45, 25, 65), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.line(screen, (110, 70, 150), (0, GROUND_Y), (WIDTH, GROUND_Y), 3)


def draw_hud(f1, f2, rounds):
    bw  = 300
    bh  = 24
    pad = 40

    pygame.draw.rect(screen, (60, 20, 20), (pad, 18, bw, bh), border_radius=5)
    fill1 = max(0, int(bw * f1.hp / 100))
    if fill1:
        pygame.draw.rect(screen, RED, (pad, 18, fill1, bh), border_radius=5)
    pygame.draw.rect(screen, WHITE, (pad, 18, bw, bh), 2, border_radius=5)

    bx2 = WIDTH - pad - bw
    pygame.draw.rect(screen, (20, 20, 60), (bx2, 18, bw, bh), border_radius=5)
    fill2 = max(0, int(bw * f2.hp / 100))
    if fill2:
        pygame.draw.rect(screen, BLUE, (WIDTH - pad - fill2, 18, fill2, bh), border_radius=5)
    pygame.draw.rect(screen, WHITE, (bx2, 18, bw, bh), 2, border_radius=5)

    screen.blit(font_small.render(f1.name, True, RED), (pad, 46))

    ai_txt = f"eps:{f2.epsilon:.2f}  states:{len(f2.q_table)}"
    n2 = font_tiny.render(ai_txt, True, YELLOW)
    screen.blit(n2, (WIDTH - pad - n2.get_width(), 46))

    vs = font_med.render(f"ROUND {rounds}", True, YELLOW)
    screen.blit(vs, (WIDTH // 2 - vs.get_width() // 2, 20))

    # Connection status indicator
    conn_txt = "CV: CONNECTED" if cv_connected else "CV: waiting..."
    conn_col = (0, 255, 120)   if cv_connected else (220, 80, 80)
    screen.blit(font_tiny.render(conn_txt, True, conn_col),
                (WIDTH // 2 - 55, 46))

    screen.blit(font_tiny.render("A=Punch  B=Block (keyboard fallback)", True, (120, 110, 140)),
                (10, HEIGHT - 22))


def draw_overlay(text):
    ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 160))
    screen.blit(ov, (0, 0))
    t = font_big.render(text, True, YELLOW)
    screen.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 - 55))


def main():
    p1 = Fighter(380, RED,  "HUMAN",     {"punch": pygame.K_a, "block": pygame.K_b}, is_ai=False)
    p2 = Fighter(470, BLUE, "MDP-AGENT", None, is_ai=True)

    rounds_played = 1
    round_over    = False
    winner        = ""
    end_timer     = 0

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        kp = pygame.key.get_pressed()

        if not round_over:
            p1.handle_input(kp)
            p1.update(p2)
            p2.update(p1)

            if p1.x < p2.x:
                p1.facing, p2.facing = 1, -1
            else:
                p1.facing, p2.facing = -1, 1

            for att, dfn in [(p1, p2), (p2, p1)]:
                if att.state == "punching" and not att.punch_landed:
                    if att.fist_rect().colliderect(dfn.rect()):
                        dfn.take_hit(15)
                        att.punch_landed = True

            if p1.hp <= 0 or p2.hp <= 0:
                round_over = True
                if p1.hp <= 0 and p2.hp <= 0:
                    winner = "DRAW!"
                elif p1.hp <= 0:
                    winner = "AI WINS!"
                else:
                    winner = "HUMAN WINS!"
                end_timer = 150
        else:
            end_timer -= 1
            if end_timer <= 0:
                p1.reset_round()
                p2.reset_round()
                p2.epsilon = 1
                rounds_played += 1
                round_over = False

        draw_bg()
        p1.draw(screen)
        p2.draw(screen)
        draw_hud(p1, p2, rounds_played)
        if round_over:
            draw_overlay(winner)

        pygame.display.flip()


if __name__ == "__main__":
    main()