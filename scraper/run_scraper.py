"""
Win Go Auto-Scraper v5 — Robust multi-timer capture with auto-login.

Improvements over v4:
  - Auto-login: fills phone/password from .env, user just solves the puzzle
  - Robust tab switching with retry + verification
  - Proactive keep-alive to prevent session timeout
  - Smarter polling: 30sec timer gets polled more frequently
  - Stale element recovery with iframe re-entry
  - Deque-based result tracking (last 5 balls, not just first)

Usage:
    python scraper/run_scraper.py                  # Monitor all timers (default)
    python scraper/run_scraper.py --timer 3min     # Monitor 3min only
    python scraper/run_scraper.py --timer 30sec    # Monitor 30sec only
    python scraper/run_scraper.py --wait 120       # Wait 120s for puzzle
"""
import os
import sys
import time
import re
import argparse
import requests
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

from config import PHONE_NUMBER, PASSWORD

API_URL = "http://127.0.0.1:8000"

# ===== SELECTORS =====
SEL = {
    "timer_tabs": ".timer-card",
    "timer_title": ".card-title",
    "recent_balls": ".TimeLeft__C-num > div",
    "game_name": ".TimeLeft__C-name",
}

TIMER_NAMES = {
    "30sec": "WinGo 30sec",
    "1min":  "WinGo 1 Min",
    "3min":  "WinGo 3 Min",
}

# How often each timer produces a new result (seconds)
TIMER_INTERVALS = {
    "30sec": 30,
    "1min":  60,
    "3min":  180,
}

READY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "scraper_ready.flag"
)


# ═══════════════════════════════════════════
#  DRIVER SETUP
# ═══════════════════════════════════════════

def create_driver():
    """Create a Chrome driver with anti-detection settings."""
    options = Options()
    options.add_argument("--window-size=420,900")
    options.add_argument("--window-position=50,50")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    print("  Downloading ChromeDriver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ═══════════════════════════════════════════
#  IFRAME HELPERS
# ═══════════════════════════════════════════

def safe_default_content(driver):
    """Switch to default content, ignoring errors."""
    try:
        driver.switch_to.default_content()
    except WebDriverException:
        pass


def enter_iframe(driver, timeout=5):
    """Enter the game iframe. Returns True on success."""
    safe_default_content(driver)
    try:
        iframe = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe"))
        )
        driver.switch_to.frame(iframe)
        return True
    except (TimeoutException, WebDriverException):
        return False


# ═══════════════════════════════════════════
#  PAGE READING (with stale-element retry)
# ═══════════════════════════════════════════

def _retry(fn, retries=3, delay=0.5):
    """Retry a function on stale element / no-such-element errors."""
    for attempt in range(retries):
        try:
            return fn()
        except (StaleElementReferenceException, NoSuchElementException):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
        except WebDriverException:
            return None
    return None


def get_balls(driver):
    """Read the 5 recent result balls (class='n0'...'n9')."""
    def _read():
        els = driver.find_elements(By.CSS_SELECTOR, SEL["recent_balls"])
        digits = []
        for el in els:
            cls = el.get_attribute("class") or ""
            m = re.search(r'\bn(\d)\b', cls)
            if m:
                digits.append(int(m.group(1)))
        return digits if digits else None
    return _retry(_read) or []


def get_active_game(driver):
    """Get the currently active game name text."""
    def _read():
        el = driver.find_element(By.CSS_SELECTOR, SEL["game_name"])
        return el.text.strip()
    return _retry(_read) or ""


# ═══════════════════════════════════════════
#  TAB SWITCHING (robust)
# ═══════════════════════════════════════════

def click_tab(driver, timer_key, max_retries=3):
    """
    Click the tab for `timer_key` and verify the active game switched.
    Retries with iframe re-entry on failure.
    Returns True if the tab is confirmed active.
    """
    target_name = TIMER_NAMES.get(timer_key)
    if not target_name:
        return False

    for attempt in range(max_retries):
        try:
            # Find and click the correct tab
            tabs = driver.find_elements(By.CSS_SELECTOR, SEL["timer_tabs"])
            clicked = False
            for tab in tabs:
                try:
                    titles = tab.find_elements(By.CSS_SELECTOR, SEL["timer_title"])
                    if titles and titles[0].text.strip() == target_name:
                        driver.execute_script("arguments[0].click();", tab)
                        clicked = True
                        break
                except StaleElementReferenceException:
                    continue

            if not clicked:
                # Tabs might be stale — re-enter iframe
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    enter_iframe(driver, timeout=3)
                continue

            # Wait for the game name to confirm the switch
            for _ in range(8):  # up to 4 seconds
                time.sleep(0.5)
                game = get_active_game(driver)
                if game and target_name in game:
                    return True

            # Tab didn't switch — retry with iframe re-entry
            if attempt < max_retries - 1:
                safe_default_content(driver)
                enter_iframe(driver, timeout=3)

        except WebDriverException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                safe_default_content(driver)
                enter_iframe(driver, timeout=3)
            else:
                print(f"  [TAB] Failed to switch to {timer_key}: {str(e)[:60]}")

    return False


def verify_active_tab(driver, timer_key):
    """Check if the currently active tab matches expectations."""
    game = get_active_game(driver)
    expected = TIMER_NAMES.get(timer_key, "")
    return bool(game and expected and expected in game)


# ═══════════════════════════════════════════
#  API
# ═══════════════════════════════════════════

def api_result(digit, timer="3min"):
    """Post a result to the local API server."""
    try:
        resp = requests.post(f"{API_URL}/api/result", json={
            "digit": digit,
            "round_id": f"{timer}_{int(time.time() * 1000)}",
            "source": f"scraper_{timer}",
        }, timeout=5)
        return resp.json()
    except Exception:
        return None


# ═══════════════════════════════════════════
#  AUTO-LOGIN
# ═══════════════════════════════════════════

def auto_login(driver, phone, password, wait_seconds=120):
    """
    Navigate directly to surat91.com login page, auto-fill phone + password,
    click login. Then wait for user to solve the puzzle and navigate to Win Go.

    Note: 91clu.org wraps surat91.com in a cross-origin iframe, making
    auto-fill impossible. We go directly to surat91.com/#/login instead.
    """
    LOGIN_URL = "https://surat91.com/#/login"

    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║   AUTO-LOGIN: Navigating to login page...    ║")
    print("  ╚═══════════════════════════════════════════════╝")

    driver.get(LOGIN_URL)
    time.sleep(4)

    # If we landed on register page, navigate to login
    current_url = driver.current_url
    if "register" in current_url.lower():
        print("  Landed on register page, switching to login...")
        driver.get(LOGIN_URL)
        time.sleep(3)

    login_filled = False

    for attempt in range(8):  # try for ~40 seconds
        try:
            # Use the exact selectors from the 91Club login page
            phone_input = None
            password_input = None

            # Primary selectors (exact match from site inspection)
            try:
                phone_input = driver.find_element(
                    By.CSS_SELECTOR, "input[placeholder*='phone number']"
                )
            except NoSuchElementException:
                pass

            # Fallback: search all inputs
            if not phone_input:
                inputs = driver.find_elements(By.CSS_SELECTOR, "input")
                for inp in inputs:
                    inp_type = (inp.get_attribute("type") or "").lower()
                    placeholder = (inp.get_attribute("placeholder") or "").lower()
                    if any(kw in placeholder for kw in ["phone", "mobile", "number"]) or inp_type == "tel":
                        phone_input = inp
                        break

            try:
                password_input = driver.find_element(
                    By.CSS_SELECTOR, "input[placeholder='Password']"
                )
            except NoSuchElementException:
                pass

            # Fallback for password
            if not password_input:
                inputs = driver.find_elements(By.CSS_SELECTOR, "input")
                for inp in inputs:
                    inp_type = (inp.get_attribute("type") or "").lower()
                    placeholder = (inp.get_attribute("placeholder") or "").lower()
                    if "password" in placeholder or inp_type == "password":
                        password_input = inp
                        break

            if phone_input and password_input:
                # Clear and fill phone number
                phone_input.click()
                time.sleep(0.2)
                phone_input.clear()
                # Use JS to set value reliably (some React inputs ignore send_keys)
                driver.execute_script(
                    "arguments[0].value = ''; arguments[0].focus();", phone_input
                )
                phone_input.send_keys(str(phone))
                time.sleep(0.3)

                # Clear and fill password
                password_input.click()
                time.sleep(0.2)
                password_input.clear()
                driver.execute_script(
                    "arguments[0].value = ''; arguments[0].focus();", password_input
                )
                password_input.send_keys(str(password))
                time.sleep(0.3)

                print(f"  ✓ Phone: {str(phone)[:3]}****{str(phone)[-2:]}")
                print(f"  ✓ Password: filled")
                login_filled = True

                # Click the "Log in" button (NOT the "Register" button)
                time.sleep(0.5)
                buttons = driver.find_elements(By.CSS_SELECTOR, "button")
                for btn in buttons:
                    txt = (btn.text or "").strip().lower()
                    classes = (btn.get_attribute("class") or "").lower()
                    # Skip the register button
                    if "register" in classes or "register" in txt:
                        continue
                    if any(kw in txt for kw in ["log in", "login", "sign in"]):
                        driver.execute_script("arguments[0].click();", btn)
                        print(f"  ✓ Login button clicked!")
                        break

                break

        except Exception as e:
            if attempt > 2:
                print(f"  Auto-login attempt {attempt + 1}: {str(e)[:60]}")

        time.sleep(5)

    if not login_filled:
        print("  ⚠ Could not auto-fill credentials. Please log in manually.")
        print("  The login page should be open in Chrome.")

    # Now wait for user to solve puzzle + navigate to Win Go
    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║   Please:                                    ║")
    print("  ║   1. Solve the puzzle/captcha (if any)       ║")
    print("  ║   2. Navigate to the Win Go section          ║")
    print("  ╚═══════════════════════════════════════════════╝")
    print()
    print(f"  Waiting up to {wait_seconds}s for game to load...")
    print("  (Or create 'data/scraper_ready.flag' to start early)")
    print()

    if os.path.exists(READY_FILE):
        os.remove(READY_FILE)

    # Wait until we can detect the game iframe with balls
    for i in range(wait_seconds, 0, -1):
        if os.path.exists(READY_FILE):
            print(f"\n  ✓ Ready flag detected! Starting...")
            break

        # Check if game is loaded
        if i % 5 == 0:
            if enter_iframe(driver, timeout=2):
                balls = get_balls(driver)
                safe_default_content(driver)
                if balls:
                    print(f"\n  ✓ Game detected! Balls: {balls}")
                    print("  Starting scraper in 3 seconds...\n")
                    time.sleep(3)
                    return True

        if i % 15 == 0:
            print(f"  Waiting... {i}s remaining")

        time.sleep(1)

    print("  Login wait complete. Attempting to start...\n")
    return True


# ═══════════════════════════════════════════
#  SESSION HEALTH
# ═══════════════════════════════════════════

def is_session_alive(driver):
    """Check if the Chrome session and game are still working."""
    try:
        _ = driver.title
        safe_default_content(driver)
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if not iframes:
            page = driver.page_source[:2000].lower()
            if "login" in page or "register" in page:
                return False
        return True
    except WebDriverException:
        return False


def keep_alive(driver):
    """Proactive keep-alive: small interaction to prevent session timeout."""
    try:
        driver.execute_script("window.scrollBy(0, 1); window.scrollBy(0, -1);")
    except WebDriverException:
        pass


def wait_for_manual_relogin(driver):
    """Alert user and wait for them to log back in."""
    ts = datetime.now().strftime("%H:%M:%S")
    print()
    print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"  [{ts}] SESSION EXPIRED!")
    print("  Please log back in manually in the Chrome window")
    print("  and navigate to Win Go. Scraper will auto-resume.")
    print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()

    for attempt in range(120):
        time.sleep(5)
        try:
            if is_session_alive(driver) and enter_iframe(driver, timeout=3):
                balls = get_balls(driver)
                safe_default_content(driver)
                if balls:
                    print(f"  [SESSION] ✓ Recovered! Resuming...")
                    return True
        except WebDriverException:
            pass
        if attempt % 12 == 0 and attempt > 0:
            print(f"  [SESSION] Still waiting... ({attempt * 5}s)")

    return False


# ═══════════════════════════════════════════
#  SINGLE TIMER MONITOR
# ═══════════════════════════════════════════

def monitor_single(driver, timer_key):
    """Monitor one timer by polling the recent balls."""
    timer_name = TIMER_NAMES.get(timer_key, timer_key)

    if not enter_iframe(driver):
        print("  ERROR: No game iframe found!")
        return

    click_tab(driver, timer_key)
    time.sleep(1)

    balls = get_balls(driver)
    print(f"  Monitoring: {timer_name}")
    print(f"  Initial balls: {balls}")
    print(f"\n  Watching for changes... (Ctrl+C to stop)")
    print("=" * 55)

    last_balls = deque(balls[:3], maxlen=3) if balls else deque(maxlen=3)
    captured = 0
    poll = 0
    last_keepalive = time.time()
    sleep_time = 2 if "30sec" in timer_key else 4

    while True:
        try:
            # Ensure we're in iframe
            try:
                driver.find_element(By.CSS_SELECTOR, SEL["game_name"])
            except (NoSuchElementException, WebDriverException):
                enter_iframe(driver)

            now_balls = get_balls(driver)
            poll += 1

            if now_balls and len(now_balls) > 0:
                # Detect new result: first ball changed AND it's not a glitch
                if now_balls[0] != (last_balls[0] if last_balls else None):
                    d = now_balls[0]
                    lab = "SMALL" if d <= 4 else "BIG"
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] [{timer_key:>5s}] >>> {d} ({lab})  balls: {now_balls}")

                    resp = api_result(d, timer_key)
                    captured += 1
                    last_balls.appendleft(d)

                    if resp and resp.get("status") == "ok":
                        c = resp.get("result", {}).get("prediction_correct")
                        if c is not None:
                            print(f"           Prev prediction: {'✓ HIT!' if c else '✗ MISS'}")

            # Periodic status
            if poll % 30 == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] alive | polls: {poll} | captured: {captured}")

            # Proactive keep-alive every 2 minutes
            if time.time() - last_keepalive > 120:
                keep_alive(driver)
                last_keepalive = time.time()

            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n  Stopped. Captured: {captured}")
            break
        except Exception as e:
            err_msg = str(e)[:80]
            print(f"  Err: {err_msg}")
            time.sleep(5)
            try:
                if not is_session_alive(driver):
                    if not wait_for_manual_relogin(driver):
                        print("  Could not recover. Exiting.")
                        break
                safe_default_content(driver)
                enter_iframe(driver)
                click_tab(driver, timer_key)
            except WebDriverException:
                pass


# ═══════════════════════════════════════════
#  MULTI-TIMER MONITOR (the main fix)
# ═══════════════════════════════════════════

def monitor_all(driver, timers):
    """
    Monitor all timers by cycling through tabs.
    
    Strategy:
    - Cycle through each timer tab, read latest result, move on.
    - 30sec timer gets polled TWICE per cycle (it changes fastest).
    - Each tab read takes ~1.5-2s, so a full cycle is ~6-8s.
    - Proactive keep-alive every 2 minutes.
    """
    print(f"\n  Multi-timer: {', '.join(timers)}")
    print("=" * 55)

    if not enter_iframe(driver):
        print("  ERROR: No game iframe!")
        return

    # Build poll order: 30sec appears twice if present
    poll_order = []
    for tk in timers:
        poll_order.append(tk)
    if "30sec" in timers:
        # Insert an extra 30sec poll in the middle
        mid = len(poll_order) // 2 + 1
        poll_order.insert(mid, "30sec")

    print(f"  Poll order per cycle: {poll_order}")

    # Initialize state per timer
    state = {}
    for tk in timers:
        if not click_tab(driver, tk):
            print(f"  ⚠ Could not switch to {tk} tab initially. Will retry.")
        time.sleep(1)
        balls = get_balls(driver)
        state[tk] = {
            "last_balls": deque(balls[:3], maxlen=3) if balls else deque(maxlen=3),
            "captured": 0,
            "last_seen_time": time.time(),
            "consecutive_fails": 0,
        }
        print(f"  [{tk}] Initial: {balls}")

    print()

    cycle = 0
    last_keepalive = time.time()
    last_full_recovery = time.time()

    while True:
        for tk in poll_order:
            try:
                # ---- Switch tab ----
                safe_default_content(driver)
                if not enter_iframe(driver, timeout=3):
                    print(f"  [{tk}] iframe lost, recovering...")
                    time.sleep(2)
                    if not enter_iframe(driver, timeout=5):
                        raise WebDriverException("Cannot enter iframe")

                if not click_tab(driver, tk):
                    state[tk]["consecutive_fails"] += 1
                    if state[tk]["consecutive_fails"] > 5:
                        print(f"  [{tk}] ⚠ Tab switch failed {state[tk]['consecutive_fails']} times")
                    time.sleep(0.3)
                    continue

                state[tk]["consecutive_fails"] = 0

                # ---- Read balls ----
                time.sleep(0.3)  # small settle time after tab switch
                now_balls = get_balls(driver)
                if not now_balls:
                    time.sleep(0.5)
                    now_balls = get_balls(driver)

                if not now_balls:
                    continue

                first = now_balls[0]
                prev_first = state[tk]["last_balls"][0] if state[tk]["last_balls"] else None

                # ---- Detect new result ----
                if first != prev_first:
                    d = first
                    lab = "SMALL" if d <= 4 else "BIG"
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] [{tk:>5s}] >>> {d} ({lab})  balls: {now_balls}")

                    resp = api_result(d, tk)
                    state[tk]["captured"] += 1
                    state[tk]["last_balls"].appendleft(first)
                    state[tk]["last_seen_time"] = time.time()

                    if resp and resp.get("status") == "ok":
                        c = resp.get("result", {}).get("prediction_correct")
                        if c is not None:
                            sym = "✓" if c else "✗"
                            print(f"           Prev prediction: {sym}")

                # Small pause between tabs
                time.sleep(0.5)

            except KeyboardInterrupt:
                raise
            except WebDriverException as e:
                err_msg = str(e)[:80]
                if "invalid session" in err_msg.lower() or "disconnected" in err_msg.lower():
                    print(f"  Chrome session lost!")
                    if not wait_for_manual_relogin(driver):
                        return
                    enter_iframe(driver)
                else:
                    print(f"  [{tk}] WebDriver err: {err_msg}")
                    time.sleep(2)
                    safe_default_content(driver)
                    enter_iframe(driver, timeout=3)
            except Exception as e:
                err_msg = str(e)[:80]
                if "no such element" not in err_msg.lower():
                    print(f"  [{tk}] err: {err_msg}")
                time.sleep(1)

        cycle += 1

        # ---- Proactive keep-alive every 2 minutes ----
        if time.time() - last_keepalive > 120:
            keep_alive(driver)
            last_keepalive = time.time()

        # ---- Full recovery check every 10 minutes ----
        if time.time() - last_full_recovery > 600:
            if not is_session_alive(driver):
                print("  [HEALTH] Session check failed, attempting recovery...")
                if not wait_for_manual_relogin(driver):
                    return
            last_full_recovery = time.time()

        # ---- Periodic stats ----
        if cycle % 20 == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            stats = " | ".join([f"{t}: {state[t]['captured']}" for t in timers])
            total = sum(state[t]['captured'] for t in timers)
            print(f"  [{ts}] Cycle {cycle} | {stats} | total: {total}")

        # Small delay between full cycles
        time.sleep(0.5)


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Win Go Auto-Scraper v5")
    parser.add_argument("--timer", default="all",
                        help="30sec, 1min, 3min, or 'all' (default: all)")
    parser.add_argument("--wait", type=int, default=120,
                        help="Seconds to wait for puzzle/login (default: 120)")
    args = parser.parse_args()

    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║     Win Go Auto-Scraper v5 (Multi-Timer)     ║")
    print("  ║     Auto-Login · Keep-Alive · Robust Tabs    ║")
    print("  ╚═══════════════════════════════════════════════╝")

    if not PHONE_NUMBER or not PASSWORD:
        print("\n  ⚠ WARNING: PHONE_NUMBER or PASSWORD not set in .env!")
        print("  Auto-login will be skipped. Log in manually.\n")

    driver = create_driver()

    try:
        # Auto-login
        auto_login(driver, PHONE_NUMBER, PASSWORD, args.wait)

        # Verify game iframe
        if not enter_iframe(driver):
            print("  ⚠ No iframe found. Retrying in 5s...")
            time.sleep(5)
            if not enter_iframe(driver):
                print("  Still no iframe. Waiting 30 more seconds...")
                time.sleep(30)
                if not enter_iframe(driver):
                    print("  Failed to find game. Exiting.")
                    return

        balls = get_balls(driver)
        game = get_active_game(driver)
        print(f"  Detected: {game}")
        print(f"  Balls: {balls}")

        safe_default_content(driver)

        if args.timer == "all":
            enter_iframe(driver)
            monitor_all(driver, ["30sec", "1min", "3min"])
        else:
            enter_iframe(driver)
            monitor_single(driver, args.timer)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    except Exception as e:
        print(f"\n  Fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except:
            pass
        print("\n  Done.")


if __name__ == "__main__":
    main()
