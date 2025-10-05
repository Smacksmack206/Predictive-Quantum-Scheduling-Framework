import rumps
import psutil
import subprocess
import time
import threading
from flask import Flask, render_template, jsonify, request
import json
import os
import signal

# --- Configuration ---
APP_NAME = "Battery Optimizer"
CONFIG_FILE = os.path.expanduser("~/.battery_optimizer_config.json")
DEFAULT_CONFIG = {
    "enabled": True,
    "amphetamine_mode": False,
    "apps_to_manage": [
        "Android Studio", "Docker", "Xcode-beta", "Warp", "Raycast", 
        "Postman Agent", "Visual Studio Code", "Google Chrome", 
        "Brave Browser", "ChatGPT", "Obsidian", "Figma", "Messenger", 
        "BlueBubbles", "WebTorrent", "OneDrive", "Slack"
    ],
    "terminal_exceptions": [
        "Terminal", "iTerm", "Warp", "Hyper", "Alacritty", "kitty",
        "AWS", "kiro", "void", "tmux", "screen"
    ],
    "cpu_threshold_percent": 10,
    "ram_threshold_mb": 200,
    "network_threshold_kbps": 50,
    "idle_tiers": {
        "high_battery": {"level": 80, "idle_seconds": 600}, # 10 mins
        "medium_battery": {"level": 40, "idle_seconds": 300}, # 5 mins
        "low_battery": {"level": 0, "idle_seconds": 120} # 2 mins
    }
}

# --- State Management ---
class AppState:
    def __init__(self):
        self.suspended_pids = {} # {pid: name}
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG
            self.save_config()

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def is_enabled(self):
        return self.config.get("enabled", True)

    def toggle_enabled(self):
        self.config["enabled"] = not self.is_enabled()
        self.save_config()
        return self.is_enabled()

state = AppState()

# --- Core Logic ---

def get_shell_output(command):
    try:
        return subprocess.check_output(command, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def is_on_battery():
    return 'Battery Power' in get_shell_output("pmset -g batt")

def get_battery_level():
    output = get_shell_output("pmset -g batt")
    try:
        level_str = output.split(';')[0].split('\t')[-1].replace('%', '')
        return int(level_str)
    except (ValueError, IndexError):
        return 100 # Assume full if parsing fails

def get_idle_time():
    output = get_shell_output("ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1000000000; exit}'")
    try:
        return float(output)
    except ValueError:
        return 0

def is_display_off():
    """Check if display is off (lid closed or display sleep)"""
    try:
        brightness = get_shell_output("brightness -l | grep 'display 0' | awk '{print $4}'")
        return brightness == "0.000000" or brightness == ""
    except:
        return False

def has_active_terminals():
    """Check if any terminal/development apps are running"""
    terminal_apps = state.config.get("terminal_exceptions", [])
    for proc in psutil.process_iter(['name']):
        try:
            p_name = proc.info['name']
            if any(term.lower() in p_name.lower() for term in terminal_apps):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def is_screen_locked():
    # Check if screen is locked using multiple methods
    try:
        # Method 1: Check screensaver status
        screensaver_output = get_shell_output("ps aux | grep -i screensaver | grep -v grep")
        if "ScreenSaverEngine" in screensaver_output:
            return True
        
        # Method 2: Check if loginwindow is active (indicates lock screen)
        loginwindow_output = get_shell_output("ps aux | grep loginwindow | grep -v grep")
        if loginwindow_output and "console" in loginwindow_output:
            # Additional check for actual lock state
            lock_check = get_shell_output("python3 -c \"import Quartz; print(Quartz.CGSessionCopyCurrentDictionary())\" 2>/dev/null")
            if "CGSSessionScreenIsLocked = 1" in lock_check:
                return True
        
        return False
    except:
        return False

def get_dynamic_idle_timeout():
    level = get_battery_level()
    tiers = state.config["idle_tiers"]
    if level > tiers["high_battery"]["level"]:
        return tiers["high_battery"]["idle_seconds"]
    if level > tiers["medium_battery"]["level"]:
        return tiers["medium_battery"]["idle_seconds"]
    return tiers["low_battery"]["idle_seconds"]

def check_and_manage_apps():
    if not state.is_enabled():
        resume_all_apps()
        return

    # Amphetamine Mode: Suspend apps when display is off but keep terminals running
    if state.config.get("amphetamine_mode", False):
        display_off = is_display_off()
        if display_off and is_on_battery():
            suspend_apps_except_terminals("Amphetamine mode - display off")
            return
        elif not display_off:
            resume_non_terminal_apps()
            return

    # Check if screen is locked - suspend immediately if locked and on battery
    screen_locked = is_screen_locked()
    if screen_locked and is_on_battery():
        suspend_resource_heavy_apps("Screen locked")
        return
    
    # If not on battery, resume all apps
    if not is_on_battery():
        resume_all_apps()
        return

    idle_time = get_idle_time()
    timeout = get_dynamic_idle_timeout()

    if idle_time < timeout:
        resume_all_apps()
        return

    # Idle timeout exceeded, check for apps to suspend
    suspend_resource_heavy_apps("Idle timeout exceeded")

def suspend_apps_except_terminals(reason):
    """Suspend all managed apps except terminal/development apps"""
    terminal_apps = state.config.get("terminal_exceptions", [])
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            p_name = proc.info['name']
            p_pid = proc.info['pid']

            # Skip if already suspended
            if p_pid in state.suspended_pids:
                continue

            # Check if it's a managed app
            if any(app_name.lower() in p_name.lower() for app_name in state.config["apps_to_manage"]):
                # Skip terminal/development apps
                if any(term.lower() in p_name.lower() for term in terminal_apps):
                    continue
                
                print(f"Suspending {p_name} (PID: {p_pid}) - {reason}")
                os.kill(p_pid, signal.SIGSTOP)
                state.suspended_pids[p_pid] = p_name

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def resume_non_terminal_apps():
    """Resume only non-terminal apps when display comes back on"""
    terminal_apps = state.config.get("terminal_exceptions", [])
    
    for pid, name in list(state.suspended_pids.items()):
        # Only resume if it's not a terminal app
        if not any(term.lower() in name.lower() for term in terminal_apps):
            try:
                os.kill(pid, signal.SIGCONT)
                print(f"Resumed {name} (PID: {pid})")
                del state.suspended_pids[pid]
            except ProcessLookupError:
                del state.suspended_pids[pid]

def suspend_resource_heavy_apps(reason)::
    """Suspend apps based on resource usage or immediately if locked"""
    net_io_before = psutil.net_io_counters()
    time.sleep(1)

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            p_name = proc.info['name']
            p_pid = proc.info['pid']

            if any(app_name.lower() in p_name.lower() for app_name in state.config["apps_to_manage"]):
                if p_pid in state.suspended_pids:
                    continue

                # If screen is locked, suspend all managed apps regardless of resource usage
                if "locked" in reason.lower():
                    print(f"Suspending {p_name} (PID: {p_pid}) - {reason}")
                    os.kill(p_pid, signal.SIGSTOP)
                    state.suspended_pids[p_pid] = p_name
                    continue

                # Check resource usage for idle timeout
                cpu_usage = proc.cpu_percent(interval=0.1)
                ram_usage_mb = proc.memory_info().rss / (1024 * 1024)
                
                net_io_after = psutil.net_io_counters()
                bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent
                bytes_recv = net_io_after.bytes_recv - net_io_before.bytes_recv
                network_kbytes_sec = (bytes_sent + bytes_recv) / 1024

                if (cpu_usage > state.config["cpu_threshold_percent"] or
                    ram_usage_mb > state.config["ram_threshold_mb"] or
                    network_kbytes_sec > state.config["network_threshold_kbps"]):
                    
                    print(f"Suspending {p_name} (PID: {p_pid}) - {reason} - CPU: {cpu_usage}%, RAM: {ram_usage_mb:.2f}MB, NET: {network_kbytes_sec:.2f} KB/s")
                    os.kill(p_pid, signal.SIGSTOP)
                    state.suspended_pids[p_pid] = p_name

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def resume_all_apps():
    if not state.suspended_pids:
        return
    
    print("Resuming all suspended applications...")
    for pid, name in list(state.suspended_pids.items()):
        try:
            os.kill(pid, signal.SIGCONT)
            print(f"Resumed {name} (PID: {pid})")
        except ProcessLookupError:
            pass # Process already gone
        del state.suspended_pids[pid]

# --- Web Server (Flask) ---
flask_app = Flask(__name__, template_folder='templates')

@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/api/status')
def api_status():
    return jsonify({
        "enabled": state.is_enabled(),
        "on_battery": is_on_battery(),
        "screen_locked": is_screen_locked(),
        "battery_level": get_battery_level(),
        "idle_time": get_idle_time(),
        "current_timeout": get_dynamic_idle_timeout(),
        "suspended_apps": list(state.suspended_pids.values())
    })

@flask_app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        new_config = request.json
        state.config.update(new_config)
        state.save_config()
        return jsonify({"success": True, "message": "Configuration saved."})
    return jsonify(state.config)

@flask_app.route('/api/toggle', methods=['POST'])
def api_toggle():
    is_now_enabled = state.toggle_enabled()
    if not is_now_enabled:
        resume_all_apps()
    return jsonify({"enabled": is_now_enabled})

def run_flask():
    # Use a production-ready server if available, otherwise fallback to dev server
    try:
        from waitress import serve
        serve(flask_app, host='0.0.0.0', port=9010)
    except ImportError:
        flask_app.run(host='0.0.0.0', port=9010, debug=False)

# --- Status Bar App (rumps) ---
class BatteryOptimizerApp(rumps.App):
    def __init__(self):
        super(BatteryOptimizerApp, self).__init__(APP_NAME, icon=None, title="⚡")
        # Store menu items that need updating
        self.status_item = rumps.MenuItem("Service Status: Enabled")
        
        self.menu = [
            self.status_item,
            "Toggle Service",
            "Toggle Amphetamine Mode",
            None,
            "View Suspended Apps",
            None,
            "Open Dashboard",
        ]
        self.check_timer = rumps.Timer(self.run_check, 15) # Check every 15 seconds
        self.check_timer.start()

    def run_check(self, _):
        check_and_manage_apps()
        self.update_menu()

    def update_menu(self):
        # Update the title of the existing status item
        status_text = "Enabled" if state.is_enabled() else "Disabled"
        self.status_item.title = f"Service Status: {status_text}"
        
        # Update title with contextual icons for each mode
        count = len(state.suspended_pids)
        locked = is_screen_locked()
        amphetamine_mode = state.config.get("amphetamine_mode", False)
        
        if amphetamine_mode:
            if locked:
                self.title = f"🧠🔒({count})" if count > 0 else "🧠🔒"  # Brain + lock for smart locked mode
            else:
                self.title = f"🧠({count})" if count > 0 else "🧠"  # Brain for smart amphetamine mode
        elif locked:
            self.title = f"🛡️({count})" if count > 0 else "🛡️"  # Shield for protection/security mode
        elif count > 0:
            self.title = f"⏸️({count})"  # Pause for suspended apps
        else:
            self.title = "🎯"  # Target for active optimization

    @rumps.clicked("View Suspended Apps")
    def view_suspended_apps(self, _):
        if state.suspended_pids:
            apps_list = "\n".join([f"• {name} (PID: {pid})" for pid, name in state.suspended_pids.items()])
            rumps.alert("Suspended Apps", apps_list)
        else:
            rumps.alert("Suspended Apps", "No apps are currently suspended")

    @rumps.clicked("Toggle Amphetamine Mode")
    def toggle_amphetamine_mode(self, _):
        state.config["amphetamine_mode"] = not state.config.get("amphetamine_mode", False)
        state.save_config()
        mode_status = "ON" if state.config["amphetamine_mode"] else "OFF"
        rumps.alert("Amphetamine Mode", f"Amphetamine Mode is now {mode_status}\n\nThis mode suspends apps when display is off but keeps terminals running for background tasks.")
        self.update_menu()

    @rumps.clicked("Toggle Service")
    def toggle_service(self, _):
        state.toggle_enabled()
        if not state.is_enabled():
            resume_all_apps()
        self.update_menu()

    @rumps.clicked("Open Dashboard")
    def open_dashboard(self, _):
        subprocess.call(["open", "http://localhost:9010"])

    def quit_app(self, _):
        resume_all_apps()
        rumps.quit_application()

if __name__ == '__main__':
    import traceback
    LOG_FILE = '/Users/home/m3.macbook.air/startup_error.log'

    try:
        # Run Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()

        # Run rumps app
        app = BatteryOptimizerApp()
        app.run()

    except Exception as e:
        with open(LOG_FILE, 'w') as f:
            f.write(f"Application failed to start:\n")
            f.write(traceback.format_exc())
        # Also print to stderr for launchctl logs
        print(traceback.format_exc())
