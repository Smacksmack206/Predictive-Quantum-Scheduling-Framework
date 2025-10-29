"""PQS Framework main entry point"""
import sys
import os
import threading
import builtins

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Main entry point with native window and GUI engine selection"""
    from universal_pqs_app import UniversalPQSApp, start_flask_server
    from native_window import show_engine_selection_alert, PQSWindowController
    import rumps
    
    print("🚀 Starting PQS Framework...")
    
    # Start Ultra Idle Battery Optimizer
    try:
        from ultra_idle_battery_optimizer import get_ultra_optimizer
        ultra_optimizer = get_ultra_optimizer()
        ultra_optimizer.start()
        print("🔋 Ultra Idle Battery Optimizer started")
    except Exception as e:
        print(f"⚠️ Ultra optimizer not available: {e}")
    
    # Start Flask in background
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Create menu bar app
    app = UniversalPQSApp()
    print("✅ Menu bar app created")
    
    # Show engine selection and window after 2 seconds
    def show_window_on_main_thread(timer):
        timer.stop()
        try:
            # Show engine selection alert first
            def on_engine_selected():
                # After engine selection, show main window
                app.window_controller = PQSWindowController.alloc().init()
                app.window_controller.show()
                print("✅ Native window shown")
            
            # Use the NSAlert modal for engine selection
            show_engine_selection_alert(on_engine_selected)
            print("✅ Engine selected")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    rumps.Timer(show_window_on_main_thread, 2).start()
    
    # Run menu bar app (blocking)
    print("🎯 Starting menu bar app...")
    app.run()

if __name__ == "__main__":
    main()
