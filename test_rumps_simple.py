#!/usr/bin/env python3
"""
Simple test to verify rumps menu bar app works
"""
import rumps
import sys

print("🧪 Testing rumps menu bar app...")
sys.stdout.flush()

class SimpleApp(rumps.App):
    def __init__(self):
        super(SimpleApp, self).__init__("Test App")
        self.title = "🧪"
        self.menu = ["Test Item"]
        print("✅ App initialized")
        sys.stdout.flush()
    
    @rumps.clicked("Test Item")
    def test_click(self, _):
        print("✅ Menu item clicked!")
        rumps.notification("Test", "Success", "Menu bar is working!")

if __name__ == "__main__":
    print("🚀 Starting simple menu bar app...")
    print("   👀 Look for 🧪 icon in menu bar")
    sys.stdout.flush()
    
    app = SimpleApp()
    print("✅ App created, starting run loop...")
    sys.stdout.flush()
    
    app.run()
