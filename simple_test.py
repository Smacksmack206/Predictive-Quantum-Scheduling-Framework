#!/usr/bin/env python3
import rumps
import time

class TestApp(rumps.App):
    def __init__(self):
        super(TestApp, self).__init__("Battery Test", title="⚡")
        self.menu = ["Test Item"]
        print("Menu bar app initialized")

    @rumps.clicked("Test Item")
    def test_click(self, _):
        rumps.alert("Test", "Menu bar is working!")

if __name__ == "__main__":
    print("Starting test menu bar app...")
    app = TestApp()
    print("About to run app...")
    app.run()
