#!/usr/bin/env python3
"""
Native macOS Window for PQS Framework
Creates a unified GUI window with embedded web views
"""

import sys
import threading
import time
import objc
from PyObjCTools import AppHelper
from Foundation import NSObject, NSURL, NSMakeRect, NSURLRequest
from AppKit import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable,
    NSBackingStoreBuffered, NSApp, NSMenu, NSMenuItem,
    NSColor, NSVisualEffectView, NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialDark, NSSplitView, NSView, NSButton,
    NSButtonTypeSwitch, NSFont, NSTextField, NSTextAlignmentCenter,
    NSApplicationActivationPolicyRegular, NSTextAlignmentLeft,
    NSBezelBorder, NSModalResponseOK, NSModalResponseCancel,
    NSWindowStyleMaskFullSizeContentView, NSImage, NSAlert, NSAlertFirstButtonReturn,
    NSAlertSecondButtonReturn
)
from WebKit import WKWebView, WKWebViewConfiguration
import os

class PQSWindowController(NSObject):
    """Controller for the PQS main window"""
    
    def init(self):
        self = objc.super(PQSWindowController, self).init()
        if self is None:
            return None
        
        # Create main window
        frame = NSMakeRect(100, 100, 1400, 900)
        style_mask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                     NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style_mask, NSBackingStoreBuffered, False
        )
        
        self.window.setTitle_("⚛️ PQS Framework")
        self.window.setBackgroundColor_(NSColor.colorWithRed_green_blue_alpha_(0.1, 0.1, 0.15, 1.0))
        
        # Create split view (sidebar + content)
        self.split_view = NSSplitView.alloc().initWithFrame_(self.window.contentView().bounds())
        self.split_view.setVertical_(True)
        self.split_view.setDividerStyle_(1)  # Thin divider
        
        # Create sidebar
        sidebar_frame = NSMakeRect(0, 0, 250, frame.size.height)
        self.sidebar = self.create_sidebar(sidebar_frame)
        
        # Create web view for content
        web_config = WKWebViewConfiguration.alloc().init()
        content_frame = NSMakeRect(250, 0, frame.size.width - 250, frame.size.height)
        self.web_view = WKWebView.alloc().initWithFrame_configuration_(content_frame, web_config)
        
        # Set zoom level to 80% to reduce zoom
        self.web_view.setPageZoom_(0.8)
        
        # Add views to split view
        self.split_view.addSubview_(self.sidebar)
        self.split_view.addSubview_(self.web_view)
        
        # Set split view as content view
        self.window.setContentView_(self.split_view)
        
        # Load default page
        self.load_url("http://localhost:5002")
        
        # Current route
        self.current_route = "/"
        
        return self
    
    def create_sidebar(self, frame):
        """Create sidebar with navigation buttons"""
        sidebar = NSView.alloc().initWithFrame_(frame)
        sidebar.setWantsLayer_(True)
        sidebar.layer().setBackgroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.08, 0.08, 0.12, 1.0).CGColor()
        )
        
        # Navigation buttons
        routes = [
            ("Dashboard", "/"),
            ("Process Monitor", "/modern"),
            ("Quantum Dashboard", "/quantum"),
            ("Battery Monitor", "/battery-monitor"),
            ("Battery History", "/battery-history"),
            ("Battery Guardian", "/battery-guardian"),
            ("System Control", "/system-control"),
            ("Intelligent Process Monitor", "/process-monitor")
        ]
        
        # Center the buttons vertically
        button_height = 36
        button_spacing = 44
        total_height = len(routes) * button_spacing
        start_y = (frame.size.height + total_height) / 2 - button_spacing
        
        y_pos = start_y
        for name, route in routes:
            btn = self.create_nav_button(name, route, NSMakeRect(10, y_pos, 230, button_height))
            sidebar.addSubview_(btn)
            y_pos -= button_spacing
        
        return sidebar
    
    def create_nav_button(self, title, route, frame):
        """Create a navigation button"""
        btn = NSButton.alloc().initWithFrame_(frame)
        btn.setTitle_(title)
        btn.setBezelStyle_(1)  # Rounded
        btn.setTarget_(self)
        btn.setAction_("navigateTo:")
        btn.setTag_(hash(route))  # Store route in tag
        
        # Store route mapping
        if not hasattr(self, 'route_map'):
            self.route_map = {}
        self.route_map[hash(route)] = route
        
        return btn
    
    def navigateTo_(self, sender):
        """Handle navigation button click"""
        route = self.route_map.get(sender.tag(), "/")
        self.load_url(f"http://localhost:5002{route}")
        self.current_route = route
    
    def load_url(self, url):
        """Load URL in web view"""
        ns_url = NSURL.URLWithString_(url)
        request = NSURLRequest.requestWithURL_(ns_url)
        self.web_view.loadRequest_(request)
    
    def show(self):
        """Show the window"""
        self.window.center()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

def show_engine_selection_alert(callback):
    """Show a simple alert dialog for engine selection"""
    alert = NSAlert.alloc().init()
    alert.setMessageText_("⚛️ Choose Your Quantum Engine")
    alert.setInformativeText_("Select the quantum computing backend for optimization:\n\n🚀 Cirq: Lightweight, fast, recommended for daily use\n🔬 Qiskit: IBM's framework with advanced algorithms")
    alert.addButtonWithTitle_("Cirq (Recommended)")
    alert.addButtonWithTitle_("Qiskit (Experimental)")
    
    response = alert.runModal()
    
    # Store choice
    import builtins
    if response == NSAlertFirstButtonReturn:
        builtins.QUANTUM_ENGINE_CHOICE = 'cirq'
        print("✅ Selected: Cirq")
    else:
        builtins.QUANTUM_ENGINE_CHOICE = 'qiskit'
        print("✅ Selected: Qiskit")
    
    # Call callback
    if callback:
        callback()

class EngineSelectionController(NSObject):
    """Controller for engine selection modal"""
    
    def initWithCallback_(self, callback):
        self = objc.super(EngineSelectionController, self).init()
        if self is None:
            return None
        
        self.callback = callback
        self.selected_engine = 'cirq'  # Default
        self.cirq_radio = None
        self.qiskit_radio = None
        
        # Use simple alert instead of complex window
        print("🔧 Showing engine selection alert...")
        show_engine_selection_alert(callback)
        return self


class PQSAppDelegate(NSObject):
    """Application delegate"""
    
    def applicationDidFinishLaunching_(self, notification):
        """Called when app finishes launching"""
        # Wait for Flask to be ready
        time.sleep(2)
        
        # Show engine selection modal first
        self.engine_controller = EngineSelectionController.alloc().initWithCallback_(self.showMainWindow)
        self.engine_controller.show()
    
    def showMainWindow(self):
        """Show main window after engine selection"""
        self.window_controller = PQSWindowController.alloc().init()
        self.window_controller.show()
    
    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        """Quit app when window closes"""
        return True

def start_native_window():
    """Start the native macOS window"""
    try:
        # Create application
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        
        # Set app icon
        icon_path = os.path.join(os.path.dirname(__file__), 'pqs_icon.png')
        if os.path.exists(icon_path):
            icon = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if icon:
                app.setApplicationIconImage_(icon)
                print("✅ App icon set")
        else:
            print("⚠️ Icon file not found, using default")
        
        # Create and set delegate
        delegate = PQSAppDelegate.alloc().init()
        app.setDelegate_(delegate)
        
        # Create menu bar
        menubar = NSMenu.alloc().init()
        app_menu_item = NSMenuItem.alloc().init()
        menubar.addItem_(app_menu_item)
        app.setMainMenu_(menubar)
        
        app_menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit PQS", "terminate:", "q"
        )
        app_menu.addItem_(quit_item)
        app_menu_item.setSubmenu_(app_menu)
        
        # Run app
        AppHelper.runEventLoop()
        
    except Exception as e:
        print(f"Error starting native window: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_native_window()
