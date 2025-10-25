# PQS Framework Development Guidelines

## Critical Lessons Learned

### Menu Bar Freeze Issues - Root Causes & Solutions

#### Problem 1: Dynamic Menu Loading
**Issue:** Attempting to modify `self.menu` after initialization caused rumps to freeze.

**Solution:** Set complete menu structure in `__init__` immediately.

```python
# ❌ WRONG - Causes freeze
def __init__(self):
    super().__init__(APP_NAME)
    self.menu = ["Basic Item"]
    threading.Thread(target=self.setup_full_menu).start()  # DON'T DO THIS

# ✅ CORRECT - Works perfectly
def __init__(self):
    super().__init__(APP_NAME)
    self.menu = [
        "Item 1",
        "Item 2",
        None,  # Separator
        "Item 3"
    ]
```

#### Problem 2: Blocking Operations in Click Handlers
**Issue:** Any synchronous operation in a click handler freezes the menu bar.

**Solution:** Always wrap operations in background threads.

```python
# ❌ WRONG - Blocks menu bar
@rumps.clicked("Do Something")
def do_something(self, _):
    webbrowser.open('http://localhost:5002')  # BLOCKS!

# ✅ CORRECT - Non-blocking
@rumps.clicked("Do Something")
def do_something(self, _):
    def open_browser():
        try:
            import webbrowser
            import time
            time.sleep(0.1)  # Small delay prevents race conditions
            webbrowser.open('http://localhost:5002')
        except Exception as e:
            print(f"Error: {e}")
    threading.Thread(target=open_browser, daemon=True).start()
```

#### Problem 3: Heavy Initialization in __init__
**Issue:** Long-running initialization blocks menu bar appearance.

**Solution:** Move heavy operations to background threads.

```python
# ❌ WRONG - Blocks initialization
def __init__(self):
    super().__init__(APP_NAME)
    self.service = HeavyService()  # BLOCKS!
    self.service.start()  # BLOCKS!

# ✅ CORRECT - Non-blocking initialization
def __init__(self):
    super().__init__(APP_NAME)
    self.service = None
    
    def init_service():
        try:
            self.service = HeavyService()
            self.service.start()
        except Exception as e:
            logger.error(f"Service init failed: {e}")
    
    threading.Thread(target=init_service, daemon=True).start()
```

#### Problem 4: API Response Structure Mismatch
**Issue:** Templates expected flat data structure, but API returned nested objects.

**Solution:** Always check actual API response structure and map correctly.

```javascript
// ❌ WRONG - Assumes flat structure
const data = await fetch('/api/status').then(r => r.json());
this.energySaved = data.energy_saved;  // undefined if nested!

// ✅ CORRECT - Handles nested structure
const data = await fetch('/api/status').then(r => r.json());
const stats = data.stats || {};
this.energySaved = stats.energy_saved || 0;
```

---

## Design Patterns - Mandatory Rules

### Rule 1: All Menu Click Handlers Must Be Non-Blocking

**Template:**
```python
@rumps.clicked("Menu Item Name")
def menu_item_handler(self, _):
    """Description of what this does"""
    def background_task():
        try:
            import time
            time.sleep(0.1)  # Prevent race conditions
            # Your actual code here
            result = do_something()
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    threading.Thread(target=background_task, daemon=True).start()
```

### Rule 2: All Heavy Initialization Must Be Async

**Template:**
```python
def __init__(self):
    super().__init__(APP_NAME)
    
    # Set menu immediately
    self.menu = ["Item 1", "Item 2"]
    
    # Initialize heavy services in background
    self.heavy_service = None
    
    def init_heavy_service():
        try:
            self.heavy_service = HeavyService()
            self.heavy_service.start()
            logger.info("✅ Service started")
        except Exception as e:
            logger.error(f"❌ Service failed: {e}")
    
    threading.Thread(target=init_heavy_service, daemon=True).start()
```

### Rule 3: All API Endpoints Must Return Consistent Structure

**Template:**
```python
@flask_app.route('/api/feature/status')
def api_feature_status():
    """Feature status API"""
    try:
        # Gather data
        data = get_feature_data()
        
        # Return consistent structure
        return jsonify({
            'success': True,
            'timestamp': time.time(),
            'data': {
                'metric1': data.metric1,
                'metric2': data.metric2
            },
            'metadata': {
                'source': 'real_system',
                'version': '1.0'
            }
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Rule 4: All Templates Must Handle Missing Data

**Template:**
```javascript
async fetchData() {
    try {
        const response = await fetch('/api/feature/status');
        const result = await response.json();
        
        // Handle nested structure safely
        const data = result.data || {};
        const metadata = result.metadata || {};
        
        // Use fallback values
        this.metric1 = data.metric1 || 0;
        this.metric2 = data.metric2 || 'N/A';
        this.source = metadata.source || 'unknown';
        
    } catch (error) {
        console.error('Fetch error:', error);
        // Set safe defaults on error
        this.metric1 = 0;
        this.metric2 = 'Error';
    }
}
```

---

## Module-Level Initialization Rules

### Rule 5: No Blocking Operations at Module Load

**What NOT to do:**
```python
# ❌ WRONG - Blocks on import
idle_manager = get_idle_manager()
idle_manager.start_monitoring()  # BLOCKS!
```

**Correct approach:**
```python
# ✅ CORRECT - Non-blocking
idle_manager = None

def start_idle_manager():
    global idle_manager
    try:
        idle_manager = get_idle_manager()
        idle_manager.start_monitoring()
        print("✅ Idle manager started")
    except Exception as e:
        print(f"❌ Idle manager failed: {e}")

# Start in background
threading.Thread(target=start_idle_manager, daemon=True).start()
print("⏳ Idle manager starting in background...")
```

---

## Error Handling Standards

### Rule 6: Every Operation Must Have Error Handling

**Template:**
```python
def risky_operation():
    """Operation that might fail"""
    try:
        # Attempt operation
        result = do_something_risky()
        
        # Validate result
        if not result:
            raise ValueError("Operation returned no result")
        
        # Log success
        logger.info(f"✅ Operation succeeded: {result}")
        return result
        
    except SpecificException as e:
        # Handle specific errors
        logger.error(f"❌ Specific error: {e}")
        return default_value
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return default_value
```

---

## Testing Requirements

### Rule 7: Test Before Committing

**Checklist:**
- [ ] No syntax errors: `python3 -m py_compile file.py`
- [ ] Menu bar appears within 2 seconds
- [ ] All menu items clickable without freeze
- [ ] All API endpoints return 200 status
- [ ] Templates render without errors
- [ ] Real data appears in UI (not zeros/blanks)

**Quick test script:**
```bash
# Test syntax
python3 -m py_compile universal_pqs_app.py

# Test templates
python3 -c "from jinja2 import Environment, FileSystemLoader; \
env = Environment(loader=FileSystemLoader('templates')); \
env.get_template('dashboard_modern.html')"

# Test API (after starting pqs)
curl http://localhost:5002/api/status | python3 -m json.tool
```

---

## Common Pitfalls to Avoid

### Pitfall 1: Autofix Breaking Code
**Problem:** IDE autofix can reformat code incorrectly, especially indentation.

**Solution:** 
- Always review autofix changes
- Test immediately after autofix
- Keep backups of working code

### Pitfall 2: Assuming API Structure
**Problem:** Templates break when API structure changes.

**Solution:**
- Always inspect actual API response first
- Use safe navigation: `data.stats?.energy_saved || 0`
- Add fallback values for all fields

### Pitfall 3: Forgetting Background Threads
**Problem:** New menu items freeze because they're not threaded.

**Solution:**
- Use the templates provided in this document
- Never call blocking operations directly
- Always wrap in `threading.Thread(..., daemon=True)`

### Pitfall 4: Not Testing Menu Bar
**Problem:** Code works but menu bar freezes.

**Solution:**
- Always test by clicking every menu item
- Watch for console errors
- Verify menu appears within 2 seconds

---

## Quick Reference

### Adding a New Menu Item
1. Add to menu list in `__init__`
2. Create click handler with background thread
3. Add error handling
4. Test by clicking

### Adding a New API Endpoint
1. Define route with `@flask_app.route`
2. Return consistent JSON structure
3. Add error handling with try-except
4. Test with curl

### Adding a New Template
1. Extend `base_modern.html`
2. Use Alpine.js for reactivity
3. Handle nested API responses
4. Add fallback values for all data

### Adding Background Service
1. Initialize as `None` in `__init__`
2. Create init function with try-except
3. Start in background thread
4. Add logging for success/failure

---

## Version History

**v2.0 - Modern UI Implementation**
- Added 4 modern dashboard routes
- Fixed menu bar freeze issues
- Implemented proper threading patterns
- Added comprehensive error handling
- Created development guidelines

**Key Changes:**
- Menu initialization: Dynamic → Static
- Click handlers: Synchronous → Async
- API responses: Flat → Nested (handled)
- Error handling: Minimal → Comprehensive
- Background services: Blocking → Non-blocking

---

## Contact & Support

If you encounter issues:
1. Check this document first
2. Review the templates
3. Test with provided scripts
4. Check console logs for errors

Remember: **When in doubt, use background threads and error handling!**
