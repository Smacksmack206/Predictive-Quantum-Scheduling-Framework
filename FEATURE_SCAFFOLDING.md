# Feature Scaffolding Templates

## Quick Start: Adding a New Feature

This document provides copy-paste templates for adding new features to PQS Framework without breaking the menu bar or causing freezes.

---

## Template 1: Add New Menu Item + Dashboard

### Step 1: Add Menu Item to universal_pqs_app.py

**Location:** In `UniversalPQSApp.__init__()` method

```python
def __init__(self):
    super(UniversalPQSApp, self).__init__(APP_NAME)
    
    self.menu = [
        "System Info",
        "Run Optimization", 
        None,
        "Open Dashboard",
        "Battery Monitor",
        "Battery History",
        "Battery Guardian Stats",
        None,
        "System Control",
        None,
        "Modern Dashboard",
        "Quantum Dashboard",
        "Battery Dashboard",
        "System Control Modern",
        None,
        "YOUR_NEW_FEATURE_NAME"  # ← Add here
    ]
```

### Step 2: Add Click Handler

**Location:** After existing click handlers in `UniversalPQSApp` class

```python
@rumps.clicked("YOUR_NEW_FEATURE_NAME")
def open_your_feature(self, _):
    """Open your feature dashboard - non-blocking"""
    def open_browser():
        try:
            import webbrowser
            import time
            time.sleep(0.1)  # Prevent race conditions
            webbrowser.open('http://localhost:5002/your-feature')
        except Exception as e:
            print(f"Error opening your feature: {e}")
    threading.Thread(target=open_browser, daemon=True).start()
```

### Step 3: Add Flask Route

**Location:** In the Flask routes section of universal_pqs_app.py

```python
@flask_app.route('/your-feature')
def your_feature_dashboard():
    """Your feature dashboard"""
    return render_template('your_feature.html')
```

### Step 4: Add API Endpoint

**Location:** In the API endpoints section

```python
@flask_app.route('/api/your-feature/status')
def api_your_feature_status():
    """Your feature status API"""
    try:
        # Gather your data
        data = {
            'metric1': get_metric1(),
            'metric2': get_metric2(),
            'active': True
        }
        
        return jsonify({
            'success': True,
            'timestamp': time.time(),
            'data': data,
            'metadata': {
                'source': 'real_system',
                'version': '1.0'
            }
        })
    except Exception as e:
        logger.error(f"Your feature API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Step 5: Create Template

**Location:** `templates/your_feature.html`

```html
{% extends "base_modern.html" %}

{% block content %}
<div x-data="yourFeature()" class="space-y-8">
    <!-- Header -->
    <div>
        <h1 class="text-4xl font-bold mb-2">🎯 Your Feature Name</h1>
        <p class="text-gray-400">Feature description</p>
    </div>
    
    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Metric 1 -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex items-center justify-between mb-2">
                <span class="text-gray-400 text-sm">Metric 1</span>
                <span class="text-2xl">📊</span>
            </div>
            <div class="text-3xl font-bold text-blue-400" x-text="metric1"></div>
        </div>
        
        <!-- Metric 2 -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex items-center justify-between mb-2">
                <span class="text-gray-400 text-sm">Metric 2</span>
                <span class="text-2xl">⚡</span>
            </div>
            <div class="text-3xl font-bold text-green-400" x-text="metric2"></div>
        </div>
        
        <!-- Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex items-center justify-between mb-2">
                <span class="text-gray-400 text-sm">Status</span>
                <span class="text-2xl">✅</span>
            </div>
            <div class="text-3xl font-bold text-purple-400" x-text="status"></div>
        </div>
    </div>
    
    <!-- Action Buttons -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button @click="performAction()" 
                class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded-lg transition">
            🚀 Perform Action
        </button>
        <button @click="refreshData()" 
                class="bg-gray-700 hover:bg-gray-600 text-white font-bold py-4 px-6 rounded-lg transition">
            🔄 Refresh Data
        </button>
    </div>
</div>

<script>
function yourFeature() {
    return {
        metric1: 0,
        metric2: 0,
        status: 'Loading...',
        
        init() {
            this.fetchData();
            setInterval(() => this.fetchData(), 5000);  // Update every 5s
        },
        
        async fetchData() {
            try {
                const response = await fetch('/api/your-feature/status');
                const result = await response.json();
                
                // Handle nested structure safely
                const data = result.data || {};
                
                // Set values with fallbacks
                this.metric1 = data.metric1 || 0;
                this.metric2 = data.metric2 || 0;
                this.status = data.active ? 'Active' : 'Inactive';
                
            } catch (error) {
                console.error('Fetch error:', error);
                this.status = 'Error';
            }
        },
        
        async performAction() {
            try {
                const response = await fetch('/api/your-feature/action', {
                    method: 'POST'
                });
                const result = await response.json();
                alert(`✅ ${result.message || 'Action completed'}`);
                this.fetchData();
            } catch (error) {
                alert('❌ Action failed');
            }
        },
        
        refreshData() {
            this.fetchData();
        }
    }
}
</script>
{% endblock %}
```

---

## Template 2: Add Background Service

### Step 1: Create Service Module

**Location:** New file `your_service.py`

```python
"""
Your Service Module
Description of what this service does
"""
import threading
import time
import logging

logger = logging.getLogger(__name__)

class YourService:
    def __init__(self):
        """Initialize your service"""
        self.running = False
        self.thread = None
        logger.info("🔧 Your Service initialized")
    
    def start(self):
        """Start the service in background"""
        if self.running:
            logger.warning("Service already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("✅ Your Service started")
    
    def stop(self):
        """Stop the service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("🛑 Your Service stopped")
    
    def _run_loop(self):
        """Main service loop"""
        while self.running:
            try:
                # Your service logic here
                self._do_work()
                
                # Sleep between iterations
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Service error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _do_work(self):
        """Perform service work"""
        # Your actual work here
        pass

# Global instance
_service_instance = None

def get_service():
    """Get or create service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = YourService()
    return _service_instance
```

### Step 2: Integrate in universal_pqs_app.py

**Location:** At module level (after imports)

```python
# Your Service Integration
try:
    from your_service import get_service, YourService
    YOUR_SERVICE_AVAILABLE = True
    print("✅ Your Service loaded successfully")
except ImportError as e:
    YOUR_SERVICE_AVAILABLE = False
    print(f"⚠️ Your Service not available: {e}")
```

**Location:** In module initialization section

```python
# Start your service in background
your_service = None

if YOUR_SERVICE_AVAILABLE:
    def start_your_service():
        global your_service
        try:
            your_service = get_service()
            your_service.start()
            print("✅ Your Service started")
        except Exception as e:
            print(f"❌ Your Service failed: {e}")
    
    threading.Thread(target=start_your_service, daemon=True).start()
    print("⏳ Your Service starting in background...")
```

---

## Template 3: Add Configuration Option

### Step 1: Update config.py

```python
@dataclass
class YourFeatureConfig:
    """Your feature configuration"""
    enabled: bool = True
    interval: int = 10  # seconds
    threshold: float = 5.0
    mode: str = 'auto'

@dataclass
class PQSConfig:
    """Main PQS Framework configuration"""
    quantum: QuantumConfig
    idle: IdleConfig
    battery: BatteryConfig
    ui: UIConfig
    your_feature: YourFeatureConfig  # ← Add here
    
    @classmethod
    def default(cls) -> 'PQSConfig':
        """Default configuration"""
        return cls(
            quantum=QuantumConfig(),
            idle=IdleConfig(),
            battery=BatteryConfig(),
            ui=UIConfig(),
            your_feature=YourFeatureConfig()  # ← Add here
        )
```

### Step 2: Update config.json

```json
{
  "quantum": { ... },
  "idle": { ... },
  "battery": { ... },
  "ui": { ... },
  "your_feature": {
    "enabled": true,
    "interval": 10,
    "threshold": 5.0,
    "mode": "auto"
  }
}
```

### Step 3: Use Configuration

```python
from config import config

# Access your config
if config.your_feature.enabled:
    interval = config.your_feature.interval
    threshold = config.your_feature.threshold
```

---

## Template 4: Add API Action Endpoint

```python
@flask_app.route('/api/your-feature/action', methods=['POST'])
def api_your_feature_action():
    """Perform action for your feature"""
    try:
        # Get request data
        data = request.json or {}
        param1 = data.get('param1', 'default')
        
        # Perform action
        result = perform_your_action(param1)
        
        # Return success
        return jsonify({
            'success': True,
            'message': 'Action completed successfully',
            'result': result,
            'timestamp': time.time()
        })
        
    except ValueError as e:
        # Handle validation errors
        return jsonify({
            'success': False,
            'error': f'Invalid input: {e}'
        }), 400
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Action error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

---

## Checklist: Adding New Feature

Use this checklist when adding any new feature:

### Planning Phase
- [ ] Define feature requirements
- [ ] Identify data sources
- [ ] Plan API structure
- [ ] Design UI layout

### Implementation Phase
- [ ] Add menu item to `__init__` menu list
- [ ] Create click handler with background thread
- [ ] Add Flask route for dashboard
- [ ] Add API endpoint(s) with error handling
- [ ] Create template extending `base_modern.html`
- [ ] Add Alpine.js component with data fetching
- [ ] Update configuration if needed

### Testing Phase
- [ ] Test menu item doesn't freeze
- [ ] Test API returns correct data structure
- [ ] Test template renders without errors
- [ ] Test real data appears (not zeros/blanks)
- [ ] Test error handling (disconnect API, etc.)
- [ ] Test on fresh restart

### Documentation Phase
- [ ] Add feature to README
- [ ] Document API endpoints
- [ ] Add configuration options
- [ ] Update user guide

---

## Common Patterns

### Pattern: Periodic Data Update

```javascript
init() {
    this.fetchData();
    // Update every 5 seconds
    setInterval(() => this.fetchData(), 5000);
}
```

### Pattern: Manual Refresh Button

```javascript
async refreshData() {
    this.loading = true;
    await this.fetchData();
    this.loading = false;
}
```

### Pattern: Action with Confirmation

```javascript
async performAction() {
    if (!confirm('Are you sure?')) return;
    
    try {
        const response = await fetch('/api/action', { method: 'POST' });
        const result = await response.json();
        alert(`✅ ${result.message}`);
    } catch (error) {
        alert('❌ Action failed');
    }
}
```

### Pattern: Loading State

```javascript
data() {
    return {
        loading: false,
        data: null
    }
},

async fetchData() {
    this.loading = true;
    try {
        const response = await fetch('/api/data');
        this.data = await response.json();
    } finally {
        this.loading = false;
    }
}
```

---

## Quick Commands

### Test New Feature

```bash
# 1. Check syntax
python3 -m py_compile universal_pqs_app.py

# 2. Check template
python3 -c "from jinja2 import Environment, FileSystemLoader; \
env = Environment(loader=FileSystemLoader('templates')); \
env.get_template('your_feature.html')"

# 3. Start PQS
pqs

# 4. Test API
curl http://localhost:5002/api/your-feature/status | python3 -m json.tool

# 5. Open dashboard
open http://localhost:5002/your-feature
```

### Debug Issues

```bash
# Check Flask routes
python3 -c "from universal_pqs_app import flask_app; \
[print(rule) for rule in flask_app.url_map.iter_rules() if 'your-feature' in str(rule)]"

# Check menu items
python3 -c "from universal_pqs_app import UniversalPQSApp; \
app = UniversalPQSApp(); print(app.menu)"

# Test API endpoint
curl -X POST http://localhost:5002/api/your-feature/action \
  -H "Content-Type: application/json" \
  -d '{"param1": "value"}'
```

---

## Remember

1. **Always use background threads** for menu click handlers
2. **Always add error handling** with try-except blocks
3. **Always test** before committing
4. **Always handle nested** API responses safely
5. **Always add fallback values** for all data fields

**When in doubt, copy from existing working examples!**
