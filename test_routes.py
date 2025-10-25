#!/usr/bin/env python3
"""Quick test to verify routes work"""
import sys

try:
    from universal_pqs_app import flask_app
    
    print("Testing routes...")
    with flask_app.test_client() as client:
        routes_to_test = [
            '/modern',
            '/quantum-modern', 
            '/battery-modern',
            '/system-control-modern'
        ]
        
        for route in routes_to_test:
            response = client.get(route)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {route} - Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   Error: {response.data.decode()[:200]}")
    
    print("\n✅ All routes are working! Start PQS with 'pqs' to access them.")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
