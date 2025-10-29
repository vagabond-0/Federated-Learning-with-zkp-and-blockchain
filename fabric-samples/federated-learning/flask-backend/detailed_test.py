#!/usr/bin/env python3

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

class TestRunner:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
    
    def print_header(self, text):
        print(f"\n{Colors.BLUE}{'='*50}")
        print(f"{text}")
        print(f"{'='*50}{Colors.END}\n")
    
    def run_test(self, name, method, url, data=None, expected_status=200, expected_in_response=None):
        self.tests_run += 1
        print(f"{Colors.YELLOW}🧪 Test {self.tests_run}: {name}{Colors.END}")
        
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            elif method == "PUT":
                response = requests.put(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Print response
            print(f"Status Code: {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            
            # Check status
            status_ok = response.status_code == expected_status
            
            # Check content if specified
            content_ok = True
            if expected_in_response:
                content_ok = expected_in_response in str(response.text)
            
            if status_ok and content_ok:
                print(f"{Colors.GREEN}✅ PASSED{Colors.END}\n")
                self.tests_passed += 1
                return True
            else:
                print(f"{Colors.RED}❌ FAILED{Colors.END}")
                if not status_ok:
                    print(f"Expected status {expected_status}, got {response.status_code}")
                if not content_ok:
                    print(f"Expected '{expected_in_response}' in response")
                print()
                self.tests_failed += 1
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"{Colors.RED}❌ FAILED - Cannot connect to {url}{Colors.END}")
            print(f"{Colors.YELLOW}Make sure Flask server is running on port 5000{Colors.END}\n")
            self.tests_failed += 1
            return False
        except Exception as e:
            print(f"{Colors.RED}❌ FAILED - {str(e)}{Colors.END}\n")
            self.tests_failed += 1
            return False
    
    def print_summary(self):
        self.print_header("TEST SUMMARY")
        print(f"Total Tests: {self.tests_run}")
        print(f"{Colors.GREEN}Passed: {self.tests_passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.tests_failed}{Colors.END}")
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%\n")

def main():
    runner = TestRunner()
    runner.print_header("VPSA Flask Backend - Detailed API Tests")
    
    # Test 1: Health Check
    runner.run_test(
        "Health Check",
        "GET",
        f"{BASE_URL}/health",
        expected_in_response="healthy"
    )
    time.sleep(1)
    
    # Test 2: Get Initial Config
    runner.run_test(
        "Get Initial Configuration",
        "GET",
        f"{BASE_URL}/api/config",
        expected_in_response="configID"
    )
    time.sleep(1)
    
    # Test 3: Register Source Client 1
    runner.run_test(
        "Register Source Client 1",
        "POST",
        f"{BASE_URL}/api/client/register",
        data={
            "clientID": "py-source-1",
            "domain": "source",
            "datasetSize": 10000
        },
        expected_status=201,
        expected_in_response="registered"
    )
    time.sleep(3)
    
    # Test 4: Register Source Client 2
    runner.run_test(
        "Register Source Client 2",
        "POST",
        f"{BASE_URL}/api/client/register",
        data={
            "clientID": "py-source-2",
            "domain": "source",
            "datasetSize": 9500
        },
        expected_status=201
    )
    time.sleep(3)
    
    # Test 5: Register Target Client
    runner.run_test(
        "Register Target Client",
        "POST",
        f"{BASE_URL}/api/client/register",
        data={
            "clientID": "py-target-1",
            "domain": "target",
            "datasetSize": 8000
        },
        expected_status=201
    )
    time.sleep(3)
    
    # Test 6: Get All Clients
    runner.run_test(
        "Get All Registered Clients",
        "GET",
        f"{BASE_URL}/api/clients",
        expected_in_response="py-source-1"
    )
    time.sleep(1)
    
    # Test 7: Submit Source Model 1
    runner.run_test(
        "Submit Source Model 1",
        "POST",
        f"{BASE_URL}/api/model/submit",
        data={
            "modelID": "py-model-s1-r0",
            "clientID": "py-source-1",
            "weights": {"layer1": [0.5, 0.3, 0.8], "layer2": [0.2, 0.9, 0.4]},
            "latentFeatures": {"latent_dim": 768, "features": [0.1, 0.2, 0.3]},
            "prototypes": {"class1": [0.9, 0.1], "class2": [0.2, 0.8]},
            "accuracy": 0.87,
            "loss": 0.13,
            "alignmentLoss": 0.04,
            "dataSize": 1000
        },
        expected_status=201
    )
    time.sleep(3)
    
    # Test 8: Submit Source Model 2
    runner.run_test(
        "Submit Source Model 2",
        "POST",
        f"{BASE_URL}/api/model/submit",
        data={
            "modelID": "py-model-s2-r0",
            "clientID": "py-source-2",
            "weights": {"layer1": [0.6, 0.4, 0.7], "layer2": [0.3, 0.8, 0.5]},
            "latentFeatures": {"latent_dim": 768, "features": [0.15, 0.25, 0.35]},
            "prototypes": {"class1": [0.85, 0.15], "class2": [0.25, 0.75]},
            "accuracy": 0.85,
            "loss": 0.15,
            "alignmentLoss": 0.05,
            "dataSize": 950
        },
        expected_status=201
    )
    time.sleep(3)
    
    # Test 9: Submit Target Model
    runner.run_test(
        "Submit Target Model",
        "POST",
        f"{BASE_URL}/api/model/submit",
        data={
            "modelID": "py-model-t1-r0",
            "clientID": "py-target-1",
            "weights": {"layer1": [0.4, 0.6, 0.5], "layer2": [0.7, 0.3, 0.8]},
            "latentFeatures": {"latent_dim": 768, "features": [0.12, 0.22, 0.32]},
            "prototypes": {"class1": [0.88, 0.12], "class2": [0.18, 0.82]},
            "accuracy": 0.79,
            "loss": 0.21,
            "alignmentLoss": 0.08,
            "dataSize": 800
        },
        expected_status=201
    )
    time.sleep(3)
    
    # Test 10: Aggregate Models
    runner.run_test(
        "Aggregate Models (VPSA Algorithm)",
        "POST",
        f"{BASE_URL}/api/aggregate",
        data={
            "modelIDs": ["py-model-s1-r0", "py-model-s2-r0", "py-model-t1-r0"],
            "sourceWeight": 0.6,
            "targetWeight": 0.4,
            "alignmentWeight": 0.1
        },
        expected_in_response="aggregated"
    )
    time.sleep(3)
    
    # Test 11: Get Global Model
    runner.run_test(
        "Get Updated Global Model",
        "GET",
        f"{BASE_URL}/api/global-model",
        expected_in_response="vpsa-global-model"
    )
    time.sleep(1)
    
    # Test 12: Get Metrics for Round 0
    runner.run_test(
        "Get Training Metrics (Round 0)",
        "GET",
        f"{BASE_URL}/api/metrics/0",
        expected_in_response="round"
    )
    time.sleep(1)
    
    # Print Summary
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.tests_failed == 0 else 1)

if __name__ == "__main__":
    main()