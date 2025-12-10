from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
import logging
from datetime import datetime
import re
import base64
import math

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FABRIC_CONFIG = {
    'channel': 'vpsa-channel',
    'chaincode': 'vpsa',
    'network_path': '/home/amalendumanoj/project/Federated-Learning-with-zkp-and-blockchain/fabric-samples/test-network'
}


class FabricGateway:
    """Interface to interact with Hyperledger Fabric network"""

    def __init__(self):
        self.network_path = FABRIC_CONFIG['network_path']
        # default to org1 environment
        self.setup_environment('org1')

    def setup_environment(self, org='org1'):
        """Setup environment variables for Fabric CLI for given org ('org1' or 'org2')"""
        org_lower = org.lower()
        if org_lower == 'org1':
            msp_id = 'Org1MSP'
            peer_address = 'localhost:7051'
            org_path = 'org1.example.com'
        else:  # org2
            msp_id = 'Org2MSP'
            peer_address = 'localhost:9051'
            org_path = 'org2.example.com'

        os.environ['CORE_PEER_TLS_ENABLED'] = 'true'
        os.environ['CORE_PEER_LOCALMSPID'] = msp_id
        os.environ['CORE_PEER_TLS_ROOTCERT_FILE'] = f"{self.network_path}/organizations/peerOrganizations/{org_path}/peers/peer0.{org_path}/tls/ca.crt"
        os.environ['CORE_PEER_MSPCONFIGPATH'] = f"{self.network_path}/organizations/peerOrganizations/{org_path}/users/Admin@{org_path}/msp"
        os.environ['CORE_PEER_ADDRESS'] = peer_address
        os.environ['PATH'] = f"{self.network_path}/../bin:" + os.environ.get('PATH', '')
        os.environ['FABRIC_CFG_PATH'] = f"{self.network_path}/../config/"

    def parse_fabric_output(self, output):
        """Parse Fabric CLI output to extract actual response"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_output = ansi_escape.sub('', output)

        lines = clean_output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and (line.startswith('{') or line.startswith('[')):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        for line in reversed(lines):
            if line.strip():
                try:
                    return json.loads(line.strip())
                except Exception:
                    return line.strip()

        return output

    def invoke_chaincode_as_org(self, function, args, org='org1', transient=None, use_both_peers=True, timeout=120):
        """Invoke chaincode as specific organization"""
        # Set environment for specific org
        self.setup_environment(org)

        args_json = json.dumps({"function": function, "Args": args})

        cmd = [
            'peer', 'chaincode', 'invoke',
            '-o', 'localhost:7050',
            '--ordererTLSHostnameOverride', 'orderer.example.com',
            '--tls',
            '--cafile', f"{self.network_path}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem",
            '-C', FABRIC_CONFIG['channel'],
            '-n', FABRIC_CONFIG['chaincode'],
            '-c', args_json,
            '--waitForEvent'
        ]

        # Add peer addresses based on requirement
        if use_both_peers:
            # Use both peers for endorsement
            cmd.extend([
                '--peerAddresses', 'localhost:7051',
                '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt",
                '--peerAddresses', 'localhost:9051',
                '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
            ])
        else:
            # Use only the org's peer
            if org.lower() == 'org1':
                cmd.extend([
                    '--peerAddresses', 'localhost:7051',
                    '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
                ])
            else:
                cmd.extend([
                    '--peerAddresses', 'localhost:9051',
                    '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
                ])

        # Add transient data if provided
        if transient:
            # transient values must be base64 strings for the peer CLI; we expect caller to encode
            transient_json = json.dumps(transient)
            cmd.extend(['--transient', transient_json])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.network_path, timeout=timeout)

            # Reset to Org1 environment after operation (safe default)
            self.setup_environment('org1')

            if result.returncode != 0:
                logger.error(f"Invoke failed for {org}: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Invoke output for {org}: {result.stdout}")
            return {"success": True, "output": result.stdout}
        except subprocess.TimeoutExpired:
            self.setup_environment('org1')
            logger.error(f"Invoke timeout for {org}")
            return {"success": False, "error": "Transaction timeout"}
        except Exception as e:
            self.setup_environment('org1')
            logger.error(f"Exception during invoke for {org}: {str(e)}")
            return {"success": False, "error": str(e)}

    def invoke_chaincode(self, function, args, transient=None, timeout=120):
        """Invoke chaincode function with endorsements from both orgs (default)"""
        # Ensure environment is Org1 by default (peer CLI identity)
        self.setup_environment('org1')

        args_json = json.dumps({"function": function, "Args": args})

        cmd = [
            'peer', 'chaincode', 'invoke',
            '-o', 'localhost:7050',
            '--ordererTLSHostnameOverride', 'orderer.example.com',
            '--tls',
            '--cafile', f"{self.network_path}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem",
            '-C', FABRIC_CONFIG['channel'],
            '-n', FABRIC_CONFIG['chaincode'],
            # Org1 peer
            '--peerAddresses', 'localhost:7051',
            '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt",
            # Org2 peer
            '--peerAddresses', 'localhost:9051',
            '--tlsRootCertFiles', f"{self.network_path}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt",
            '-c', args_json,
            '--waitForEvent'
        ]

        if transient:
            transient_json = json.dumps(transient)
            cmd.extend(['--transient', transient_json])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.network_path, timeout=timeout)
            if result.returncode != 0:
                logger.error(f"Invoke failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Invoke output: {result.stdout}")
            return {"success": True, "output": result.stdout}
        except subprocess.TimeoutExpired:
            logger.error("Invoke timeout")
            return {"success": False, "error": "Transaction timeout"}
        except Exception as e:
            logger.error(f"Exception during invoke: {str(e)}")
            return {"success": False, "error": str(e)}

    def query_chaincode(self, function, args, timeout=30):
        """Query chaincode function"""
        # Ensure Org1 env
        self.setup_environment('org1')
        args_json = json.dumps({"function": function, "Args": args})

        cmd = [
            'peer', 'chaincode', 'query',
            '-C', FABRIC_CONFIG['channel'],
            '-n', FABRIC_CONFIG['chaincode'],
            '-c', args_json
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.network_path, timeout=timeout)
            if result.returncode != 0:
                logger.error(f"Query failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Query raw output: {repr(result.stdout)}")
            parsed_output = self.parse_fabric_output(result.stdout)
            logger.info(f"Query parsed output: {parsed_output}")

            return {"success": True, "data": parsed_output}
        except subprocess.TimeoutExpired:
            logger.error("Query timeout")
            return {"success": False, "error": "Query timeout"}
        except Exception as e:
            logger.error(f"Exception during query: {str(e)}")
            return {"success": False, "error": str(e)}


fabric = FabricGateway()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'network': 'Hyperledger Fabric',
        'chaincode': FABRIC_CONFIG['chaincode'],
        'features': ['private_data_collections', 'vpsa_aggregation', 'secure_prediction']
    })


# ===========================
# CLIENT MANAGEMENT
# ===========================


@app.route('/api/client/register', methods=['POST'])
def register_client():
    """Register a new client in the federated learning system"""
    try:
        data = request.json
        client_id = data.get('clientID')
        domain = data.get('domain')
        try:
            dataset_size_int = int(data.get('datasetSize', 0))
        except Exception:
            return jsonify({'error': 'datasetSize must be an integer'}), 400
        dataset_size = str(dataset_size_int)

        if not all([client_id, domain]):
            return jsonify({'error': 'Missing required fields: clientID, domain'}), 400

        if domain not in ['source', 'target']:
            return jsonify({'error': 'Domain must be either "source" or "target"'}), 400

        result = fabric.invoke_chaincode('RegisterClient', [client_id, domain, dataset_size])

        if result['success']:
            return jsonify({
                'message': f'Client {client_id} registered successfully',
                'clientID': client_id,
                'domain': domain
            }), 201
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error registering client: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/client/<client_id>', methods=['GET'])
def get_client(client_id):
    """Get client details by ID"""
    try:
        result = fabric.query_chaincode('GetClient', [client_id])

        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 404

    except Exception as e:
        logger.error(f"Error getting client: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/client/<client_id>/exists', methods=['GET'])
def check_client_exists(client_id):
    """Check if client exists"""
    try:
        result = fabric.query_chaincode('ClientExists', [client_id])

        if result['success']:
            exists = result['data']
            if isinstance(exists, str):
                exists = exists.lower() == 'true'
            return jsonify({'exists': exists, 'clientID': client_id}), 200
        else:
            return jsonify({'exists': False, 'clientID': client_id}), 200

    except Exception as e:
        logger.error(f"Error checking client: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clients', methods=['GET'])
def get_all_clients():
    """Get all registered clients (Note: Not implemented in chaincode, returns empty list)"""
    try:
        return jsonify({
            'message': 'Client listing not available in current chaincode version',
            'clients': []
        }), 200

    except Exception as e:
        logger.error(f"Error getting clients: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===========================
# MODEL SUBMISSION (WITH PRIVATE DATA)
# ===========================


@app.route('/api/model/submit', methods=['POST'])
def submit_local_model():
    """Submit local model with private data partitions"""
    try:
        data = request.json
        model_id = data.get('modelID')
        client_id = data.get('clientID')
        domain = data.get('domain')

        # Model partitions for private collections
        parts = data.get('parts', {})
        org1_part = parts.get('collectionOrg1Private', [])
        org2_part = parts.get('collectionOrg2Private', [])

        # Metadata (optional)
        meta = data.get('meta', {})

        if not all([model_id, client_id, domain]):
            return jsonify({'error': 'Missing required fields: modelID, clientID, domain'}), 400

        # Prepare transient data with model partitions
        transient_data = {
            "modelID": model_id,
            "clientID": client_id,
            "domain": domain,
            "parts": {
                "collectionOrg1Private": org1_part,
                "collectionOrg2Private": org2_part
            },
            "meta": meta
        }

        # Base64 encode the transient data (peer CLI expects base64 values)
        transient_encoded = base64.b64encode(json.dumps(transient_data).encode()).decode()

        result = fabric.invoke_chaincode(
            'SubmitLocalModelTransient',
            [],
            transient={"localModelParts": transient_encoded}
        )

        if result['success']:
            return jsonify({
                'message': 'Model submitted successfully with private data',
                'modelID': model_id,
                'clientID': client_id,
                'partitions': ['collectionOrg1Private', 'collectionOrg2Private']
            }), 201
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error submitting model: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===========================
# VPSA AGGREGATION
# ===========================


@app.route('/api/aggregate/vpsa', methods=['POST'])
def aggregate_models_vpsa():
    """Aggregate models using VPSA algorithm with coordinate-wise trimming"""
    try:
        data = request.json
        model_client_pairs = data.get('modelClientPairs', [])  # Format: ["modelID::clientID", ...]
        beta = str(data.get('beta', 1))  # Trimming parameter (0 or 1)

        if not model_client_pairs:
            return jsonify({'error': 'No model-client pairs provided'}), 400

        # Convert list to JSON string for chaincode
        pairs_json = json.dumps(model_client_pairs)

        result = fabric.invoke_chaincode('AggregateModelsVPSA', [pairs_json, beta])

        if result['success']:
            return jsonify({
                'message': 'VPSA aggregation completed successfully',
                'num_models': len(model_client_pairs),
                'beta': beta,
                'trimming': 'enabled' if beta == '1' else 'disabled'
            }), 200
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error aggregating models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/aggregate', methods=['POST'])
def aggregate_models():
    """Legacy endpoint - redirects to VPSA aggregation"""
    return aggregate_models_vpsa()


# ===========================
# GLOBAL MODEL
# ===========================


@app.route('/api/global-model', methods=['GET'])
def get_global_model():
    """Get current global model"""
    try:
        result = fabric.query_chaincode('GetGlobalModel', [])

        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================
# CONFIGURATION
# ===========================


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get aggregation configuration"""
    try:
        result = fabric.query_chaincode('GetAggregationConfig', [])

        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================
# SECURE PREDICTION (NEW)
# ===========================


@app.route('/api/prediction/setup', methods=['POST'])
def setup_secure_prediction():
    """Setup secure prediction by splitting query vector into shares"""
    try:
        data = request.json
        query_id = data.get('queryID')
        query_vector = data.get('queryVector', [])

        if not query_id or not query_vector:
            return jsonify({'error': 'Missing required fields: queryID, queryVector'}), 400

        # Base64 encode the query vector
        query_vec_encoded = base64.b64encode(json.dumps(query_vector).encode()).decode()

        result = fabric.invoke_chaincode(
            'SecurePredictSetup',
            [query_id],
            transient={"queryVec": query_vec_encoded}
        )

        if result['success']:
            return jsonify({
                'message': 'Query vector split into additive shares',
                'queryID': query_id,
                'collections': ['collectionOrg1Private', 'collectionOrg2Private'],
                'dimensions': len(query_vector)
            }), 201
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error setting up prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/compute/<query_id>', methods=['POST'])
def compute_partial_prediction(query_id):
    """Compute partial prediction for a query (called by each org)"""
    try:
        # Base64 encode query ID
        query_id_encoded = base64.b64encode(query_id.encode()).decode()

        result = fabric.invoke_chaincode(
            'ComputePartialPrediction',
            [],
            transient={"queryID": query_id_encoded}
        )

        if result['success']:
            return jsonify({
                'message': 'Partial prediction computed',
                'queryID': query_id
            }), 200
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error computing partial prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/reconstruct/<query_id>', methods=['POST'])
def reconstruct_prediction(query_id):
    """Reconstruct final prediction from partial predictions"""
    try:
        # Base64 encode query ID
        query_id_encoded = base64.b64encode(query_id.encode()).decode()

        result = fabric.invoke_chaincode(
            'ReconstructPrediction',
            [],
            transient={"queryID": query_id_encoded}
        )

        if result['success']:
            return jsonify({
                'message': 'Prediction reconstructed successfully',
                'queryID': query_id
            }), 200
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error reconstructing prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/<query_id>', methods=['GET'])
def get_prediction(query_id):
    """Get final prediction result"""
    try:
        result = fabric.query_chaincode('GetPrediction', [query_id])

        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 404

    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/complete', methods=['POST'])
def complete_secure_prediction():
    """Complete secure prediction workflow (setup -> compute -> reconstruct)"""
    try:
        data = request.json
        query_id = data.get('queryID')
        query_vector = data.get('queryVector', [])

        if not query_id or not query_vector:
            return jsonify({'error': 'Missing required fields: queryID, queryVector'}), 400

        # Validate query vector dimension against global model
        gm_result = fabric.query_chaincode('GetGlobalModel', [])
        if gm_result['success']:
            gm_data = gm_result['data']
            if isinstance(gm_data, str):
                try:
                    gm_data = json.loads(gm_data)
                except Exception:
                    pass

            # Parse model weights to check dimension
            weights_str = gm_data.get('weights', '[]')
            try:
                weights = json.loads(weights_str) if isinstance(weights_str, str) else weights_str
                model_dim = len(weights)
                query_dim = len(query_vector)

                if query_dim != model_dim:
                    return jsonify({
                        'error': f'Dimension mismatch: query vector has {query_dim} elements, but model requires {model_dim} elements',
                        'required_dimension': model_dim,
                        'provided_dimension': query_dim
                    }), 400
            except Exception as e:
                logger.warning(f"Could not validate dimensions: {str(e)}")

        # Step 1: Setup (split query) - Use both peers for endorsement
        query_vec_encoded = base64.b64encode(json.dumps(query_vector).encode()).decode()
        setup_result = fabric.invoke_chaincode(
            'SecurePredictSetup',
            [query_id],
            transient={"queryVec": query_vec_encoded}
        )

        if not setup_result['success']:
            return jsonify({'error': f"Setup failed: {setup_result['error']}"}), 500

        logger.info(f"Query vector split successfully for {query_id}")

        # Step 2a: Compute partial prediction for Org1
        query_id_encoded = base64.b64encode(query_id.encode()).decode()

        logger.info("Computing partial prediction for Org1...")
        compute_org1_result = fabric.invoke_chaincode_as_org(
            'ComputePartialPrediction',
            [],
            org='org1',
            transient={"queryID": query_id_encoded},
            use_both_peers=True
        )

        if not compute_org1_result['success']:
            return jsonify({'error': f"Org1 compute failed: {compute_org1_result['error']}"}), 500

        logger.info("Org1 partial prediction computed successfully")

        # Step 2b: Compute partial prediction for Org2
        logger.info("Computing partial prediction for Org2...")
        compute_org2_result = fabric.invoke_chaincode_as_org(
            'ComputePartialPrediction',
            [],
            org='org2',
            transient={"queryID": query_id_encoded},
            use_both_peers=True
        )

        if not compute_org2_result['success']:
            return jsonify({'error': f"Org2 compute failed: {compute_org2_result['error']}"}), 500

        logger.info("Org2 partial prediction computed successfully")

        # Step 3: Reconstruct final prediction - Use both peers
        logger.info("Reconstructing final prediction...")
        reconstruct_result = fabric.invoke_chaincode(
            'ReconstructPrediction',
            [],
            transient={"queryID": query_id_encoded}
        )

        if not reconstruct_result['success']:
            return jsonify({'error': f"Reconstruction failed: {reconstruct_result['error']}"}), 500

        logger.info("Prediction reconstructed successfully")

        # Step 4: Get final result
        result = fabric.query_chaincode('GetPrediction', [query_id])

        if result['success']:
            prediction_data = result['data']
            if isinstance(prediction_data, str):
                try:
                    prediction_data = json.loads(prediction_data)
                except Exception:
                    pass

            return jsonify({
                'message': 'Secure prediction completed successfully',
                'queryID': query_id,
                'prediction': prediction_data,
                'workflow': ['setup', 'compute_org1', 'compute_org2', 'reconstruct', 'retrieve']
            }), 200
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        logger.error(f"Error in complete prediction workflow: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/dimensions', methods=['GET'])
def get_model_dimensions():
    """Get the dimensions of the current global model"""
    try:
        result = fabric.query_chaincode('GetGlobalModel', [])

        if result['success']:
            gm_data = result['data']
            if isinstance(gm_data, str):
                try:
                    gm_data = json.loads(gm_data)
                except Exception:
                    pass

            weights_str = gm_data.get('weights', '[]')
            try:
                weights = json.loads(weights_str) if isinstance(weights_str, str) else weights_str
                return jsonify({
                    'dimensions': len(weights),
                    'modelVersion': gm_data.get('version', 0),
                    'round': gm_data.get('round', 0)
                }), 200
            except Exception as e:
                return jsonify({'error': f'Failed to parse model weights: {str(e)}'}), 500
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================
# UTILITY ENDPOINTS
# ===========================


@app.route('/api/round/current', methods=['GET'])
def get_current_round():
    """Get current training round from config"""
    try:
        result = fabric.query_chaincode('GetAggregationConfig', [])

        if result['success']:
            config = result['data']
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except Exception:
                    return jsonify({'error': 'Failed to parse config'}), 500

            current_round = config.get('currentRound', 0)
            return jsonify({
                'currentRound': current_round,
                'beta': config.get('beta', 1),
                'collections': config.get('collections', [])
            }), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        # Get global model
        gm_result = fabric.query_chaincode('GetGlobalModel', [])
        global_model = {}
        if gm_result['success']:
            gm_data = gm_result['data']
            if isinstance(gm_data, str):
                try:
                    global_model = json.loads(gm_data)
                except Exception:
                    pass

        # Get config
        cfg_result = fabric.query_chaincode('GetAggregationConfig', [])
        config = {}
        if cfg_result['success']:
            cfg_data = cfg_result['data']
            if isinstance(cfg_data, str):
                try:
                    config = json.loads(cfg_data)
                except Exception:
                    pass

        return jsonify({
            'globalModel': {
                'version': global_model.get('version', 0),
                'round': global_model.get('round', 0),
                'hasWeights': bool(global_model.get('weights'))
            },
            'config': {
                'currentRound': config.get('currentRound', 0),
                'beta': config.get('beta', 1),
                'collections': config.get('collections', [])
            },
            'features': {
                'privateDataCollections': True,
                'vpsaAggregation': True,
                'securePrediction': True,
                'coordinateTrimming': True
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
