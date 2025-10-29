from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
import logging
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fabric network configuration
FABRIC_CONFIG = {
    'channel': 'vpsa-channel',
    'chaincode': 'vpsa',
    'network_path': '/home/amalendu/college/federatedLearning/fabric-samples/test-network'
}

class FabricGateway:
    """Interface to interact with Hyperledger Fabric network"""
    
    def __init__(self):
        self.network_path = FABRIC_CONFIG['network_path']
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment variables for Fabric CLI"""
        os.environ['CORE_PEER_TLS_ENABLED'] = 'true'
        os.environ['CORE_PEER_LOCALMSPID'] = 'Org1MSP'
        os.environ['CORE_PEER_TLS_ROOTCERT_FILE'] = f"{self.network_path}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
        os.environ['CORE_PEER_MSPCONFIGPATH'] = f"{self.network_path}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
        os.environ['CORE_PEER_ADDRESS'] = 'localhost:7051'
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
                except:
                    return line.strip()
        
        return output
    
    def invoke_chaincode(self, function, args):
        """Invoke chaincode function with endorsements from both orgs"""
        args_json = json.dumps({"function": function, "Args": args})
        
        # Build command with BOTH peer addresses for endorsement
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
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.network_path, timeout=30)
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
    
    def query_chaincode(self, function, args):
        """Query chaincode function"""
        args_json = json.dumps({"function": function, "Args": args})
        
        cmd = [
            'peer', 'chaincode', 'query',
            '-C', FABRIC_CONFIG['channel'],
            '-n', FABRIC_CONFIG['chaincode'],
            '-c', args_json
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.network_path, timeout=15)
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

class VPSAAggregator:
    """VPSA Aggregation Logic"""
    
    @staticmethod
    def aggregate_weights(local_models, source_weight=0.6, target_weight=0.4):
        source_weights = []
        target_weights = []
        
        for model in local_models:
            weights = json.loads(model.get('weights', '{}'))
            if model.get('domain') == 'source':
                source_weights.append(weights)
            else:
                target_weights.append(weights)
        
        aggregated = {}
        
        if source_weights:
            for key in source_weights[0].keys():
                values = [w[key] for w in source_weights if key in w]
                if isinstance(values[0], list):
                    aggregated[key] = [sum(x)/len(x) * source_weight for x in zip(*values)]
                else:
                    aggregated[key] = sum(values) / len(values) * source_weight
        
        if target_weights:
            for key in target_weights[0].keys():
                values = [w[key] for w in target_weights if key in w]
                if isinstance(values[0], list):
                    if key in aggregated:
                        aggregated[key] = [a + (sum(x)/len(x) * target_weight) 
                                          for a, x in zip(aggregated[key], zip(*values))]
                    else:
                        aggregated[key] = [sum(x)/len(x) * target_weight for x in zip(*values)]
                else:
                    if key in aggregated:
                        aggregated[key] += sum(values) / len(values) * target_weight
                    else:
                        aggregated[key] = sum(values) / len(values) * target_weight
        
        return json.dumps(aggregated)
    
    @staticmethod
    def aggregate_prototypes(local_models, alignment_weight=0.1):
        all_prototypes = []
        
        for model in local_models:
            prototypes = json.loads(model.get('prototypes', '{}'))
            all_prototypes.append(prototypes)
        
        aggregated = {}
        if all_prototypes:
            for key in all_prototypes[0].keys():
                values = [p[key] for p in all_prototypes if key in p]
                if isinstance(values[0], list):
                    aggregated[key] = [sum(x)/len(x) for x in zip(*values)]
                else:
                    aggregated[key] = sum(values) / len(values)
        
        return json.dumps(aggregated)
    
    @staticmethod
    def compute_metrics(local_models):
        accuracies = [m.get('accuracy', 0) for m in local_models]
        losses = [m.get('loss', 0) for m in local_models]
        alignment_losses = [m.get('alignmentLoss', 0) for m in local_models]
        
        return {
            'global_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'global_loss': sum(losses) / len(losses) if losses else 0,
            'alignment_score': 1.0 - (sum(alignment_losses) / len(alignment_losses)) if alignment_losses else 0
        }

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'network': 'Hyperledger Fabric',
        'chaincode': FABRIC_CONFIG['chaincode']
    })

@app.route('/api/client/register', methods=['POST'])
def register_client():
    try:
        data = request.json
        client_id = data.get('clientID')
        domain = data.get('domain')
        dataset_size = str(data.get('datasetSize', 0))
        
        if not all([client_id, domain]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        result = fabric.invoke_chaincode('RegisterClient', [client_id, domain, dataset_size])
        
        if result['success']:
            return jsonify({
                'message': f'Client {client_id} registered successfully',
                'clientID': client_id
            }), 201
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error registering client: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/client/<client_id>', methods=['GET'])
def get_client(client_id):
    try:
        result = fabric.query_chaincode('GetClient', [client_id])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 404
            
    except Exception as e:
        logger.error(f"Error getting client: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clients', methods=['GET'])
def get_all_clients():
    try:
        result = fabric.query_chaincode('GetAllClients', [])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                data = data.strip()
                if not data or data == '\n':
                    return jsonify([]), 200
                try:
                    data = json.loads(data)
                except:
                    return jsonify([]), 200
            
            if not isinstance(data, list):
                data = [data] if data else []
            
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error getting clients: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/submit', methods=['POST'])
def submit_local_model():
    try:
        data = request.json
        weights = json.dumps(data.get('weights', {}))
        latent_features = json.dumps(data.get('latentFeatures', {}))
        prototypes = json.dumps(data.get('prototypes', {}))
        
        args = [
            data.get('modelID'),
            data.get('clientID'),
            weights,
            latent_features,
            prototypes,
            str(data.get('accuracy', 0)),
            str(data.get('loss', 0)),
            str(data.get('alignmentLoss', 0)),
            str(data.get('dataSize', 0))
        ]
        
        result = fabric.invoke_chaincode('SubmitLocalModel', args)
        
        if result['success']:
            return jsonify({
                'message': 'Model submitted successfully',
                'modelID': data.get('modelID')
            }), 201
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error submitting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/<model_id>', methods=['GET'])
def get_local_model(model_id):
    try:
        result = fabric.query_chaincode('GetLocalModel', [model_id])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 404
            
    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/round/<int:round_num>', methods=['GET'])
def get_models_by_round(round_num):
    try:
        result = fabric.query_chaincode('GetLocalModelsByRound', [str(round_num)])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    data = []
            if not isinstance(data, list):
                data = [data] if data else []
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aggregate', methods=['POST'])
def aggregate_models():
    try:
        data = request.json
        model_ids = data.get('modelIDs', [])
        source_weight = data.get('sourceWeight', 0.6)
        target_weight = data.get('targetWeight', 0.4)
        alignment_weight = data.get('alignmentWeight', 0.1)
        
        if not model_ids:
            return jsonify({'error': 'No models provided'}), 400
        
        local_models = []
        for model_id in model_ids:
            result = fabric.query_chaincode('GetLocalModel', [model_id])
            if result['success']:
                model_data = result['data']
                if isinstance(model_data, str):
                    try:
                        model_data = json.loads(model_data)
                    except:
                        continue
                local_models.append(model_data)
        
        if not local_models:
            return jsonify({'error': 'No valid models found'}), 404
        
        aggregator = VPSAAggregator()
        aggregated_weights = aggregator.aggregate_weights(local_models, source_weight, target_weight)
        aggregated_prototypes = aggregator.aggregate_prototypes(local_models, alignment_weight)
        metrics = aggregator.compute_metrics(local_models)
        
        args = [
            json.dumps(model_ids),
            aggregated_weights,
            aggregated_prototypes,
            str(metrics['global_accuracy']),
            str(metrics['global_loss']),
            str(metrics['alignment_score'])
        ]
        
        result = fabric.invoke_chaincode('AggregateModels', args)
        
        if result['success']:
            return jsonify({
                'message': 'Models aggregated successfully',
                'metrics': metrics,
                'num_models': len(model_ids)
            }), 200
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error aggregating models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/global-model', methods=['GET'])
def get_global_model():
    try:
        result = fabric.query_chaincode('GetGlobalModel', [])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        result = fabric.query_chaincode('GetAggregationConfig', [])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/update', methods=['PUT'])
def update_config():
    try:
        data = request.json
        
        args = [
            str(data.get('minClients', 3)),
            str(data.get('sourceWeight', 0.6)),
            str(data.get('targetWeight', 0.4)),
            str(data.get('alignmentWeight', 0.1))
        ]
        
        result = fabric.invoke_chaincode('UpdateAggregationConfig', args)
        
        if result['success']:
            return jsonify({'message': 'Configuration updated successfully'}), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/<int:round_num>', methods=['GET'])
def get_round_metrics(round_num):
    try:
        result = fabric.query_chaincode('GetTrainingMetrics', [str(round_num)])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_all_metrics():
    try:
        result = fabric.query_chaincode('GetAllTrainingMetrics', [])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    data = []
            if not isinstance(data, list):
                data = [data] if data else []
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/global-model/history', methods=['GET'])
def get_global_model_history():
    try:
        result = fabric.query_chaincode('GetModelHistory', ['vpsa-global-model'])
        
        if result['success']:
            data = result['data']
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    data = []
            return jsonify(data), 200
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)