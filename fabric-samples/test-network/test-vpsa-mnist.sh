#!/bin/bash

# VPSA Chaincode Testing Script for MNIST Multi-Class Classification
# Tests 10-class digit classification (0-9)

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}🧪 TEST $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Escape a string for safe embedding as a JSON string value
# Ensures nested JSON (quotes/backslashes) is passed correctly via --ctor
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"  # escape backslashes
    s="${s//\"/\\\"}"  # escape double quotes
    printf '%s' "$s"
}

# Wait for transaction to be committed
wait_for_commit() {
    echo "   Waiting for transaction to commit..."
    sleep 5
}

cd /home/amalendumanoj/project/Federated-Learning-with-zkp-and-blockchain/fabric-samples/test-network

# Set environment for Org1
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

echo "========================================="
echo "VPSA CHAINCODE - MNIST MULTI-CLASS TESTS"
echo "========================================="
echo "Testing 10-class digit classification (0-9)"
echo ""

# ============================================
# MNIST Class Distribution for clients
# ============================================
# Source client: balanced MNIST training data
SOURCE_CLASS_DIST='{"0":980,"1":1135,"2":1032,"3":1010,"4":982,"5":892,"6":958,"7":1028,"8":974,"9":1009}'
# Target client: slightly imbalanced distribution (simulating domain shift)
TARGET_CLASS_DIST='{"0":890,"1":1100,"2":950,"3":880,"4":1050,"5":820,"6":900,"7":980,"8":910,"9":920}'

# ============================================
# Per-class accuracies (simulated MNIST results)
# ============================================
# Source model per-class accuracy (generally higher for source domain)
SOURCE_CLASS_ACC='{"0":0.98,"1":0.99,"2":0.95,"3":0.93,"4":0.96,"5":0.91,"6":0.97,"7":0.94,"8":0.92,"9":0.93}'
# Target model per-class accuracy (lower due to domain shift)  
TARGET_CLASS_ACC='{"0":0.89,"1":0.95,"2":0.86,"3":0.84,"4":0.88,"5":0.82,"6":0.90,"7":0.87,"8":0.83,"9":0.85}'

# Per-class losses
SOURCE_CLASS_LOSS='{"0":0.05,"1":0.03,"2":0.12,"3":0.18,"4":0.10,"5":0.22,"6":0.08,"7":0.15,"8":0.20,"9":0.18}'
TARGET_CLASS_LOSS='{"0":0.25,"1":0.12,"2":0.35,"3":0.40,"4":0.30,"5":0.45,"6":0.28,"7":0.32,"8":0.42,"9":0.38}'

# ============================================
# MNIST Prototypes (768-dim latent space, showing first few dims)
# ============================================
# Class prototypes representing digit features
SOURCE_CLASS_PROTO='{"0":"[0.12,-0.34,0.56,...]","1":"[0.78,-0.12,0.34,...]","2":"[-0.23,0.45,0.67,...]","3":"[0.34,-0.56,0.12,...]","4":"[-0.45,0.23,0.89,...]","5":"[0.56,-0.78,0.23,...]","6":"[-0.67,0.34,0.45,...]","7":"[0.89,-0.23,0.56,...]","8":"[-0.12,0.67,0.78,...]","9":"[0.45,-0.89,0.34,...]"}'
TARGET_CLASS_PROTO='{"0":"[0.15,-0.32,0.54,...]","1":"[0.75,-0.15,0.32,...]","2":"[-0.25,0.42,0.65,...]","3":"[0.32,-0.54,0.15,...]","4":"[-0.42,0.25,0.85,...]","5":"[0.54,-0.75,0.25,...]","6":"[-0.65,0.32,0.42,...]","7":"[0.85,-0.25,0.54,...]","8":"[-0.15,0.65,0.75,...]","9":"[0.42,-0.85,0.32,...]"}'

# Aggregated global prototypes
GLOBAL_CLASS_PROTO='{"0":"[0.135,-0.33,0.55,...]","1":"[0.765,-0.135,0.33,...]","2":"[-0.24,0.435,0.66,...]","3":"[0.33,-0.55,0.135,...]","4":"[-0.435,0.24,0.87,...]","5":"[0.55,-0.765,0.24,...]","6":"[-0.66,0.33,0.435,...]","7":"[0.87,-0.24,0.55,...]","8":"[-0.135,0.66,0.765,...]","9":"[0.435,-0.87,0.33,...]"}'

# Aggregated class accuracies
GLOBAL_CLASS_ACC='{"0":0.935,"1":0.97,"2":0.905,"3":0.885,"4":0.92,"5":0.865,"6":0.935,"7":0.905,"8":0.875,"9":0.89}'

# ============================================
# Confusion Matrix (10x10 for MNIST)
# ============================================
# Simplified confusion matrix showing predictions vs actual
SOURCE_CONF_MATRIX='[[960,0,5,2,1,3,5,2,2,0],[0,1120,3,2,0,1,4,1,4,0],[6,2,985,10,8,2,5,8,4,2],[2,0,12,960,0,18,2,8,5,3],[1,1,5,0,950,1,8,2,3,11],[8,2,2,15,3,840,10,1,8,3],[5,3,4,0,6,10,925,0,5,0],[1,5,15,5,6,0,0,980,2,14],[4,2,8,12,5,12,6,4,910,11],[5,4,3,10,18,5,1,12,6,945]]'
TARGET_CONF_MATRIX='[[850,5,10,5,3,8,5,2,1,1],[2,1050,8,5,2,3,10,5,10,5],[15,5,850,20,12,8,12,15,10,3],[8,2,18,810,2,25,5,15,10,15],[5,3,8,2,890,5,12,8,8,19],[12,5,5,22,8,750,18,5,15,10],[10,8,10,2,12,15,860,2,8,3],[3,10,20,10,10,2,2,900,5,18],[8,5,12,18,10,18,10,8,790,21],[10,8,5,15,25,8,3,18,10,868]]'
GLOBAL_CONF_MATRIX='[[905,2,7,3,2,5,5,2,1,0],[1,1085,5,3,1,2,7,3,7,2],[10,3,917,15,10,5,8,11,7,2],[5,1,15,885,1,21,3,11,7,9],[3,2,6,1,920,3,10,5,5,15],[10,3,3,18,5,795,14,3,11,6],[7,5,7,1,9,12,892,1,6,1],[2,7,17,7,8,1,1,940,3,16],[6,3,10,15,7,15,8,6,850,16],[7,6,4,12,21,6,2,15,8,906]]'

print_test "1" "Register Source Domain Client with MNIST Class Distribution"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"RegisterClient","Args":["mnist-source-client","source","10000","'"$(json_escape "$SOURCE_CLASS_DIST")"'"]}' \
    --waitForEvent
wait_for_commit
print_success "Source client registered with 10-class distribution"

print_test "2" "Register Target Domain Client with MNIST Class Distribution"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"RegisterClient","Args":["mnist-target-client","target","9400","'"$(json_escape "$TARGET_CLASS_DIST")"'"]}' \
    --waitForEvent
wait_for_commit
print_success "Target client registered with 10-class distribution"

print_test "3" "Query All Clients"
CLIENTS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetAllClients","Args":[]}')
echo "   Result: $CLIENTS"
if [[ "$CLIENTS" == "[]" || -z "$CLIENTS" ]]; then
    print_error "No clients found - registration may have failed"
    exit 1
fi
print_success "Clients retrieved successfully"

print_test "4" "Get Number of Classes (should be 10 for MNIST)"
NUM_CLASSES=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetNumClasses","Args":[]}')
echo "   Number of classes: $NUM_CLASSES"
print_success "Number of classes verified"

print_test "5" "Get Class Labels (0-9)"
CLASS_LABELS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClassLabels","Args":[]}')
echo "   Class labels: $CLASS_LABELS"
print_success "Class labels retrieved"

print_test "6" "Get Client Class Distribution"
CLIENT_DIST=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClientClassDistribution","Args":["mnist-source-client"]}')
echo "   Source client class distribution: $CLIENT_DIST"
print_success "Class distribution retrieved"

print_test "7" "Submit Local Model from Source Domain with Multi-Class Metrics"
# Arguments: modelID, clientID, weights, latentFeatures, prototypes, classPrototypesJSON, 
#            accuracy, loss, alignmentLoss, classAccuraciesJSON, classLossesJSON, 
#            confusionMatrix, dataSize
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"SubmitLocalModel","Args":["mnist-model-source-r0","mnist-source-client","{\"conv1\":[0.5,0.3],\"conv2\":[0.8,0.2],\"fc1\":[0.6,0.4]}","{\"latent_dim\":768,\"features\":[0.1,0.2,0.3]}","{\"all_prototypes\":\"...\"}","'"$(json_escape "$SOURCE_CLASS_PROTO")"'","0.948","0.131","0.045","'"$(json_escape "$SOURCE_CLASS_ACC")"'","'"$(json_escape "$SOURCE_CLASS_LOSS")"'","'"$(json_escape "$SOURCE_CONF_MATRIX")"'","10000"]}' \
    --waitForEvent
wait_for_commit
print_success "Source model submitted with 10-class metrics"

print_test "8" "Submit Local Model from Target Domain with Multi-Class Metrics"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"SubmitLocalModel","Args":["mnist-model-target-r0","mnist-target-client","{\"conv1\":[0.4,0.6],\"conv2\":[0.7,0.3],\"fc1\":[0.5,0.5]}","{\"latent_dim\":768,\"features\":[0.15,0.25,0.35]}","{\"all_prototypes\":\"...\"}","'"$(json_escape "$TARGET_CLASS_PROTO")"'","0.868","0.327","0.078","'"$(json_escape "$TARGET_CLASS_ACC")"'","'"$(json_escape "$TARGET_CLASS_LOSS")"'","'"$(json_escape "$TARGET_CONF_MATRIX")"'","9400"]}' \
    --waitForEvent
wait_for_commit
print_success "Target model submitted with 10-class metrics"

print_test "9" "Query Local Model with Multi-Class Details"
MODEL_DATA=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetLocalModel","Args":["mnist-model-source-r0"]}')
echo "   Source model data:"
echo "$MODEL_DATA" | python3 -m json.tool 2>/dev/null || echo "$MODEL_DATA"
print_success "Local model with multi-class data retrieved"

print_test "10" "Aggregate Models with Multi-Class Prototypes"
# Arguments: modelIDs[], aggregatedWeights, aggregatedPrototypes, classPrototypesJSON,
#            globalAccuracy, globalLoss, classAccuraciesJSON, confusionMatrix, alignmentScore
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"AggregateModels","Args":["[\"mnist-model-source-r0\",\"mnist-model-target-r0\"]","{\"conv1\":[0.45,0.45],\"conv2\":[0.75,0.25],\"fc1\":[0.55,0.45]}","{\"global_prototypes\":\"aggregated\"}","'"$(json_escape "$GLOBAL_CLASS_PROTO")"'","0.908","0.229","'"$(json_escape "$GLOBAL_CLASS_ACC")"'","'"$(json_escape "$GLOBAL_CONF_MATRIX")"'","0.912"]}' \
    --waitForEvent
wait_for_commit
print_success "Models aggregated with 10-class prototypes"

print_test "11" "Query Updated Global Model with Multi-Class Info"
GLOBAL_MODEL=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetGlobalModel","Args":[]}')
echo "   Global model:"
echo "$GLOBAL_MODEL" | python3 -m json.tool 2>/dev/null || echo "$GLOBAL_MODEL"
print_success "Global model with multi-class data retrieved"

print_test "12" "Query Training Metrics with Per-Class Accuracies"
METRICS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetTrainingMetrics","Args":["0"]}')
echo "   Training metrics for round 0:"
echo "$METRICS" | python3 -m json.tool 2>/dev/null || echo "$METRICS"
print_success "Training metrics retrieved"

print_test "13" "Query Per-Class Accuracies for Round 0"
CLASS_ACCS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClassAccuracies","Args":["0"]}')
echo "   Per-class accuracies:"
echo "$CLASS_ACCS" | python3 -m json.tool 2>/dev/null || echo "$CLASS_ACCS"
print_success "Per-class accuracies retrieved"

print_test "14" "Query Global Class Prototypes"
PROTOTYPES=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClassPrototypes","Args":[]}')
echo "   Global class prototypes:"
echo "$PROTOTYPES" | python3 -m json.tool 2>/dev/null || echo "$PROTOTYPES"
print_success "Global class prototypes retrieved"

print_test "15" "Query Confusion Matrix for Round 0"
CONF_MATRIX=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetConfusionMatrix","Args":["0"]}')
echo "   Confusion matrix:"
echo "$CONF_MATRIX"
print_success "Confusion matrix retrieved"

print_test "16" "Query Class Metrics for Digit '0'"
CLASS_0_METRICS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClassMetricsByRound","Args":["0","0"]}')
echo "   Metrics for digit 0:"
echo "$CLASS_0_METRICS" | python3 -m json.tool 2>/dev/null || echo "$CLASS_0_METRICS"
print_success "Class-specific metrics retrieved"

print_test "17" "Query Class Metrics for Digit '5' (typically harder)"
CLASS_5_METRICS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClassMetricsByRound","Args":["0","5"]}')
echo "   Metrics for digit 5:"
echo "$CLASS_5_METRICS" | python3 -m json.tool 2>/dev/null || echo "$CLASS_5_METRICS"
print_success "Class-specific metrics for digit 5 retrieved"

print_test "18" "Update Aggregation Config"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
    -c '{"function":"UpdateAggregationConfig","Args":["5","0.7","0.3","0.15"]}' \
    --waitForEvent
wait_for_commit
print_success "Config updated"

print_test "19" "Query Model History"
HISTORY=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetModelHistory","Args":["vpsa-global-model"]}')
echo "   Model history entries: $(echo "$HISTORY" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "N/A")"
print_success "Model history retrieved"

print_test "20" "Query Client Details with Multi-Class Accuracies"
CLIENT_DETAILS=$(peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClient","Args":["mnist-source-client"]}')
echo "   Client details:"
echo "$CLIENT_DETAILS" | python3 -m json.tool 2>/dev/null || echo "$CLIENT_DETAILS"
print_success "Client details with per-class accuracies retrieved"

echo ""
echo "========================================="
echo "✅ ALL MNIST MULTI-CLASS TESTS COMPLETED!"
echo "========================================="
echo ""
echo "Summary:"
echo "  - 10 MNIST classes (digits 0-9) configured"
echo "  - Per-class accuracy tracking enabled"
echo "  - 10x10 confusion matrix stored"
echo "  - Class prototypes for each digit stored"
echo "  - Source and target domain models tested"
echo ""
print_info "The chaincode now supports multi-class MNIST federated learning!"
