#!/bin/bash

# VPSA Chaincode Deployment Script with Private Data Collections
# This script deploys the VPSA chaincode with PDC support

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_section() { echo -e "\n${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }

# Configuration
CHANNEL_NAME="vpsa-channel"
CC_NAME="vpsa"
CC_VERSION="1.0"
CC_SEQUENCE="1"
CC_PATH="../federated-learning/chaincode"
COLLECTIONS_CONFIG="${CC_PATH}/collections_config.json"

cd /home/amalendumanoj/project/Federated-Learning-with-zkp-and-blockchain/fabric-samples/test-network

print_section "STEP 1: AGGRESSIVE CLEANUP"
print_info "Stopping all Fabric containers..."

# Function to run docker commands (with sudo fallback)
docker_cmd() {
    if docker "$@" 2>/dev/null; then
        return 0
    else
        sudo docker "$@" 2>/dev/null || true
    fi
}

# Stop and remove all Fabric-related containers
print_info "Stopping Fabric containers..."
docker_cmd stop $(docker ps -aq --filter name='peer0' 2>/dev/null) 2>/dev/null || true
docker_cmd stop $(docker ps -aq --filter name='orderer' 2>/dev/null) 2>/dev/null || true
docker_cmd stop $(docker ps -aq --filter name='ca_' 2>/dev/null) 2>/dev/null || true
docker_cmd stop $(docker ps -aq --filter name='dev-peer' 2>/dev/null) 2>/dev/null || true

print_info "Removing Fabric containers..."
docker_cmd rm -f $(docker ps -aq --filter name='peer0' 2>/dev/null) 2>/dev/null || true
docker_cmd rm -f $(docker ps -aq --filter name='orderer' 2>/dev/null) 2>/dev/null || true
docker_cmd rm -f $(docker ps -aq --filter name='ca_' 2>/dev/null) 2>/dev/null || true
docker_cmd rm -f $(docker ps -aq --filter name='dev-peer' 2>/dev/null) 2>/dev/null || true

# Remove chaincode containers
print_info "Removing chaincode containers..."
docker_cmd rm -f $(docker ps -aq --filter name='vpsa' 2>/dev/null) 2>/dev/null || true

# Remove volumes
print_info "Removing Docker volumes..."
docker_cmd volume rm docker_orderer.example.com 2>/dev/null || true
docker_cmd volume rm docker_peer0.org1.example.com 2>/dev/null || true
docker_cmd volume rm docker_peer0.org2.example.com 2>/dev/null || true
docker volume prune -f 2>/dev/null || sudo docker volume prune -f 2>/dev/null || true

# Remove chaincode images
print_info "Removing chaincode images..."
docker rmi -f $(docker images -q 'dev-peer*') 2>/dev/null || true
docker rmi -f $(docker images -q '*vpsa*') 2>/dev/null || true

# Clean up network artifacts
print_info "Cleaning up network artifacts..."
rm -rf organizations/peerOrganizations 2>/dev/null || sudo rm -rf organizations/peerOrganizations 2>/dev/null || true
rm -rf organizations/ordererOrganizations 2>/dev/null || sudo rm -rf organizations/ordererOrganizations 2>/dev/null || true
rm -rf organizations/fabric-ca/org1/msp organizations/fabric-ca/org1/*.pem 2>/dev/null || true
rm -rf organizations/fabric-ca/org2/msp organizations/fabric-ca/org2/*.pem 2>/dev/null || true
rm -rf organizations/fabric-ca/ordererOrg/msp organizations/fabric-ca/ordererOrg/*.pem 2>/dev/null || true
rm -rf channel-artifacts system-genesis-block 2>/dev/null || true

# Run network.sh down
print_info "Running network.sh down..."
./network.sh down 2>/dev/null || true

# Remove package file
rm -f ${CC_NAME}.tar.gz

# Wait for containers to fully stop
print_info "Waiting for cleanup to complete..."
sleep 3

print_info "Cleanup completed"

print_section "STEP 2: START NETWORK"
print_info "Starting Fabric network with CA..."

# Retry logic for network startup
MAX_RETRIES=3
RETRY_COUNT=0
NETWORK_UP=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$NETWORK_UP" = "false" ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    print_info "Attempt $RETRY_COUNT of $MAX_RETRIES..."
    
    if ./network.sh up createChannel -c ${CHANNEL_NAME} -ca; then
        NETWORK_UP=true
        print_info "Network started successfully!"
    else
        print_warning "Network startup failed, cleaning up and retrying..."
        ./network.sh down 2>/dev/null || true
        sleep 5
    fi
done

if [ "$NETWORK_UP" = "false" ]; then
    print_error "Failed to start network after $MAX_RETRIES attempts"
    exit 1
fi

print_info "Network started with channel: ${CHANNEL_NAME}"

# Wait for network to stabilize
print_info "Waiting for network to stabilize..."
sleep 10

print_section "STEP 3: ENVIRONMENT SETUP"
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/
export CORE_PEER_TLS_ENABLED=true

# Set Docker build timeouts
export CORE_CHAINCODE_BUILDER_TIMEOUT=240s
export DOCKER_BUILD_TIMEOUT=300s

print_info "Environment configured"
print_info "Docker build timeout: 300s"
print_info "Chaincode builder timeout: 240s"

# Verify peer binary
if ! command -v peer &> /dev/null; then
    print_error "peer command not found. Check PATH: ${PATH}"
    exit 1
fi
print_info "Peer binary found: $(which peer)"

print_section "STEP 4: VERIFY COLLECTIONS CONFIG"
if [ ! -f "${COLLECTIONS_CONFIG}" ]; then
    print_warning "Collections config not found at ${COLLECTIONS_CONFIG}"
    print_info "Creating collections_config.json..."
    
    # Ensure directory exists
    mkdir -p "$(dirname ${COLLECTIONS_CONFIG})"
    
    cat > "${COLLECTIONS_CONFIG}" << 'EOF'
[
    {
        "name": "collectionSharedModels",
        "policy": "OR('Org1MSP.member', 'Org2MSP.member')",
        "requiredPeerCount": 1,
        "maxPeerCount": 2,
        "blockToLive": 0,
        "memberOnlyRead": true,
        "memberOnlyWrite": true
    },
    {
        "name": "collectionAggregatedModels",
        "policy": "OR('Org1MSP.member', 'Org2MSP.member')",
        "requiredPeerCount": 1,
        "maxPeerCount": 2,
        "blockToLive": 0,
        "memberOnlyRead": true,
        "memberOnlyWrite": true
    }
]
EOF
fi

print_info "Collections config verified: ${COLLECTIONS_CONFIG}"

# Verify chaincode path exists
if [ ! -d "${CC_PATH}" ]; then
    print_error "Chaincode path not found: ${CC_PATH}"
    exit 1
fi

if [ ! -f "${CC_PATH}/vpsa_chaincode.go" ]; then
    print_error "Chaincode file not found: ${CC_PATH}/vpsa_chaincode.go"
    exit 1
fi
print_info "Chaincode verified at: ${CC_PATH}"

print_section "STEP 5: PACKAGE CHAINCODE"
print_info "Packaging chaincode..."
peer lifecycle chaincode package ${CC_NAME}.tar.gz \
    --path ${CC_PATH} \
    --lang golang \
    --label ${CC_NAME}_${CC_VERSION}

if [ ! -f "${CC_NAME}.tar.gz" ]; then
    print_error "Failed to create chaincode package"
    exit 1
fi
print_info "Chaincode packaged: ${CC_NAME}.tar.gz ($(ls -lh ${CC_NAME}.tar.gz | awk '{print $5}'))"

print_section "STEP 6: INSTALL ON ORG1"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

# Verify TLS cert exists
if [ ! -f "${CORE_PEER_TLS_ROOTCERT_FILE}" ]; then
    print_error "TLS cert not found for Org1: ${CORE_PEER_TLS_ROOTCERT_FILE}"
    exit 1
fi

print_info "Installing on Org1 (peer0.org1.example.com:7051)..."
print_info "Waiting for peer to be ready..."
sleep 5

# Retry logic for install
INSTALL_RETRY=0
MAX_INSTALL_RETRIES=3
INSTALL_SUCCESS=false

while [ $INSTALL_RETRY -lt $MAX_INSTALL_RETRIES ] && [ "$INSTALL_SUCCESS" = "false" ]; do
    INSTALL_RETRY=$((INSTALL_RETRY + 1))
    print_info "Install attempt $INSTALL_RETRY of $MAX_INSTALL_RETRIES..."
    
    if peer lifecycle chaincode install ${CC_NAME}.tar.gz; then
        print_info "Successfully installed on Org1"
        INSTALL_SUCCESS=true
    else
        if [ $INSTALL_RETRY -lt $MAX_INSTALL_RETRIES ]; then
            print_warning "Install failed, waiting 15 seconds before retry..."
            sleep 15
        else
            print_error "Failed to install on Org1 after $MAX_INSTALL_RETRIES attempts"
            exit 1
        fi
    fi
done

print_section "STEP 7: INSTALL ON ORG2"
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

# Verify TLS cert exists
if [ ! -f "${CORE_PEER_TLS_ROOTCERT_FILE}" ]; then
    print_error "TLS cert not found for Org2: ${CORE_PEER_TLS_ROOTCERT_FILE}"
    exit 1
fi

print_info "Installing on Org2 (peer0.org2.example.com:9051)..."
print_info "Waiting for peer to be ready..."
sleep 5

# Retry logic for install
INSTALL_RETRY=0
MAX_INSTALL_RETRIES=3
INSTALL_SUCCESS=false

while [ $INSTALL_RETRY -lt $MAX_INSTALL_RETRIES ] && [ "$INSTALL_SUCCESS" = "false" ]; do
    INSTALL_RETRY=$((INSTALL_RETRY + 1))
    print_info "Install attempt $INSTALL_RETRY of $MAX_INSTALL_RETRIES..."
    
    if peer lifecycle chaincode install ${CC_NAME}.tar.gz; then
        print_info "Successfully installed on Org2"
        INSTALL_SUCCESS=true
    else
        if [ $INSTALL_RETRY -lt $MAX_INSTALL_RETRIES ]; then
            print_warning "Install failed, waiting 15 seconds before retry..."
            sleep 15
        else
            print_error "Failed to install on Org2 after $MAX_INSTALL_RETRIES attempts"
            exit 1
        fi
    fi
done

print_section "STEP 8: GET PACKAGE ID"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

print_info "Querying installed chaincodes..."
peer lifecycle chaincode queryinstalled

PACKAGE_ID=$(peer lifecycle chaincode queryinstalled 2>&1 | grep "${CC_NAME}_${CC_VERSION}" | awk -F "[, ]+" '{print $3}')

if [ -z "$PACKAGE_ID" ]; then
    # Try alternative parsing
    PACKAGE_ID=$(peer lifecycle chaincode queryinstalled --output json 2>&1 | grep -o "${CC_NAME}_${CC_VERSION}:[a-f0-9]*" | head -1)
fi

if [ -z "$PACKAGE_ID" ]; then
    print_error "Failed to get package ID"
    print_info "Installed chaincodes:"
    peer lifecycle chaincode queryinstalled
    exit 1
fi

print_info "Package ID: ${PACKAGE_ID}"

print_section "STEP 9: APPROVE FOR ORG1 (WITH COLLECTIONS)"
print_info "Approving chaincode for Org1..."
peer lifecycle chaincode approveformyorg \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    --channelID ${CHANNEL_NAME} \
    --name ${CC_NAME} \
    --version ${CC_VERSION} \
    --package-id ${PACKAGE_ID} \
    --sequence ${CC_SEQUENCE} \
    --collections-config ${COLLECTIONS_CONFIG}
print_info "Approved for Org1 with collections config"

print_section "STEP 10: APPROVE FOR ORG2 (WITH COLLECTIONS)"
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

print_info "Approving chaincode for Org2..."
peer lifecycle chaincode approveformyorg \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    --channelID ${CHANNEL_NAME} \
    --name ${CC_NAME} \
    --version ${CC_VERSION} \
    --package-id ${PACKAGE_ID} \
    --sequence ${CC_SEQUENCE} \
    --collections-config ${COLLECTIONS_CONFIG}
print_info "Approved for Org2 with collections config"

print_section "STEP 11: CHECK COMMIT READINESS"
print_info "Checking commit readiness..."
peer lifecycle chaincode checkcommitreadiness \
    --channelID ${CHANNEL_NAME} \
    --name ${CC_NAME} \
    --version ${CC_VERSION} \
    --sequence ${CC_SEQUENCE} \
    --collections-config ${COLLECTIONS_CONFIG} \
    --output json

print_section "STEP 12: COMMIT CHAINCODE (WITH COLLECTIONS)"
print_info "Committing chaincode to channel..."
peer lifecycle chaincode commit \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    --channelID ${CHANNEL_NAME} \
    --name ${CC_NAME} \
    --version ${CC_VERSION} \
    --sequence ${CC_SEQUENCE} \
    --collections-config ${COLLECTIONS_CONFIG} \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
print_info "Chaincode committed with Private Data Collections"

print_section "STEP 13: VERIFY DEPLOYMENT"
print_info "Querying committed chaincode..."
peer lifecycle chaincode querycommitted \
    --channelID ${CHANNEL_NAME} \
    --name ${CC_NAME}

# Wait for chaincode container to start
print_info "Waiting for chaincode container to start (30 seconds)..."
sleep 30

print_section "STEP 14: INITIALIZE LEDGER"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

print_info "Initializing ledger..."
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    -C ${CHANNEL_NAME} \
    -n ${CC_NAME} \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -c '{"function":"InitLedger","Args":[]}'

print_info "Waiting for transaction to be committed..."
sleep 5

print_section "STEP 15: TEST QUERIES"
print_info "Testing GetGlobalModel..."
GLOBAL_MODEL=$(peer chaincode query \
    -C ${CHANNEL_NAME} \
    -n ${CC_NAME} \
    -c '{"function":"GetGlobalModel","Args":[]}' 2>&1)

if echo "$GLOBAL_MODEL" | grep -q "modelID"; then
    echo "$GLOBAL_MODEL" | python3 -m json.tool 2>/dev/null || echo "$GLOBAL_MODEL"
    print_info "GetGlobalModel: SUCCESS"
else
    print_warning "GetGlobalModel returned: $GLOBAL_MODEL"
fi

print_info "Testing GetAggregationConfig..."
AGG_CONFIG=$(peer chaincode query \
    -C ${CHANNEL_NAME} \
    -n ${CC_NAME} \
    -c '{"function":"GetAggregationConfig","Args":[]}' 2>&1)

if echo "$AGG_CONFIG" | grep -q "configID"; then
    echo "$AGG_CONFIG" | python3 -m json.tool 2>/dev/null || echo "$AGG_CONFIG"
    print_info "GetAggregationConfig: SUCCESS"
else
    print_warning "GetAggregationConfig returned: $AGG_CONFIG"
fi

print_section "STEP 16: QUICK FUNCTIONAL TEST"
print_info "Registering a test client..."
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
    -C ${CHANNEL_NAME} \
    -n ${CC_NAME} \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
    -c '{"function":"RegisterClient","Args":["test-client-1","source","1000"]}' 2>&1 || true

sleep 3

print_info "Querying all clients..."
CLIENTS=$(peer chaincode query \
    -C ${CHANNEL_NAME} \
    -n ${CC_NAME} \
    -c '{"function":"GetAllClients","Args":[]}' 2>&1)
echo "$CLIENTS" | python3 -m json.tool 2>/dev/null || echo "$CLIENTS"

print_section "DEPLOYMENT COMPLETE"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VPSA Chaincode with PDC deployed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Network Details:"
echo "  Channel: ${CHANNEL_NAME}"
echo "  Chaincode: ${CC_NAME}"
echo "  Version: ${CC_VERSION}"
echo "  Package ID: ${PACKAGE_ID}"
echo ""
echo "Private Data Collections:"
echo "  - collectionSharedModels"
echo "  - collectionAggregatedModels"
echo ""
echo "Available Functions:"
echo "  Invoke:"
echo "    - InitLedger()"
echo "    - RegisterClient(clientID, domain, datasetSize)"
echo "    - SubmitLocalModel(modelID, clientID, weights, latentFeatures, prototypes, accuracy, loss, alignmentLoss, dataSize)"
echo "    - SubmitLocalModelWithTransient(modelID, clientID, round, dataHash)"
echo "    - AggregateModelsWithPrivateData(modelIDsJSON)"
echo "    - UpdateAggregationConfig(minClients, sourceWeight, targetWeight, alignmentWeight)"
echo ""
echo "  Query:"
echo "    - GetGlobalModel()"
echo "    - GetAggregationConfig()"
echo "    - GetClient(clientID)"
echo "    - GetAllClients()"
echo "    - GetLocalModel(modelID)"
echo "    - GetLocalModelsByRound(round)"
echo "    - GetTrainingMetrics(round)"
echo "    - GetAllTrainingMetrics()"
echo "    - GetModelHistory(modelID)"
echo ""
echo "Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(peer|orderer|ca_)"
echo ""
echo -e "${GREEN}Ready for federated learning!${NC}"