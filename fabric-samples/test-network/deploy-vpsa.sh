#!/bin/bash

# VPSA Chaincode Deployment Script
# This script automates the complete deployment of VPSA chaincode on Hyperledger Fabric

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

# Change to test-network directory
cd /home/amalendu/college/federatedLearning/fabric-samples/test-network

print_section "STEP 1: CLEANING UP EXISTING NETWORK"
./network.sh down
docker volume prune -f
rm -f vpsa.tar.gz
print_info "Cleanup completed"

print_section "STEP 2: STARTING NETWORK"
./network.sh up createChannel -c vpsa-channel -ca
print_info "Network started with channel: vpsa-channel"

print_section "STEP 3: SETTING ENVIRONMENT VARIABLES"
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/
export CORE_PEER_TLS_ENABLED=true
print_info "Environment configured"

print_section "STEP 4: PACKAGING CHAINCODE"
peer lifecycle chaincode package vpsa.tar.gz \
    --path ../federated-learning/chaincode \
    --lang golang \
    --label vpsa_1.0
print_info "Chaincode packaged: vpsa.tar.gz"

print_section "STEP 5: INSTALLING ON ORG1"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

peer lifecycle chaincode install vpsa.tar.gz
print_info "Chaincode installed on Org1"

print_section "STEP 6: INSTALLING ON ORG2"
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
export CORE_PEER_ADDRESS=localhost:9051

peer lifecycle chaincode install vpsa.tar.gz
print_info "Chaincode installed on Org2"

print_section "STEP 7: GETTING PACKAGE ID"
peer lifecycle chaincode queryinstalled

export CC_PACKAGE_ID=$(peer lifecycle chaincode queryinstalled 2>&1 | grep vpsa_1.0 | sed -n 's/^Package ID: //; s/, Label:.*$//; p')

if [ -z "$CC_PACKAGE_ID" ]; then
    print_error "Package ID not found!"
    exit 1
fi

print_info "Package ID: $CC_PACKAGE_ID"

print_section "STEP 8: APPROVING FOR ORG2"
peer lifecycle chaincode approveformyorg \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID vpsa-channel \
    --name vpsa \
    --version 1.0 \
    --package-id $CC_PACKAGE_ID \
    --sequence 1 \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
print_info "Approved for Org2"

print_section "STEP 9: APPROVING FOR ORG1"
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_ADDRESS=localhost:7051

peer lifecycle chaincode approveformyorg \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID vpsa-channel \
    --name vpsa \
    --version 1.0 \
    --package-id $CC_PACKAGE_ID \
    --sequence 1 \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
print_info "Approved for Org1"

print_section "STEP 10: CHECKING COMMIT READINESS"
peer lifecycle chaincode checkcommitreadiness \
    --channelID vpsa-channel \
    --name vpsa \
    --version 1.0 \
    --sequence 1 \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    --output json

print_section "STEP 11: COMMITTING CHAINCODE"
peer lifecycle chaincode commit \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --channelID vpsa-channel \
    --name vpsa \
    --version 1.0 \
    --sequence 1 \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    --peerAddresses localhost:9051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
print_info "Chaincode committed"

print_section "STEP 12: VERIFYING DEPLOYMENT"
peer lifecycle chaincode querycommitted --channelID vpsa-channel --name vpsa

print_section "STEP 13: INITIALIZING LEDGER"
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
    -c '{"function":"InitLedger","Args":[]}'

print_info "Waiting for chaincode containers to start..."
sleep 5

print_section "STEP 14: TESTING DEPLOYMENT"
print_info "Querying Global Model..."
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetGlobalModel","Args":[]}'

echo ""
print_info "Querying Aggregation Config..."
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetAggregationConfig","Args":[]}'

print_section "✅ VPSA CHAINCODE DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo ""
echo "You can now run the test script to test all functions:"
echo "  bash test-vpsa.sh"
echo ""