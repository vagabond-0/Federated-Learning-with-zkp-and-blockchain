#!/bin/bash

# VPSA Chaincode Testing Script

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}🧪 TEST $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

cd /home/amalendu/college/federatedLearning/fabric-samples/test-network

# Set environment for Org1
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

echo "========================================="
echo "VPSA CHAINCODE FUNCTIONAL TESTS"
echo "========================================="
echo ""

print_test "1" "Register Source Domain Client"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"RegisterClient","Args":["client-org1-source","source","10000"]}'
sleep 3
print_success "Source client registered"

print_test "2" "Register Target Domain Client"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"RegisterClient","Args":["client-org2-target","target","8000"]}'
sleep 3
print_success "Target client registered"

print_test "3" "Query All Clients"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetAllClients","Args":[]}'
print_success "Clients retrieved"

print_test "4" "Submit Local Model from Source Domain"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"SubmitLocalModel","Args":["model-source-r0","client-org1-source","{\"layer1\":[0.5,0.3],\"layer2\":[0.8,0.2]}","{\"latent_dim\":768,\"features\":[0.1,0.2]}","{\"class1\":[0.9,0.1],\"class2\":[0.2,0.8]}","0.85","0.15","0.05","1000"]}'
sleep 3
print_success "Source model submitted"

print_test "5" "Submit Local Model from Target Domain"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"SubmitLocalModel","Args":["model-target-r0","client-org2-target","{\"layer1\":[0.4,0.6],\"layer2\":[0.7,0.3]}","{\"latent_dim\":768,\"features\":[0.15,0.25]}","{\"class1\":[0.85,0.15],\"class2\":[0.25,0.75]}","0.78","0.22","0.08","800"]}'
sleep 3
print_success "Target model submitted"

print_test "6" "Query Local Model"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetLocalModel","Args":["model-source-r0"]}'
print_success "Local model retrieved"

print_test "7" "Aggregate Models"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"AggregateModels","Args":["[\"model-source-r0\",\"model-target-r0\"]","{\"aggregated_layer1\":[0.45,0.45],\"aggregated_layer2\":[0.75,0.25]}","{\"aggregated_class1\":[0.875,0.125],\"aggregated_class2\":[0.225,0.775]}","0.82","0.18","0.92"]}'
sleep 3
print_success "Models aggregated"

print_test "8" "Query Updated Global Model"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetGlobalModel","Args":[]}'
print_success "Global model retrieved"

print_test "9" "Query Training Metrics"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetTrainingMetrics","Args":["0"]}'
print_success "Training metrics retrieved"

print_test "10" "Update Aggregation Config"
peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls \
    --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
    -C vpsa-channel \
    -n vpsa \
    --peerAddresses localhost:7051 \
    --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
    -c '{"function":"UpdateAggregationConfig","Args":["5","0.7","0.3","0.15"]}'
sleep 3
print_success "Config updated"

print_test "11" "Query Model History"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetModelHistory","Args":["vpsa-global-model"]}'
print_success "Model history retrieved"

print_test "12" "Query Client Details"
peer chaincode query \
    -C vpsa-channel \
    -n vpsa \
    -c '{"function":"GetClient","Args":["client-org1-source"]}'
print_success "Client details retrieved"

echo ""
echo "========================================="
echo "✅ ALL TESTS COMPLETED SUCCESSFULLY!"
echo "========================================="