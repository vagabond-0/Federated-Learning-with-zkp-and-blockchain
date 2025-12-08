'use strict';

const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Paths configuration
const cryptoPath = path.resolve(__dirname, '..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const certPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'signcerts', 'cert.pem');
const keyPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'keystore');
const tlsCertPath = path.resolve(cryptoPath, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');
const peerEndpoint = 'localhost:7051';
const peerHostAlias = 'peer0.org1.example.com';

/**
 * Create a new gRPC connection to the peer
 */
async function newGrpcConnection() {
    const tlsRootCert = fs.readFileSync(tlsCertPath);
    const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
    return new grpc.Client(peerEndpoint, tlsCredentials, {
        'grpc.ssl_target_name_override': peerHostAlias,
    });
}

/**
 * Get user identity
 */
function newIdentity() {
    const credentials = fs.readFileSync(certPath);
    return { mspId: 'Org1MSP', credentials };
}

/**
 * Get user signer
 */
function newSigner() {
    const files = fs.readdirSync(keyPath);
    const keyFile = path.resolve(keyPath, files[0]);
    const privateKeyPem = fs.readFileSync(keyFile);
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}

/**
 * Connect to Fabric Gateway
 */
async function connectGateway() {
    const client = await newGrpcConnection();
    const gateway = connect({
        client,
        identity: newIdentity(),
        signer: newSigner(),
        evaluateOptions: () => ({ deadline: Date.now() + 5000 }),
        endorseOptions: () => ({ deadline: Date.now() + 15000 }),
        submitOptions: () => ({ deadline: Date.now() + 5000 }),
        commitStatusOptions: () => ({ deadline: Date.now() + 60000 }),
    });

    return { gateway, client };
}

/**
 * Get the VPSA contract
 */
async function getContract(gateway) {
    const network = gateway.getNetwork('vpsa-channel');
    return network.getContract('vpsa');
}

module.exports = {
    connectGateway,
    getContract
};