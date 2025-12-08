'use strict';

const { Wallets, Gateway } = require('fabric-network');
const FabricCAServices = require('fabric-ca-client');
const path = require('path');
const fs = require('fs');

async function main() {
    try {
        // Load connection profile
        const ccpPath = path.resolve(__dirname, '..', '..', 'test-network', 'organizations',
            'peerOrganizations', 'org1.example.com', 'connection-org1.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create CA client
        const caURL = ccp.certificateAuthorities['ca.org1.example.com'].url;
        const ca = new FabricCAServices(caURL);

        // Create wallet
        const walletPath = path.join(__dirname, 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);

        // Check if user exists
        const userIdentity = await wallet.get('User1');
        if (userIdentity) {
            console.log('User1 already exists in wallet');
            return;
        }

        // Check for admin
        const adminIdentity = await wallet.get('admin');
        if (!adminIdentity) {
            console.log('Admin not found. Run enrollAdmin.js first.');
            return;
        }

        // Build user object
        const provider = wallet.getProviderRegistry().getProvider(adminIdentity.type);
        const adminUser = await provider.getUserContext(adminIdentity, 'admin');

        // Register user
        const secret = await ca.register({
            affiliation: 'org1.department1',
            enrollmentID: 'User1',
            role: 'client'
        }, adminUser);

        // Enroll user
        const enrollment = await ca.enroll({
            enrollmentID: 'User1',
            enrollmentSecret: secret
        });

        // Import to wallet
        const x509Identity = {
            credentials: {
                certificate: enrollment.certificate,
                privateKey: enrollment.key.toBytes()
            },
            mspId: 'Org1MSP',
            type: 'X.509'
        };
        await wallet.put('User1', x509Identity);
        console.log('User1 enrolled and imported to wallet');

    } catch (error) {
        console.error(`Failed to register user: ${error}`);
        process.exit(1);
    }
}

main();