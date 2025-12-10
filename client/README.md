# Federated Learning with ZKP & Blockchain - Frontend

A Next.js application for privacy-preserving federated learning with in-browser model training, blockchain integration via Hyperledger Fabric, and secure multi-party computation.

## 🚀 Features

- **In-Browser Training**: Train machine learning models directly in the browser using TensorFlow.js
- **Privacy-Preserving**: Model weights are split into private collections before submission
- **Blockchain Integration**: All operations recorded on Hyperledger Fabric for transparency
- **Secure Prediction**: Make predictions using secret sharing and MPC
- **VPSA Aggregation**: Verifiable Privacy-preserving Secure Aggregation with coordinate-wise trimming

## 📋 Prerequisites

1. **Flask Backend Running**: The backend API must be running at `http://localhost:5000`
   - Navigate to: `/home/amalendumanoj/project/Federated-Learning-with-zkp-and-blockchain/fabric-samples/federated-learning/flask-backend`
   - Run: `python app.py`

2. **Hyperledger Fabric Network**: The Fabric network must be up and running
   - Navigate to: `/home/amalendumanoj/project/Federated-Learning-with-zkp-and-blockchain/fabric-samples/test-network`
   - Ensure network is started with VPSA chaincode deployed

3. **Node.js**: Version 18+ recommended

## 🛠️ Installation

1. **Install dependencies**:
```bash
npm install
```

2. **Run the development server**:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
