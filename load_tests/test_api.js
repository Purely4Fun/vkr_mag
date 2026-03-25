import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '30s', target: 200 },
    { duration: '1m', target: 500 },
    { duration: '30s', target: 1000 },
    { duration: '30s', target: 100 },
  ],
};

function randomTx() {
  return JSON.stringify({
    SENDER_ACCOUNT_ID: Math.floor(Math.random() * 10000) + 1,
    RECEIVER_ACCOUNT_ID: Math.floor(Math.random() * 10000) + 1,
    TX_TYPE: "TRANSFER",
    TX_AMOUNT: Math.random() * 1000,
    TIMESTAMP: Date.now()
  });
}

export default function () {
  http.post(
    'http://localhost:8000/predict',
    randomTx(),
    { headers: { 'Content-Type': 'application/json' } }
  );

  sleep(0.05);
}