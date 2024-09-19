// firebase.js
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyB4ox1-F75qRuXyyppKHHEHu-yLgojHEK0",
    authDomain: "deco-windash.firebaseapp.com",
    projectId: "deco-windash",
    storageBucket: "deco-windash.appspot.com",
    messagingSenderId: "664553408702",
    appId: "1:664553408702:web:49d53579c57464310ef6ef"
  };

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };