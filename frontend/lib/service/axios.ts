import axios from "axios";

export const API = axios.create({
  baseURL: "http://localhost:8000/", //ANCHOR - Change this to your backend API URL and use .env
});
