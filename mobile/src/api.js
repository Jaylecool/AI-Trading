// ============================================================
// API configuration
// Change BASE_URL to your server's IP / public URL.
// When running on a physical device on the same Wi-Fi network,
// use your computer's local IP, e.g. http://192.168.1.42:5000
// ============================================================
export const BASE_URL = 'http://192.168.100.29:5000';

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  withCredentials: true,
});

// Attach session cookie header automatically
api.interceptors.request.use(async (config) => {
  const cookie = await AsyncStorage.getItem('session_cookie');
  if (cookie) {
    config.headers['Cookie'] = cookie;
  }
  return config;
});

// Persist Set-Cookie from login response
api.interceptors.response.use(async (response) => {
  const setCookie = response.headers['set-cookie'];
  if (setCookie) {
    await AsyncStorage.setItem('session_cookie', setCookie[0]);
  }
  return response;
});

export default api;
