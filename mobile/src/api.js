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

// ─── Response cache helpers ───────────────────────────────────────────────────
// Lightweight AsyncStorage-backed cache with TTL.  Used by screens to show
// stale data instantly while fresh data loads in the background.

const CACHE_PREFIX = '@api_cache/';

export async function readCache(key) {
  try {
    const raw = await AsyncStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    const { data, expires } = JSON.parse(raw);
    if (Date.now() > expires) return null;   // expired
    return data;
  } catch {
    return null;
  }
}

export async function writeCache(key, data, ttlSeconds) {
  try {
    const payload = JSON.stringify({ data, expires: Date.now() + ttlSeconds * 1000 });
    await AsyncStorage.setItem(CACHE_PREFIX + key, payload);
  } catch { /* ignore storage errors */ }
}

export async function readStaleCache(key) {
  /** Return cached data even if expired (for stale-while-revalidate). */
  try {
    const raw = await AsyncStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    return JSON.parse(raw).data;
  } catch {
    return null;
  }
}

export default api;
