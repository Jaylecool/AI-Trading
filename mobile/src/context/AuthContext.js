import React, { createContext, useContext, useState, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import api from '../api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const login = useCallback(async (email, password) => {
    setLoading(true);
    setError('');
    try {
      const res = await api.post('/api/auth/login', { email, password });
      setUser(res.data.user);
      await AsyncStorage.setItem('user', JSON.stringify(res.data.user));
      return true;
    } catch (e) {
      setError(e.response?.data?.error || 'Login failed');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.post('/api/auth/logout');
    } catch (_) {}
    await AsyncStorage.multiRemove(['user', 'session_cookie']);
    setUser(null);
  }, []);

  const restoreSession = useCallback(async () => {
    const stored = await AsyncStorage.getItem('user');
    if (stored) setUser(JSON.parse(stored));
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, error, login, logout, restoreSession }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
