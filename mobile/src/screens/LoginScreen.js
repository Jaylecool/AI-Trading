import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, KeyboardAvoidingView, Platform,
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { Colors } from '../theme';

export default function LoginScreen() {
  const { login, loading, error } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.title}>AI Trading</Text>
      <Text style={styles.subtitle}>Sign in to your account</Text>

      {!!error && <Text style={styles.error}>{error}</Text>}

      <TextInput
        style={styles.input}
        placeholder="Email"
        placeholderTextColor={Colors.textSecondary}
        autoCapitalize="none"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor={Colors.textSecondary}
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />

      <TouchableOpacity
        style={styles.button}
        onPress={() => login(email, password)}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color={Colors.bg} />
        ) : (
          <Text style={styles.buttonText}>Sign In</Text>
        )}
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1, backgroundColor: Colors.bg,
    alignItems: 'center', justifyContent: 'center', padding: 28,
  },
  title: { fontSize: 36, fontWeight: '800', color: Colors.primary, marginBottom: 4 },
  subtitle: { fontSize: 15, color: Colors.textSecondary, marginBottom: 32 },
  error: { color: Colors.red, marginBottom: 12, textAlign: 'center' },
  input: {
    width: '100%', backgroundColor: Colors.card,
    borderColor: Colors.border, borderWidth: 1, borderRadius: 10,
    padding: 14, color: Colors.text, marginBottom: 14, fontSize: 15,
  },
  button: {
    width: '100%', backgroundColor: Colors.primary,
    padding: 15, borderRadius: 10, alignItems: 'center', marginTop: 4,
  },
  buttonText: { color: Colors.bg, fontWeight: '700', fontSize: 16 },
});
