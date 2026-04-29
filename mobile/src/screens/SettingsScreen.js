import React from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Alert, ScrollView,
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { Colors, Typography } from '../theme';
import api from '../api';

function Row({ label, value }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

export default function SettingsScreen() {
  const { user, logout } = useAuth();

  const closeAll = () => {
    Alert.alert(
      '⚠️ Close All Positions',
      'This will immediately market-sell ALL open positions. This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Close Everything', style: 'destructive', onPress: async () => {
            try {
              const res = await api.post('/api/broker/positions/close-all');
              Alert.alert('Done', `Closed ${res.data.closed?.length ?? 0} position(s).`);
            } catch (e) {
              Alert.alert('Error', e.response?.data?.error || 'Failed to close all positions');
            }
          }
        }
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.heading}>Settings</Text>

      {user && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Account</Text>
          <Row label="Name" value={user.name} />
          <Row label="Email" value={user.email} />
          <Row label="User ID" value={`#${user.id}`} />
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Danger Zone</Text>
        <Text style={styles.danger}>The button below closes all live/paper positions immediately at market price.</Text>
        <TouchableOpacity style={styles.dangerBtn} onPress={closeAll}>
          <Text style={styles.dangerBtnText}>Close All Positions</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.logoutBtn} onPress={logout}>
        <Text style={styles.logoutText}>Sign Out</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg, padding: 16 },
  heading: { ...Typography.h1, marginBottom: 20 },
  card: {
    backgroundColor: Colors.card, borderRadius: 12, padding: 16,
    marginBottom: 16, borderWidth: 1, borderColor: Colors.border,
  },
  cardTitle: { ...Typography.h3, marginBottom: 12, color: Colors.textSecondary },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderColor: Colors.border },
  rowLabel: { color: Colors.textSecondary, fontSize: 14 },
  rowValue: { color: Colors.text, fontSize: 14, fontWeight: '500' },
  danger: { color: Colors.textSecondary, fontSize: 13, marginBottom: 12 },
  dangerBtn: {
    borderWidth: 1, borderColor: Colors.red, borderRadius: 10,
    padding: 13, alignItems: 'center',
  },
  dangerBtnText: { color: Colors.red, fontWeight: '700', fontSize: 14 },
  logoutBtn: {
    backgroundColor: Colors.surface, borderRadius: 10, padding: 15,
    alignItems: 'center', borderWidth: 1, borderColor: Colors.border, marginTop: 4,
  },
  logoutText: { color: Colors.text, fontWeight: '600', fontSize: 15 },
});
