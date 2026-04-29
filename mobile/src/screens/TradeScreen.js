import React, { useState, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, ScrollView, Alert, Switch,
} from 'react-native';
import api from '../api';
import { Colors, Typography } from '../theme';

const SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA'];

export default function TradeScreen() {
  const [symbol, setSymbol] = useState('AAPL');
  const [side, setSide] = useState('buy');
  const [qty, setQty] = useState('');
  const [orderType, setOrderType] = useState('market');
  const [limitPrice, setLimitPrice] = useState('');
  const [loading, setLoading] = useState(false);
  const [price, setPrice] = useState(null);
  const [priceLoading, setPriceLoading] = useState(false);
  const [result, setResult] = useState(null);

  const fetchPrice = useCallback(async (sym) => {
    setPriceLoading(true);
    setPrice(null);
    try {
      const res = await api.get(`/api/broker/price/${sym}`);
      setPrice(res.data.price);
    } catch (_) {
      setPrice(null);
    } finally {
      setPriceLoading(false);
    }
  }, []);

  const onSymbolChange = (sym) => {
    setSymbol(sym);
    fetchPrice(sym);
  };

  const placeOrder = async () => {
    if (!qty || isNaN(Number(qty)) || Number(qty) <= 0) {
      Alert.alert('Validation', 'Enter a valid quantity greater than 0.');
      return;
    }
    if (orderType === 'limit' && (!limitPrice || isNaN(Number(limitPrice)) || Number(limitPrice) <= 0)) {
      Alert.alert('Validation', 'Enter a valid limit price greater than 0.');
      return;
    }

    const confirmMsg = `${side.toUpperCase()} ${qty} share(s) of ${symbol} at ${orderType === 'market' ? 'market price' : `$${limitPrice}`}`;
    Alert.alert('Confirm Order', confirmMsg, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Confirm', style: 'destructive', onPress: async () => {
          setLoading(true);
          setResult(null);
          try {
            const body = { symbol, side, qty: Number(qty), type: orderType };
            if (orderType === 'limit') body.limit_price = Number(limitPrice);
            const res = await api.post('/api/broker/order', body);
            setResult({ success: true, data: res.data });
          } catch (e) {
            setResult({ success: false, error: e.response?.data?.error || 'Order failed' });
          } finally {
            setLoading(false);
          }
        }
      }
    ]);
  };

  return (
    <ScrollView style={styles.container} keyboardShouldPersistTaps="handled">
      <Text style={styles.heading}>Place Order</Text>

      {/* Symbol picker */}
      <Text style={styles.label}>Symbol</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 14 }}>
        {SYMBOLS.map((s) => (
          <TouchableOpacity
            key={s}
            style={[styles.chip, symbol === s && styles.chipActive]}
            onPress={() => onSymbolChange(s)}
          >
            <Text style={[styles.chipText, symbol === s && styles.chipTextActive]}>{s}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Live price */}
      {priceLoading
        ? <ActivityIndicator color={Colors.primary} style={{ marginBottom: 14 }} />
        : price != null && (
          <Text style={styles.priceDisplay}>
            Last: <Text style={{ color: Colors.primary }}>${Number(price).toFixed(2)}</Text>
          </Text>
        )}

      {/* Side toggle */}
      <Text style={styles.label}>Side</Text>
      <View style={styles.toggleRow}>
        {['buy', 'sell'].map((s) => (
          <TouchableOpacity
            key={s}
            style={[styles.toggle, side === s && { backgroundColor: s === 'buy' ? Colors.green : Colors.red }]}
            onPress={() => setSide(s)}
          >
            <Text style={[styles.toggleText, side === s && { color: '#fff', fontWeight: '700' }]}>
              {s.toUpperCase()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Order type toggle */}
      <Text style={styles.label}>Order Type</Text>
      <View style={styles.toggleRow}>
        {['market', 'limit'].map((t) => (
          <TouchableOpacity
            key={t}
            style={[styles.toggle, orderType === t && { backgroundColor: Colors.primary }]}
            onPress={() => setOrderType(t)}
          >
            <Text style={[styles.toggleText, orderType === t && { color: Colors.bg, fontWeight: '700' }]}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Quantity */}
      <Text style={styles.label}>Quantity (shares)</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. 10"
        placeholderTextColor={Colors.textSecondary}
        keyboardType="numeric"
        value={qty}
        onChangeText={setQty}
      />

      {/* Limit price */}
      {orderType === 'limit' && (
        <>
          <Text style={styles.label}>Limit Price ($)</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 150.00"
            placeholderTextColor={Colors.textSecondary}
            keyboardType="numeric"
            value={limitPrice}
            onChangeText={setLimitPrice}
          />
        </>
      )}

      {/* Submit */}
      <TouchableOpacity style={styles.button} onPress={placeOrder} disabled={loading}>
        {loading ? <ActivityIndicator color={Colors.bg} /> : <Text style={styles.buttonText}>Submit Order</Text>}
      </TouchableOpacity>

      {/* Result */}
      {result && (
        <View style={[styles.resultBox, { borderColor: result.success ? Colors.green : Colors.red }]}>
          {result.success ? (
            <>
              <Text style={[styles.resultTitle, { color: Colors.green }]}>Order Submitted ✓</Text>
              <Text style={styles.resultText}>ID: {result.data.id}</Text>
              <Text style={styles.resultText}>Status: {result.data.status}</Text>
            </>
          ) : (
            <>
              <Text style={[styles.resultTitle, { color: Colors.red }]}>Order Failed</Text>
              <Text style={styles.resultText}>{result.error}</Text>
            </>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg, padding: 16 },
  heading: { ...Typography.h1, marginBottom: 20 },
  label: { color: Colors.textSecondary, fontSize: 13, marginBottom: 6 },
  priceDisplay: { color: Colors.text, fontSize: 15, marginBottom: 14 },
  chip: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
    backgroundColor: Colors.card, borderWidth: 1, borderColor: Colors.border, marginRight: 8,
  },
  chipActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '22' },
  chipText: { color: Colors.textSecondary, fontSize: 13 },
  chipTextActive: { color: Colors.primary, fontWeight: '600' },
  toggleRow: { flexDirection: 'row', marginBottom: 16, gap: 10 },
  toggle: {
    flex: 1, paddingVertical: 12, borderRadius: 10,
    backgroundColor: Colors.card, alignItems: 'center',
    borderWidth: 1, borderColor: Colors.border,
  },
  toggleText: { color: Colors.textSecondary, fontSize: 14 },
  input: {
    backgroundColor: Colors.card, borderColor: Colors.border, borderWidth: 1,
    borderRadius: 10, padding: 14, color: Colors.text, marginBottom: 14, fontSize: 15,
  },
  button: {
    backgroundColor: Colors.primary, padding: 16, borderRadius: 12,
    alignItems: 'center', marginTop: 4, marginBottom: 24,
  },
  buttonText: { color: Colors.bg, fontWeight: '700', fontSize: 16 },
  resultBox: {
    borderWidth: 1, borderRadius: 10, padding: 14, marginBottom: 24,
  },
  resultTitle: { fontSize: 16, fontWeight: '700', marginBottom: 6 },
  resultText: { color: Colors.text, fontSize: 14, marginTop: 2 },
});
