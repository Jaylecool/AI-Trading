import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, ScrollView, RefreshControl,
  ActivityIndicator, TouchableOpacity, Switch, Alert,
} from 'react-native';
import api from '../api';
import { Colors, Typography } from '../theme';

const SYMBOLS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META'];
const STRATEGIES = ['AUTO', 'AGGRESSIVE', 'BALANCED', 'CONSERVATIVE'];

export default function DashboardScreen() {
  const [symbol, setSymbol] = useState('AAPL');
  const [livePrice, setLivePrice] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [autoTrade, setAutoTrade] = useState(null);
  const [account, setAccount] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [stratLoading, setStratLoading] = useState(false);

  const load = useCallback(async (sym = symbol) => {
    try {
      const [priceRes, predRes, autoRes, acctRes] = await Promise.all([
        api.get(`/api/portfolio/live-price?symbol=${sym}`).catch(() => null),
        api.get(`/api/next-day-prediction?symbol=${sym}`).catch(() => null),
        api.get('/api/auto-trade/status').catch(() => null),
        api.get('/api/broker/account').catch(() => null),
      ]);
      if (priceRes) setLivePrice(priceRes.data);
      if (predRes) setPrediction(predRes.data);
      if (autoRes) setAutoTrade(autoRes.data);
      if (acctRes) setAccount(acctRes.data);
    } catch (_) {}
  }, [symbol]);

  useEffect(() => {
    setLoading(true);
    load().finally(() => setLoading(false));
  }, [symbol]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    load().finally(() => setRefreshing(false));
  }, [load]);

  const switchSymbol = (sym) => {
    setSymbol(sym);
    setLivePrice(null);
    setPrediction(null);
  };

  const toggleAutoTrade = async () => {
    if (!autoTrade) return;
    try {
      const res = await api.post('/api/auto-trade/toggle');
      setAutoTrade((prev) => ({ ...prev, enabled: res.data.enabled }));
    } catch (e) {
      Alert.alert('Error', 'Could not toggle auto-trading');
    }
  };

  const setStrategy = async (strat) => {
    setStratLoading(true);
    try {
      const res = await api.post('/api/auto-trade/strategy', { strategy: strat });
      setAutoTrade((prev) => ({ ...prev, active_strategy: res.data.active_strategy }));
    } catch (_) {}
    setStratLoading(false);
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={Colors.primary} size="large" />
      </View>
    );
  }

  const fmt = (n) => n == null ? '—' : `$${Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const fmtPct = (n) => n == null ? '—' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(2)}%`;
  const signalColor = (s) => s === 'BULLISH' ? Colors.green : s === 'BEARISH' ? Colors.red : Colors.yellow;

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />}
    >
      {/* Symbol Selector */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.symbolRow}>
        {SYMBOLS.map((s) => (
          <TouchableOpacity
            key={s}
            style={[styles.chip, symbol === s && styles.chipActive]}
            onPress={() => switchSymbol(s)}
          >
            <Text style={[styles.chipText, symbol === s && styles.chipTextActive]}>{s}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Live Price */}
      <View style={styles.priceCard}>
        <View style={{ flex: 1 }}>
          <Text style={styles.priceSymbol}>{symbol}</Text>
          {livePrice ? (
            <>
              <Text style={styles.priceValue}>{fmt(livePrice.current_price)}</Text>
              <Text style={[styles.priceChange, { color: livePrice.change >= 0 ? Colors.green : Colors.red }]}>
                {livePrice.change >= 0 ? '+' : ''}{fmt(livePrice.change)} ({fmtPct(livePrice.change_percent)})
              </Text>
            </>
          ) : (
            <ActivityIndicator color={Colors.primary} style={{ marginTop: 8 }} />
          )}
        </View>
        {account && (
          <View style={{ alignItems: 'flex-end' }}>
            <Text style={styles.acctLabel}>Portfolio</Text>
            <Text style={styles.acctValue}>{fmt(account.portfolio_value)}</Text>
            <Text style={[styles.acctPnl, { color: account.pnl_today >= 0 ? Colors.green : Colors.red }]}>
              Today: {fmtPct(account.pnl_today_pct)}
            </Text>
          </View>
        )}
      </View>

      {/* AI Prediction Panel */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Next-Day AI Prediction</Text>
        {prediction ? (
          <View style={styles.predCard}>
            <View style={styles.predRow}>
              <View style={styles.predItem}>
                <Text style={styles.predLabel}>Forecast Price</Text>
                <Text style={styles.predPrice}>{fmt(prediction.forecast_price)}</Text>
                <Text style={[styles.predChange, {
                  color: prediction.forecast_price > prediction.current_price ? Colors.green : Colors.red
                }]}>
                  {prediction.forecast_price > prediction.current_price ? '↑' : '↓'}{' '}
                  {fmtPct(((prediction.forecast_price - prediction.current_price) / prediction.current_price) * 100)}
                </Text>
              </View>
              <View style={styles.predItem}>
                <Text style={styles.predLabel}>Current Price</Text>
                <Text style={styles.predPrice}>{fmt(prediction.current_price)}</Text>
              </View>
            </View>

            {/* Signal */}
            <View style={styles.signalRow}>
              <View style={[styles.signalBadge, { backgroundColor: signalColor(prediction.signal) + '22', borderColor: signalColor(prediction.signal) }]}>
                <Text style={[styles.signalText, { color: signalColor(prediction.signal) }]}>
                  {prediction.signal || 'NEUTRAL'}
                </Text>
              </View>
              <Text style={styles.modelText}>{prediction.model}</Text>
            </View>

            {/* Confidence Bar */}
            <Text style={styles.predLabel}>Confidence: {prediction.confidence_level}%</Text>
            <View style={styles.confBarBg}>
              <View style={[styles.confBarFill, {
                width: `${prediction.confidence_level || 0}%`,
                backgroundColor: prediction.confidence_level > 70 ? Colors.green : prediction.confidence_level > 50 ? Colors.yellow : Colors.red,
              }]} />
            </View>
          </View>
        ) : (
          <ActivityIndicator color={Colors.primary} />
        )}
      </View>

      {/* Auto-Trade Widget */}
      {autoTrade && (
        <View style={styles.section}>
          <View style={styles.autoHeader}>
            <View style={styles.autoTitleRow}>
              <View style={[styles.dot, { backgroundColor: autoTrade.enabled ? Colors.green : Colors.red }]} />
              <Text style={styles.sectionTitle}>Auto-Trading</Text>
            </View>
            <Switch
              value={autoTrade.enabled}
              onValueChange={toggleAutoTrade}
              trackColor={{ false: Colors.card, true: Colors.primary + '66' }}
              thumbColor={autoTrade.enabled ? Colors.primary : Colors.textSecondary}
            />
          </View>

          {/* Stats */}
          <View style={styles.autoStats}>
            <View style={styles.autoStat}>
              <Text style={styles.autoStatNum}>{autoTrade.open_positions}</Text>
              <Text style={styles.autoStatLabel}>Open Positions</Text>
            </View>
            <View style={styles.autoStat}>
              <Text style={styles.autoStatNum}>{autoTrade.total_auto_trades}</Text>
              <Text style={styles.autoStatLabel}>Total Trades</Text>
            </View>
          </View>

          {/* Strategy Selector */}
          <Text style={[styles.predLabel, { marginBottom: 8 }]}>Strategy</Text>
          <View style={styles.stratRow}>
            {STRATEGIES.map((s) => (
              <TouchableOpacity
                key={s}
                style={[styles.stratBtn, autoTrade.active_strategy === s && styles.stratBtnActive]}
                onPress={() => setStrategy(s)}
                disabled={stratLoading}
              >
                <Text style={[styles.stratText, autoTrade.active_strategy === s && styles.stratTextActive]}>
                  {s === 'AUTO' ? 'Auto' : s.charAt(0) + s.slice(1).toLowerCase()}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Last Action */}
          {autoTrade.recent_actions?.length > 0 && (() => {
            const a = autoTrade.recent_actions[0];
            const label = typeof a === 'string' ? a
              : `${a.action || ''} ${a.symbol || ''} ${a.shares ? `x${a.shares}` : ''} ${a.price ? `@ $${Number(a.price).toFixed(2)}` : ''} ${a.time ? `(${String(a.time).slice(0,10)})` : ''}`.trim();
            return <Text style={styles.lastAction}>Last: {label}</Text>;
          })()}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg, padding: 16 },
  center: { flex: 1, backgroundColor: Colors.bg, justifyContent: 'center', alignItems: 'center' },
  symbolRow: { marginBottom: 16 },
  chip: {
    paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20,
    backgroundColor: Colors.card, borderWidth: 1, borderColor: Colors.border, marginRight: 8,
  },
  chipActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '22' },
  chipText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  chipTextActive: { color: Colors.primary },
  priceCard: {
    backgroundColor: Colors.card, borderRadius: 12, padding: 16, marginBottom: 16,
    borderWidth: 1, borderColor: Colors.border, flexDirection: 'row', alignItems: 'center',
  },
  priceSymbol: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600', marginBottom: 4 },
  priceValue: { fontSize: 28, fontWeight: '700', color: Colors.text },
  priceChange: { fontSize: 14, marginTop: 2 },
  acctLabel: { color: Colors.textSecondary, fontSize: 11 },
  acctValue: { color: Colors.text, fontSize: 16, fontWeight: '700' },
  acctPnl: { fontSize: 12, marginTop: 2 },
  section: {
    backgroundColor: Colors.card, borderRadius: 12, padding: 16, marginBottom: 16,
    borderWidth: 1, borderColor: Colors.border,
  },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: Colors.text, marginBottom: 12 },
  predCard: {},
  predRow: { flexDirection: 'row', marginBottom: 14 },
  predItem: { flex: 1 },
  predLabel: { color: Colors.textSecondary, fontSize: 12, marginBottom: 4 },
  predPrice: { fontSize: 20, fontWeight: '700', color: Colors.text },
  predChange: { fontSize: 13, marginTop: 2 },
  signalRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 14, gap: 12 },
  signalBadge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: 6, borderWidth: 1 },
  signalText: { fontWeight: '700', fontSize: 14 },
  modelText: { color: Colors.textSecondary, fontSize: 11, flex: 1 },
  confBarBg: { height: 8, backgroundColor: Colors.bg, borderRadius: 4, marginTop: 6, overflow: 'hidden' },
  confBarFill: { height: '100%', borderRadius: 4 },
  autoHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 },
  autoTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  autoStats: { flexDirection: 'row', marginBottom: 16, gap: 12 },
  autoStat: {
    flex: 1, backgroundColor: Colors.bg, borderRadius: 10, padding: 12,
    alignItems: 'center', borderWidth: 1, borderColor: Colors.border,
  },
  autoStatNum: { fontSize: 22, fontWeight: '700', color: Colors.text },
  autoStatLabel: { color: Colors.textSecondary, fontSize: 11, marginTop: 2 },
  stratRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  stratBtn: {
    flex: 1, paddingVertical: 8, borderRadius: 8, alignItems: 'center',
    backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.border,
  },
  stratBtnActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '22' },
  stratText: { color: Colors.textSecondary, fontSize: 11, fontWeight: '600' },
  stratTextActive: { color: Colors.primary },
  lastAction: { color: Colors.textSecondary, fontSize: 12, fontStyle: 'italic' },
});
