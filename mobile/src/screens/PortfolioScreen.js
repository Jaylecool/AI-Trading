import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, ScrollView, RefreshControl,
  ActivityIndicator, TouchableOpacity, Alert,
} from 'react-native';
import api, { readStaleCache, writeCache } from '../api';
import { Colors, Typography } from '../theme';

export default function PortfolioScreen() {
  const [summary, setSummary] = useState(null);
  const [stats, setStats] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    // Show stale cache immediately
    const cached = await readStaleCache('portfolio_screen');
    if (cached) {
      if (cached.summary)   setSummary(cached.summary);
      if (cached.stats)     setStats(cached.stats);
      if (cached.pred)      setPrediction(cached.pred);
      if (cached.positions) setPositions(cached.positions);
      setLoading(false);
    }

    // Fetch fresh data in parallel
    try {
      const [sumRes, statRes, predRes, posRes] = await Promise.all([
        api.get('/api/portfolio/summary').catch(() => null),
        api.get('/api/portfolio/statistics').catch(() => null),
        api.get('/api/next-day-prediction?symbol=AAPL').catch(() => null),
        api.get('/api/broker/positions').catch(() => null),
      ]);
      const summary   = sumRes?.data?.summary  ?? sumRes?.data?.raw  ?? null;
      const stats     = statRes?.data?.detailed ?? null;
      const pred      = predRes?.data           ?? null;
      const positions = posRes?.data            ?? [];
      if (summary)          setSummary(summary);
      if (stats)            setStats(stats);
      if (pred)             setPrediction(pred);
      if (posRes)           setPositions(positions);
      await writeCache('portfolio_screen', { summary, stats, pred, positions }, 60);
    } catch (_) {}
  }, []);

  useEffect(() => { load().finally(() => setLoading(false)); }, [load]);

  const onRefresh = () => {
    setRefreshing(true);
    load().finally(() => setRefreshing(false));
  };

  const closePosition = (symbol) => {
    Alert.alert('Close Position', `Market-sell all shares of ${symbol}?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Close', style: 'destructive', onPress: async () => {
          try {
            await api.post(`/api/broker/position/${symbol}/close`);
            load();
          } catch (e) {
            Alert.alert('Error', e.response?.data?.error || 'Failed to close position');
          }
        }
      }
    ]);
  };

  const fmt = (n) => `$${Number(n || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const fmtPct = (n) => `${Number(n || 0) >= 0 ? '+' : ''}${Number(n || 0).toFixed(2)}%`;
  const signalColor = (s) => s === 'BULLISH' ? Colors.green : s === 'BEARISH' ? Colors.red : Colors.yellow;

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={Colors.primary} size="large" />
      </View>
    );
  }

  // Aggregate stats across all symbols
  const aggStats = stats ? Object.values(stats).reduce((acc, s) => ({
    total_trades: (acc.total_trades || 0) + (s.total_trades || 0),
    winning_trades: (acc.winning_trades || 0) + (s.winning_trades || 0),
    avg_win: Math.max(acc.avg_win || 0, s.avg_win || 0),
    avg_loss: Math.min(acc.avg_loss || 0, s.avg_loss || 0),
    best_trade: Math.max(acc.best_trade || 0, s.best_trade || 0),
    worst_trade: Math.min(acc.worst_trade || 0, s.worst_trade || 0),
    total_pnl: (acc.total_pnl || 0) + (s.total_pnl || 0),
  }), {}) : null;

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />}
    >
      {/* Portfolio Summary */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Portfolio Summary</Text>
        <View style={styles.metricsGrid}>
          {[
            { label: 'Initial Balance', value: fmt(summary?.initial_value ?? summary?.initial_balance ?? 100000) },
            { label: 'Current Balance', value: fmt(summary?.current_value ?? summary?.current_balance) },
            { label: 'Total P&L', value: fmt(summary?.total_pnl), pnl: summary?.total_pnl },
            { label: 'Return %', value: fmtPct(summary?.return_pct ?? summary?.total_return_pct), pnl: summary?.total_pnl },
          ].map(({ label, value, pnl }) => (
            <View key={label} style={styles.metricCard}>
              <Text style={styles.metricLabel}>{label}</Text>
              <Text style={[styles.metricValue, pnl != null && { color: pnl >= 0 ? Colors.green : Colors.red }]}>
                {value}
              </Text>
            </View>
          ))}
        </View>
      </View>

      {/* Trading Signal */}
      {prediction && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Trading Signal (AAPL)</Text>
          <View style={styles.signalCard}>
            <View style={styles.signalHeader}>
              <View style={[styles.signalBadge, {
                backgroundColor: signalColor(prediction.signal) + '22',
                borderColor: signalColor(prediction.signal),
              }]}>
                <Text style={[styles.signalBadgeText, { color: signalColor(prediction.signal) }]}>
                  {prediction.signal}
                </Text>
              </View>
              <Text style={[styles.signalRec, {
                color: prediction.forecast_price > prediction.current_price ? Colors.green : Colors.red,
              }]}>
                {prediction.forecast_price > prediction.current_price ? 'BUY' : 'SELL'}
              </Text>
            </View>
            <View style={styles.signalMetrics}>
              <View>
                <Text style={styles.metricLabel}>Next Forecast</Text>
                <Text style={styles.metricValue}>{fmt(prediction.forecast_price)}</Text>
              </View>
              <View>
                <Text style={styles.metricLabel}>Confidence</Text>
                <Text style={styles.metricValue}>{prediction.confidence_level}%</Text>
              </View>
            </View>
          </View>
        </View>
      )}

      {/* Performance Statistics */}
      {aggStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance Statistics</Text>
          <View style={styles.statsGrid}>
            {[
              { name: 'Total Trades', val: aggStats.total_trades },
              { name: 'Win Rate', val: aggStats.total_trades > 0 ? `${Math.round((aggStats.winning_trades / aggStats.total_trades) * 100)}%` : '0%' },
              { name: 'Avg Win', val: fmt(aggStats.avg_win) },
              { name: 'Avg Loss', val: fmt(aggStats.avg_loss) },
              { name: 'Best Trade', val: fmt(aggStats.best_trade) },
              { name: 'Worst Trade', val: fmt(aggStats.worst_trade) },
            ].map(({ name, val }) => (
              <View key={name} style={styles.statItem}>
                <Text style={styles.statName}>{name}</Text>
                <Text style={styles.statVal}>{val}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Live Positions (Alpaca) */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Open Positions ({positions.length})</Text>
        {positions.length === 0 ? (
          <Text style={styles.empty}>No open positions</Text>
        ) : (
          positions.map((p) => (
            <View key={p.symbol} style={styles.posCard}>
              <View style={styles.posHeader}>
                <Text style={styles.posSymbol}>{p.symbol}</Text>
                <TouchableOpacity onPress={() => closePosition(p.symbol)} style={styles.closeBtn}>
                  <Text style={styles.closeBtnText}>Close</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.posDetails}>
                <View><Text style={styles.detailLabel}>Shares</Text><Text style={styles.detailValue}>{p.qty}</Text></View>
                <View><Text style={styles.detailLabel}>Avg</Text><Text style={styles.detailValue}>{fmt(p.avg_entry_price)}</Text></View>
                <View><Text style={styles.detailLabel}>Current</Text><Text style={styles.detailValue}>{fmt(p.current_price)}</Text></View>
                <View>
                  <Text style={styles.detailLabel}>P&L</Text>
                  <Text style={[styles.detailValue, { color: p.unrealized_pl >= 0 ? Colors.green : Colors.red }]}>
                    {fmtPct(p.unrealized_plpc)}
                  </Text>
                </View>
              </View>
              <Text style={styles.marketValue}>Market Value: {fmt(p.market_value)}</Text>
            </View>
          ))
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg, padding: 16 },
  center: { flex: 1, backgroundColor: Colors.bg, justifyContent: 'center', alignItems: 'center' },
  section: {
    backgroundColor: Colors.card, borderRadius: 12, padding: 16,
    marginBottom: 16, borderWidth: 1, borderColor: Colors.border,
  },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: Colors.text, marginBottom: 12 },
  metricsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  metricCard: {
    flex: 1, minWidth: '45%', backgroundColor: Colors.bg,
    borderRadius: 8, padding: 12, borderWidth: 1, borderColor: Colors.border,
  },
  metricLabel: { color: Colors.textSecondary, fontSize: 11, marginBottom: 4 },
  metricValue: { fontSize: 15, fontWeight: '700', color: Colors.text },
  signalCard: {},
  signalHeader: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 14 },
  signalBadge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: 6, borderWidth: 1 },
  signalBadgeText: { fontWeight: '700', fontSize: 14 },
  signalRec: { fontSize: 16, fontWeight: '700' },
  signalMetrics: { flexDirection: 'row', gap: 24 },
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  statItem: {
    flex: 1, minWidth: '45%', backgroundColor: Colors.bg,
    borderRadius: 8, padding: 12, borderWidth: 1, borderColor: Colors.border,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  statName: { color: Colors.textSecondary, fontSize: 12 },
  statVal: { color: Colors.text, fontSize: 13, fontWeight: '700' },
  posCard: {
    backgroundColor: Colors.bg, borderRadius: 10, padding: 12,
    marginBottom: 10, borderWidth: 1, borderColor: Colors.border,
  },
  posHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  posSymbol: { fontSize: 16, fontWeight: '700', color: Colors.text },
  closeBtn: { paddingHorizontal: 12, paddingVertical: 5, backgroundColor: Colors.red + '22', borderRadius: 6, borderWidth: 1, borderColor: Colors.red },
  closeBtnText: { color: Colors.red, fontSize: 12, fontWeight: '600' },
  posDetails: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  detailLabel: { color: Colors.textSecondary, fontSize: 10 },
  detailValue: { color: Colors.text, fontSize: 13, fontWeight: '600' },
  marketValue: { color: Colors.textSecondary, fontSize: 12 },
  empty: { color: Colors.textSecondary, textAlign: 'center', paddingVertical: 10 },
});

