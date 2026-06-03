import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, FlatList,
  RefreshControl, ActivityIndicator, TouchableOpacity,
} from 'react-native';
import api, { readStaleCache, writeCache } from '../api';
import { Colors } from '../theme';

const TABS = ['History', 'Live Orders'];

export default function OrdersScreen() {
  const [tab, setTab] = useState('History');
  const [trades, setTrades] = useState([]);
  const [orders, setOrders] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadHistory = useCallback(async () => {
    // Show stale cache first
    const cached = await readStaleCache('orders_history');
    if (cached) {
      if (cached.trades)  setTrades(cached.trades);
      if (cached.summary) setSummary(cached.summary);
      setLoading(false);
    }
    try {
      const [trRes, sumRes] = await Promise.all([
        api.get('/api/trades/history?limit=50').catch(() => null),
        api.get('/api/portfolio/summary').catch(() => null),
      ]);
      const trades  = trRes?.data?.trades ?? [];
      const summary = sumRes?.data?.summary ?? sumRes?.data?.raw ?? null;
      if (trRes)  setTrades(trades);
      if (sumRes) setSummary(summary);
      await writeCache('orders_history', { trades, summary }, 60);
    } catch (_) {}
  }, []);

  const loadOrders = useCallback(async () => {
    try {
      const res = await api.get('/api/broker/orders?status=all');
      setOrders(res.data || []);
    } catch (_) {}
  }, []);

  const load = useCallback(async () => {
    if (tab === 'History') await loadHistory();
    else await loadOrders();
  }, [tab, loadHistory, loadOrders]);

  useEffect(() => {
    setLoading(true);
    load().finally(() => setLoading(false));
  }, [tab]);

  const onRefresh = () => {
    setRefreshing(true);
    load().finally(() => setRefreshing(false));
  };

  const fmt = (n) => n == null ? 'â€”' : `$${Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const fmtPct = (n) => n == null ? 'â€”' : `${Number(n) >= 0 ? '+' : ''}${Number(n).toFixed(2)}%`;

  const statusColor = (s) => {
    if (!s) return Colors.textSecondary;
    const sl = s.toLowerCase();
    if (sl.includes('fill')) return Colors.green;
    if (sl.includes('cancel') || sl.includes('reject')) return Colors.red;
    return Colors.yellow;
  };

  const renderTrade = ({ item: t }) => {
    const pnl = t.pnl ?? t.total_pnl ?? t.profit_loss;
    const pnlColor = pnl >= 0 ? Colors.green : Colors.red;
    const action = t.action || t.type || 'â€”';
    return (
      <View style={styles.row}>
        <View style={styles.rowLeft}>
          <View style={styles.rowTop}>
            <Text style={styles.symbol}>{t.symbol}</Text>
            <Text style={[styles.action, { color: action.toUpperCase() === 'BUY' ? Colors.green : Colors.red }]}>
              {action.toUpperCase()}
            </Text>
          </View>
          <Text style={styles.sub}>{t.date || t.entry_date} â†’ {t.exit_date || 'â€”'}</Text>
          <Text style={styles.sub}>
            {fmt(t.entry_price)} â†’ {fmt(t.exit_price)}
          </Text>
        </View>
        <View style={styles.rowRight}>
          <Text style={[styles.pnl, { color: pnlColor }]}>{fmt(pnl)}</Text>
          <Text style={[styles.pnlPct, { color: pnlColor }]}>{fmtPct(t.return_pct ?? t.pnl_pct)}</Text>
          <Text style={styles.qty}>{t.quantity ?? t.qty} shares</Text>
        </View>
      </View>
    );
  };

  const renderOrder = ({ item: o }) => (
    <View style={styles.row}>
      <View style={styles.rowLeft}>
        <View style={styles.rowTop}>
          <Text style={styles.symbol}>{o.symbol}</Text>
          <Text style={[styles.action, { color: o.side?.toString().toLowerCase() === 'buy' ? Colors.green : Colors.red }]}>
            {o.side?.toString().toUpperCase()}
          </Text>
        </View>
        <Text style={styles.sub}>{o.type} Â· {o.qty} shares{o.limit_price ? ` @ $${o.limit_price}` : ''}</Text>
        <Text style={styles.sub}>Filled: {o.filled_qty}/{o.qty}</Text>
      </View>
      <View style={styles.rowRight}>
        <Text style={[styles.orderStatus, { color: statusColor(o.status?.toString()) }]}>
          {o.status?.toString().toUpperCase()}
        </Text>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Tab Switcher */}
      <View style={styles.tabRow}>
        {TABS.map((t) => (
          <TouchableOpacity
            key={t}
            style={[styles.tabBtn, tab === t && styles.tabBtnActive]}
            onPress={() => setTab(t)}
          >
            <Text style={[styles.tabText, tab === t && styles.tabTextActive]}>{t}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Stats Header (History tab) */}
      {tab === 'History' && summary && (
        <View style={styles.statsBar}>
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{trades.length}</Text>
            <Text style={styles.statLabel}>Trades</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={[styles.statNum, { color: (summary.total_pnl ?? 0) >= 0 ? Colors.green : Colors.red }]}>
              {fmt(summary.total_pnl ?? summary.net_pnl)}
            </Text>
            <Text style={styles.statLabel}>Total P&L</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNum}>{summary.win_rate != null ? `${Number(summary.win_rate).toFixed(0)}%` : 'â€”'}</Text>
            <Text style={styles.statLabel}>Win Rate</Text>
          </View>
        </View>
      )}

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={Colors.primary} size="large" />
        </View>
      ) : (
        <FlatList
          data={tab === 'History' ? trades : orders}
          keyExtractor={(item, i) => item.id?.toString() || item.trade_id?.toString() || i.toString()}
          renderItem={tab === 'History' ? renderTrade : renderOrder}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />}
          contentContainerStyle={{ padding: 12 }}
          ListEmptyComponent={<Text style={styles.empty}>{tab === 'History' ? 'No trade history yet' : 'No orders found'}</Text>}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  tabRow: { flexDirection: 'row', backgroundColor: Colors.surface, borderBottomWidth: 1, borderColor: Colors.border },
  tabBtn: { flex: 1, paddingVertical: 12, alignItems: 'center', borderBottomWidth: 2, borderColor: 'transparent' },
  tabBtnActive: { borderColor: Colors.primary },
  tabText: { color: Colors.textSecondary, fontWeight: '600', fontSize: 14 },
  tabTextActive: { color: Colors.primary },
  statsBar: {
    flexDirection: 'row', backgroundColor: Colors.card,
    paddingVertical: 12, paddingHorizontal: 16,
    borderBottomWidth: 1, borderColor: Colors.border,
  },
  statItem: { flex: 1, alignItems: 'center' },
  statNum: { fontSize: 16, fontWeight: '700', color: Colors.text },
  statLabel: { color: Colors.textSecondary, fontSize: 11, marginTop: 2 },
  row: {
    backgroundColor: Colors.card, borderRadius: 10, padding: 14, marginBottom: 10,
    flexDirection: 'row', borderWidth: 1, borderColor: Colors.border,
  },
  rowLeft: { flex: 1 },
  rowRight: { alignItems: 'flex-end', justifyContent: 'center' },
  rowTop: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 },
  symbol: { fontSize: 15, fontWeight: '700', color: Colors.text },
  action: { fontSize: 12, fontWeight: '700' },
  sub: { color: Colors.textSecondary, fontSize: 12, marginTop: 2 },
  pnl: { fontSize: 15, fontWeight: '700' },
  pnlPct: { fontSize: 12, marginTop: 2 },
  qty: { color: Colors.textSecondary, fontSize: 11, marginTop: 4 },
  orderStatus: { fontSize: 13, fontWeight: '700' },
  empty: { color: Colors.textSecondary, textAlign: 'center', marginTop: 40 },
});

