import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';

import { AuthProvider, useAuth } from './context/AuthContext';
import LoginScreen from './screens/LoginScreen';
import DashboardScreen from './screens/DashboardScreen';
import TradeScreen from './screens/TradeScreen';
import OrdersScreen from './screens/OrdersScreen';
import PortfolioScreen from './screens/PortfolioScreen';
import SettingsScreen from './screens/SettingsScreen';
import { Colors } from './theme';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const tabBarStyle = {
  backgroundColor: Colors.surface,
  borderTopColor: Colors.border,
};

const screenOptions = ({ route }) => ({
  tabBarIcon: ({ focused, color, size }) => {
    const icons = {
      Dashboard: focused ? 'home' : 'home-outline',
      Trade: focused ? 'trending-up' : 'trending-up-outline',
      History: focused ? 'list' : 'list-outline',
      Portfolio: focused ? 'pie-chart' : 'pie-chart-outline',
      Settings: focused ? 'settings' : 'settings-outline',
    };
    return <Ionicons name={icons[route.name]} size={size} color={color} />;
  },
  tabBarActiveTintColor: Colors.primary,
  tabBarInactiveTintColor: Colors.textSecondary,
  tabBarStyle,
  headerStyle: { backgroundColor: Colors.surface },
  headerTintColor: Colors.text,
  headerTitleStyle: { fontWeight: '700' },
});

function MainTabs() {
  return (
    <Tab.Navigator screenOptions={screenOptions}>
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Trade" component={TradeScreen} />
      <Tab.Screen name="History" component={OrdersScreen} />
      <Tab.Screen name="Portfolio" component={PortfolioScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

function RootNavigator() {
  const { user, restoreSession } = useAuth();

  useEffect(() => {
    restoreSession();
  }, []);

  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <Stack.Screen name="Main" component={MainTabs} />
        ) : (
          <Stack.Screen name="Login" component={LoginScreen} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <RootNavigator />
    </AuthProvider>
  );
}
