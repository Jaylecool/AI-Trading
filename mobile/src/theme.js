// Shared dark-theme colour palette & typography
export const Colors = {
  bg: '#0d1117',
  surface: '#161b22',
  card: '#1c2128',
  border: '#30363d',
  primary: '#00d4ff',
  green: '#3fb950',
  red: '#f85149',
  yellow: '#e3b341',
  text: '#e6edf3',
  textSecondary: '#8b949e',
  white: '#ffffff',
};

export const Typography = {
  h1: { fontSize: 24, fontWeight: '700', color: Colors.text },
  h2: { fontSize: 18, fontWeight: '600', color: Colors.text },
  h3: { fontSize: 15, fontWeight: '600', color: Colors.text },
  body: { fontSize: 14, color: Colors.text },
  small: { fontSize: 12, color: Colors.textSecondary },
  mono: { fontSize: 13, fontFamily: 'monospace', color: Colors.text },
};
