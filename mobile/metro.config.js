const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// On Windows, Metro crashes with ENOENT when React Native extracts a temp
// Gradle plugin directory and then deletes it while Metro is watching.
// Fix: spread the existing blockList array and append our exclusion pattern.
const existing = config.resolver.blockList;
const gradlePluginPattern = /node_modules[\\/]@react-native[\\/]\.gradle-plugin-[^/\\]+[\\/]/;

config.resolver.blockList = [
  ...(Array.isArray(existing) ? existing : existing ? [existing] : []),
  gradlePluginPattern,
];

module.exports = config;
