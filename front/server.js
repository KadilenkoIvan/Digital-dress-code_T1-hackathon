/**
 * Custom HTTP server with headers for SharedArrayBuffer support
 * 
 * This server adds the necessary headers to enable:
 * - Multi-threaded WebAssembly (SharedArrayBuffer)
 * - WebGPU acceleration
 * 
 * Required headers:
 * - Cross-Origin-Embedder-Policy: require-corp
 * - Cross-Origin-Opener-Policy: same-origin
 */

const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware to add required headers for SharedArrayBuffer
app.use((req, res, next) => {
  // Enable SharedArrayBuffer for multi-threading
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  
  // Additional security headers (optional but recommended)
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  
  next();
});

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'build')));

// Handle React routing - return index.html for all routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🚀 Server started successfully!');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`📡 URL: http://localhost:${PORT}`);
  console.log(`💻 CPU Cores: ${require('os').cpus().length}`);
  console.log('✅ SharedArrayBuffer: ENABLED');
  console.log('✅ Multi-threading: ENABLED');
  console.log('✅ WebGPU: ENABLED (if browser supports)');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('\n💡 Open Chrome DevTools Console to see backend info');
  console.log('🛑 Press Ctrl+C to stop the server\n');
});

