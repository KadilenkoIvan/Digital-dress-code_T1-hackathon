/**
 * Development proxy configuration for Create React App
 * Adds necessary HTTP headers for SharedArrayBuffer support in development mode
 * 
 * This file is automatically picked up by react-scripts (webpack dev server)
 * when you run 'npm start'
 * 
 * NOTE: After changing this file, you MUST restart the dev server!
 */

module.exports = function(app) {
  app.use((req, res, next) => {
    // Enable SharedArrayBuffer for multi-threading
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    
    // Log headers (only once on startup)
    if (!global.headersLogged) {
      console.log('✅ HTTP Headers configured for SharedArrayBuffer:');
      console.log('   - Cross-Origin-Embedder-Policy: require-corp');
      console.log('   - Cross-Origin-Opener-Policy: same-origin');
      global.headersLogged = true;
    }
    
    next();
  });
};
