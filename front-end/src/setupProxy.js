/**
 * CRA dev server: proxy /api to Flask. Uses http-proxy-middleware v0.x API
 * (react-scripts 4 bundles 0.19.x — no createProxyMiddleware).
 */
const proxy = require('http-proxy-middleware');

module.exports = function setupProxy(app) {
    app.use(
        '/api',
        proxy({
            target: 'http://127.0.0.1:8765',
            changeOrigin: true,
        }),
    );
};
