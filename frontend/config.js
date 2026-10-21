// Backend base URL. For production on GitHub Pages, set this to your deployed
// API origin (see docs/DEPLOY.md), e.g. "https://jianpu-api.onrender.com".
// When opening locally with `python -m http.server`, the default localhost:8000
// is used. A ?api=https://... query string overrides it at runtime.
window.JIANPU_CONFIG = {
  apiBase:
    new URLSearchParams(location.search).get("api") ||
    (location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:8000"
      : "https://REPLACE-WITH-YOUR-BACKEND"),
};
