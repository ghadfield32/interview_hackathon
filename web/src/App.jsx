import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import MLModelFrontend from './components/MLModelFrontend';
import Login from './components/Login';
import Layout from './components/Layout';
import { apiService } from './services/api';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [backendReady, setBackendReady] = useState(false);

  // unified poller ref so we can cancel / avoid duplicates
  const pollerRef = useRef(null);
  const destroyedRef = useRef(false);

  // --- DEBUG DIAGNOSTICS (safe to remove later) -----------------
  if (import.meta.env.DEV) {
    // This will run every render – fine for dev introspection
    // eslint-disable-next-line no-console
    console.debug('[App:render]', {
      path: window.location.pathname,
      isAuthenticated,
      isLoading,
      backendReady
    });
  }

  useEffect(() => {
    destroyedRef.current = false;
    (async () => {
      await checkAuthStatus();
      startUnifiedReadinessPoll();
    })();

    return () => {
      destroyedRef.current = true;
      if (pollerRef.current) {
        clearTimeout(pollerRef.current);
        pollerRef.current = null;
      }
    };
    // empty dep array – intentional; StrictMode double-mount safe because cleanup runs
  }, []);

  const startUnifiedReadinessPoll = async (attempt = 0) => {
    if (destroyedRef.current) return;
    try {
      const res = await apiService.getReadyFull(); // use full – includes model_status
      // eslint-disable-next-line no-console
      console.debug('[readiness] attempt', attempt, res);
      if (res?.ready) {
        setBackendReady(true);
        return; // stop polling
      }
    } catch (err) {
      console.error('[readiness] poll error', err);
    }
    const delay = Math.min(1500 * 2 ** attempt, 8000);
    pollerRef.current = setTimeout(() => startUnifiedReadinessPoll(attempt + 1), delay);
  };

  const checkAuthStatus = async () => {
    const token = localStorage.getItem('jwt');
    if (!token) {
      setIsLoading(false);
      return;
    }
    try {
      await apiService.getHealth(); // lightweight probe
      setUser({ username: 'authenticated' });
      setIsAuthenticated(true);
    } catch (err) {
      console.warn('[auth] stored token invalid – clearing', err);
      localStorage.removeItem('jwt');
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (credentials) => {
    try {
      const response = await apiService.login(credentials);
      localStorage.setItem('jwt', response.access_token);
      setUser({ username: credentials.username });
      setIsAuthenticated(true);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('jwt');
    setUser(null);
    setIsAuthenticated(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        {/* fallback size so spinner always visible */}
        <div className="spinner" style={{ width: 40, height: 40 }} />
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: { background: '#363636', color: '#fff' }
          }}
        />
        <Routes>
          <Route
            path="/login"
            element={
              isAuthenticated
                ? <Navigate to="/" replace />
                : <Login onLogin={handleLogin} backendReady={backendReady} />
            }
          />
          <Route
            path="/"
            element={
              isAuthenticated
                ? (
                  <Layout user={user} onLogout={handleLogout}>
                    <MLModelFrontend backendReady={backendReady} />
                  </Layout>
                )
                : <Navigate to="/login" replace />
            }
          />
          <Route
            path="/dashboard"
            element={
              isAuthenticated
                ? (
                  <Layout user={user} onLogout={handleLogout}>
                    <MLModelFrontend backendReady={backendReady} />
                  </Layout>
                )
                : <Navigate to="/login" replace />
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 





