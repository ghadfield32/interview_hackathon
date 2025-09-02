export function installDebug() {
  if (window.__debugInstalled) return;
  window.__debugInstalled = true;

  // Track interval creation
  let intervalCounter = 0;
  const intervalTracker = new Map();

  window.addEventListener('error', (e) => {
    console.error('[GlobalError]', e.error || e.message);
    showOverlay(e.message);
  });

  window.addEventListener('unhandledrejection', (e) => {
    console.error('[GlobalRejection]', e.reason);
    showOverlay('Unhandled Promise: ' + (e.reason?.message || e.reason));
  });

  // Override setInterval to track creation
  const origSetInterval = window.setInterval;
  window.setInterval = function(fn, ms, ...rest) {
    const id = origSetInterval(fn, ms, ...rest);
    intervalCounter++;
    const stack = new Error().stack.split('\n').slice(1, 4).join('\n');
    console.log('[IntervalCreated]', { 
      id, 
      ms, 
      counter: intervalCounter,
      stack 
    });
    intervalTracker.set(id, { fn: fn.toString(), ms, created: Date.now() });
    return id;
  };

  // Override clearInterval to track cleanup
  const origClearInterval = window.clearInterval;
  window.clearInterval = function(id) {
    origClearInterval(id);
    if (intervalTracker.has(id)) {
      console.log('[IntervalCleared]', { 
        id, 
        duration: Date.now() - intervalTracker.get(id).created 
      });
      intervalTracker.delete(id);
    }
  };

  // Add visual debug info
  function showOverlay(msg) {
    let el = document.getElementById('__debug_overlay');
    if (!el) {
      el = document.createElement('div');
      el.id = '__debug_overlay';
      el.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#b91c1c;color:#fff;padding:8px;z-index:99999;font:12px/1.4 monospace;';
      document.body.appendChild(el);
    }
    el.textContent = '[ERROR] ' + msg;
  }

  // Log initial state
  console.log('[Debug] Global debug instrumentation installed');
} 
