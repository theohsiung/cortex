import { useState, useEffect, useRef } from 'react';
import './App.css';

interface Step {
  id: number;
  description: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'blocked';
  notes?: string;
  tools: string[];
  output?: string;
  error?: string;
}



interface Log {
  timestamp: string;
  message: string;
  type: 'info' | 'error' | 'warning' | 'success';
}

function App() {
  const [query, setQuery] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle');
  const [steps, setSteps] = useState<Step[]>([]);
  const [result, setResult] = useState<string | null>(null);
  const [logs, setLogs] = useState<Log[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (message: string, type: Log['type'] = 'info') => {
    setLogs(prev => [...prev, {
      timestamp: new Date().toLocaleTimeString(),
      message,
      type
    }]);
  };

  const startTask = async () => {
    if (!query.trim()) return;

    setStatus('running');
    setSteps([]);
    setResult(null);
    setLogs([]);
    addLog(`Starting task: ${query}`, 'info');

    try {
      const res = await fetch('http://localhost:8999/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!res.ok) throw new Error('Failed to start task');

      const data = await res.json();
      setTaskId(data.task_id);
      addLog(`Task created, ID: ${data.task_id}`, 'success');
    } catch (err) {
      console.error(err);
      setStatus('error');
      addLog(`Error starting task: ${err}`, 'error');
    }
  };

  // SSE Connection
  useEffect(() => {
    if (!taskId) return;

    addLog('Connecting to event stream...', 'info');
    const evtSource = new EventSource(`http://localhost:8999/api/tasks/${taskId}/events`);

    evtSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        const { event, data } = payload;

        console.log("Event:", event, data);

        switch (event) {
          case 'connected':
            addLog('Stream connected', 'success');
            break;

          case 'plan_created':
            addLog(`Plan created: ${data.plan_id}`, 'info');
            break;

          case 'plan_updated':
            // Initialize steps
            setSteps(data.steps.map((desc: string, i: number) => ({
              id: i,
              description: desc,
              status: 'not_started',
              tools: []
            })));
            addLog('Plan received/updated', 'info');
            break;

          case 'step_status':
            setSteps(prev => prev.map(s => {
              if (s.id === data.step_idx) {
                const newStatus = data.status;
                return {
                  ...s,
                  status: newStatus,
                  output: data.output || s.output,
                  error: data.error || s.error
                };
              }
              return s;
            }));
            if (data.status === 'completed') {
              addLog(`Step ${data.step_idx} completed`, 'success');
            } else if (data.status === 'blocked') {
              addLog(`Step ${data.step_idx} blocked: ${data.error}`, 'error');
            } else {
              addLog(`Step ${data.step_idx} started`, 'info');
            }
            break;

          case 'replanning':
            addLog(`Replanning triggered at step ${data.step_idx} (attempt ${data.attempt})`, 'warning');
            break;

          case 'execution_complete':
            setResult(data.result);
            setStatus('completed');
            addLog('Execution completed successfully', 'success');
            evtSource.close();
            break;

          case 'error':
            addLog(`Error: ${data.message}`, 'error');
            setStatus('error');
            evtSource.close();
            break;
        }
      } catch (err) {
        console.error('Error parsing event', err);
      }
    };

    evtSource.onerror = (err) => {
      console.error('EventSource failed', err);
      // evtSource.close(); // Don't close immediately, let it retry or handle gracefully
    };

    return () => {
      evtSource.close();
    };
  }, [taskId]);

  return (
    <div className="container">
      <header>
        <h1>Cortex Microservice</h1>
      </header>

      <div className="input-section">
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Enter your task here..."
          disabled={status === 'running'}
        />
        <button onClick={startTask} disabled={status === 'running' || !query}>
          {status === 'running' ? 'Running...' : 'Execute Task'}
        </button>
      </div>

      <div className="main-content">
        <div className="left-panel">
          <h2>Plan Execution</h2>
          {steps.length === 0 ? (
            <p className="placeholder">No plan generated yet.</p>
          ) : (
            <div className="steps-list">
              {steps.map(step => (
                <div key={step.id} className={`step-card ${step.status}`}>
                  <div className="step-header">
                    <span className="step-id">#{step.id}</span>
                    <span className="step-status-icon">
                      {step.status === 'completed' && '✓'}
                      {step.status === 'in_progress' && '⟳'}
                      {step.status === 'blocked' && '✗'}
                      {step.status === 'not_started' && '○'}
                    </span>
                  </div>
                  <div className="step-content">
                    <p className="step-desc">{step.description}</p>
                    {step.output && (
                      <details>
                        <summary>Result</summary>
                        <pre>{step.output}</pre>
                      </details>
                    )}
                    {step.error && <p className="error-msg">{step.error}</p>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="right-panel">
          <div className="logs-panel">
            <h2>Logs</h2>
            <div className="logs-container">
              {logs.map((log, i) => (
                <div key={i} className={`log-entry ${log.type}`}>
                  <span className="timestamp">[{log.timestamp}]</span>
                  <span className="message">{log.message}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          <div className="result-panel">
            <h2>Final Result</h2>
            {result ? (
              <div className="result-content markdown-body">
                <pre>{result}</pre>
              </div>
            ) : (
              <p className="placeholder">Result will appear here...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
