import { useState } from 'react'
import { Send, User, Bot, AlertTriangle, ShieldCheck, Cpu } from 'lucide-react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'customer', text: 'My order #12345 hasn\'t arrived yet. What is going on??' }
  ])
  const [draft, setDraft] = useState(null)
  const [loading, setLoading] = useState(false)
  const [input, setInput] = useState('')

  const handleSimulateReceive = async () => {
    if (!input.trim()) return;
    const userMsg = { id: Date.now(), sender: 'customer', text: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    // Simulate fetching draft from ML backend
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      setDraft(data);
    } catch (err) {
      console.error("Backend error:", err);
      setDraft({
        reply: "Error connecting to backend.",
        intent: "Error",
        escalated: true,
        retrieved_evidence: 0
      });
    } finally {
      setLoading(false);
    }
  }

  const handleApprove = () => {
    if (draft) {
      setMessages(prev => [...prev, { id: Date.now(), sender: 'agent', text: draft.reply }])
      setDraft(null)
    }
  }

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <Cpu size={28} />
          <h2>Hiver AI</h2>
        </div>
        <div className="menu-item active">
          Inbox
          <span className="badge">1</span>
        </div>
        <div className="menu-item">
          Resolved
        </div>
        <div className="menu-item">
          Settings
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-content">
        <header className="header">
          <div className="header-info">
            <h1>Active Ticket: #8492</h1>
            <span className="status-pill open">Open</span>
          </div>
          <div className="agent-profile">
            <ShieldCheck size={20} /> AI Copilot Active
          </div>
        </header>

        <div className="chat-area">
          {messages.map(msg => (
            <div key={msg.id} className={`message-bubble ${msg.sender}`}>
              <div className="avatar">
                {msg.sender === 'customer' ? <User size={20} /> : <Bot size={20} />}
              </div>
              <div className="text-content">
                {msg.text}
              </div>
            </div>
          ))}

          {loading && (
            <div className="loading-indicator">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="text">AI Copilot is drafting a response...</span>
            </div>
          )}

          {draft && !loading && (
            <div className="draft-panel">
              <div className="draft-header">
                <h3>Drafted Response</h3>
                <div className="draft-meta">
                  <span className="meta-tag intent">{draft.intent}</span>
                  <span className="meta-tag evidence">{draft.retrieved_evidence} cases retrieved</span>
                  {draft.escalated && (
                    <span className="meta-tag escalate">
                      <AlertTriangle size={14} /> Escalation Recommended
                    </span>
                  )}
                </div>
              </div>
              <p className="draft-text">{draft.reply}</p>
              <div className="draft-actions">
                <button className="btn-approve" onClick={handleApprove}>Approve & Send</button>
                <button className="btn-discard" onClick={() => setDraft(null)}>Discard</button>
              </div>
            </div>
          )}
        </div>

        {/* Input Bar (For Simulation) */}
        <div className="input-area">
          <div className="input-container">
            <input
              type="text"
              placeholder="Simulate an incoming tweet or customer message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSimulateReceive()}
            />
            <button className="btn-send" onClick={handleSimulateReceive}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
