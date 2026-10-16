import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Send,
  Bot,
  User,
  Loader2,
  Plus,
  Trash2,
  GitCommit,
  Bug,
  BookOpen,
  Sparkles,
  Copy,
  Check,
  ShieldCheck,
  Cpu,
  Layers,
  Terminal,
  ExternalLink,
  Share2,
  Network,
  Lock,
  LogOut,
  LogIn,
  RefreshCw,
  Info,
} from "lucide-react";

const API_BASE = process.env.REACT_APP_API_URL
  ? process.env.REACT_APP_API_URL.replace(/\/api\/chat\/?$/, "")
  : "http://localhost:8000";

const getStarterPrompts = (repo) => {
  const isDhis = repo === "dhis2/dhis2-core";
  const label = isDhis ? "DHIS2" : "TransportPCE";
  return [
    {
      icon: GitCommit,
      title: "Recent GitHub Commits",
      description: `Summarize recent ${label} commits and code changes`,
      prompt: `What are the most recent GitHub commits about?`,
      color: "from-blue-500/20 to-cyan-500/20 text-cyan-400 border-cyan-500/30",
    },
    {
      icon: Bug,
      title: "Jira Issue Status",
      description: `Inspect recent ${label} Jira tickets and their resolutions`,
      prompt: `What recent Jira tickets have been worked on?`,
      color: "from-amber-500/20 to-orange-500/20 text-amber-400 border-amber-500/30",
    },
    {
      icon: Network,
      title: "Commit-to-Issue Links",
      description: `Find commits explicitly linked to Jira tickets via Knowledge Graph`,
      prompt: `Which GitHub commits are linked to Jira tickets in ${label}?`,
      color: "from-emerald-500/20 to-teal-500/20 text-emerald-400 border-emerald-500/30",
    },
    {
      icon: BookOpen,
      title: "API Documentation",
      description: `Query ${label} OpenAPI endpoints and specifications`,
      prompt: `What API endpoints are available in ${label}?`,
      color: "from-purple-500/20 to-indigo-500/20 text-purple-400 border-purple-500/30",
    },
  ];
};

function App() {
  const [messages, setMessages] = useState([
    {
      id: "initial",
      role: "assistant",
      content:
        "👋 **Welcome! I'm your Cross-Platform Software Engineering Knowledge Assistant.**\n\nI combine a genuine **Hybrid Graph-RAG architecture**: retrieving semantic documentation from **ChromaDB** while traversing explicit relationships across **GitHub Commits**, **Jira Tickets**, and **OpenAPI Specifications** in a Software Knowledge Graph.\n\nAsk me anything about commits, issue resolutions, or API architecture!",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      retrievalMode: "hybrid_graph_rag",
      sources: [],
      graphEntities: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [activeRepo, setActiveRepo] = useState("dhis2/dhis2-core");

  // Authentication State
  const [authToken, setAuthToken] = useState(localStorage.getItem("auth_token") || "");
  const [currentUser, setCurrentUser] = useState(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState("login"); // 'login' or 'register'
  const [authUsername, setAuthUsername] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  // Sync / Verification State
  const [syncStatus, setSyncStatus] = useState(null);
  const [syncLoading, setSyncLoading] = useState(false);

  // Graph Modal State
  const [activeGraphModal, setActiveGraphModal] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Check auth profile on mount
  useEffect(() => {
    if (authToken) {
      axios
        .get(`${API_BASE}/api/auth/me/`, {
          headers: { Authorization: `Token ${authToken}` },
        })
        .then((res) => setCurrentUser(res.data))
        .catch(() => {
          localStorage.removeItem("auth_token");
          setAuthToken("");
          setCurrentUser(null);
        });
    }
  }, [authToken]);

  // Check verified sync status for active project
  const fetchSyncStatus = async (repo) => {
    setSyncLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/api/status/?repo=${encodeURIComponent(repo)}`);
      setSyncStatus(res.data);
    } catch (e) {
      console.warn("Could not retrieve sync status", e);
      setSyncStatus(null);
    } finally {
      setSyncLoading(false);
    }
  };

  useEffect(() => {
    fetchSyncStatus(activeRepo);
  }, [activeRepo]);

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError("");
    setAuthLoading(true);

    const endpoint = authMode === "login" ? `${API_BASE}/api/auth/login/` : `${API_BASE}/api/auth/register/`;
    const payload =
      authMode === "login"
        ? { username: authUsername, password: authPassword }
        : { username: authUsername, email: authEmail, password: authPassword };

    try {
      const res = await axios.post(endpoint, payload);
      const token = res.data.token;
      localStorage.setItem("auth_token", token);
      setAuthToken(token);
      setCurrentUser(res.data.user);
      setAuthModalOpen(false);
      setAuthUsername("");
      setAuthPassword("");
      setAuthEmail("");
    } catch (err) {
      const msg = err.response?.data?.error || "Authentication failed. Please verify credentials.";
      setAuthError(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    if (authToken) {
      try {
        await axios.post(`${API_BASE}/api/auth/logout/`, {}, {
          headers: { Authorization: `Token ${authToken}` },
        });
      } catch (e) {
        // ignore
      }
    }
    localStorage.removeItem("auth_token");
    setAuthToken("");
    setCurrentUser(null);
  };

  const triggerReindex = async () => {
    if (!authToken) {
      setAuthModalOpen(true);
      return;
    }
    setSyncLoading(true);
    try {
      await axios.post(
        `${API_BASE}/api/reindex/`,
        { repo: activeRepo },
        { headers: { Authorization: `Token ${authToken}` } }
      );
      await fetchSyncStatus(activeRepo);
    } catch (e) {
      alert("Reindexing failed: " + (e.response?.data?.error || e.message));
    } finally {
      setSyncLoading(false);
    }
  };

  const sendMessage = async (customPrompt) => {
    const textToSend = customPrompt || input;
    if (!textToSend.trim() || loading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    if (!customPrompt) setInput("");
    setLoading(true);

    try {
      const headers = authToken ? { Authorization: `Token ${authToken}` } : {};
      const response = await axios.post(
        `${API_BASE}/api/chat/`,
        {
          message: textToSend,
          repo: activeRepo,
        },
        { headers }
      );

      const data = response.data;
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        retrievalMode: data.retrieval_mode || "vector_only",
        sources: data.sources || [],
        graphEntities: data.graph_entities || [],
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errDetail = error.response?.data?.error || "Could not connect to the Backend server.";
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `⚠️ **Request Failed**\n\n${errDetail}`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: "cleared",
        role: "assistant",
        content: "🧹 Chat history cleared! Ask me anything about commits, Jira tickets, or API endpoints.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
  };

  const renderFormattedContent = (content) => {
    const lines = content.split("\n");
    return lines.map((line, idx) => {
      let formattedLine = line;

      // Handle bold
      formattedLine = formattedLine.replace(/\*\*(.*?)\*\*/g, (_, text) => `__BOLD__${text}__BOLD__`);

      const parts = formattedLine.split("__BOLD__");
      const renderedParts = parts.map((part, pIdx) => {
        if (pIdx % 2 === 1) {
          return (
            <strong key={pIdx} className="font-semibold text-white">
              {part}
            </strong>
          );
        }

        // Inline code rendering
        const codeParts = part.split(/(`[^`]+`)/g);
        return codeParts.map((sub, sIdx) => {
          if (sub.startsWith("`") && sub.endsWith("`")) {
            return (
              <code
                key={sIdx}
                className="px-1.5 py-0.5 mx-0.5 rounded bg-slate-800/80 border border-slate-700/60 font-mono text-[13px] text-cyan-300"
              >
                {sub.slice(1, -1)}
              </code>
            );
          }
          return sub;
        });
      });

      if (line.trim().startsWith("•") || line.trim().startsWith("-") || line.trim().startsWith("*")) {
        return (
          <div key={idx} className="flex items-start gap-2 my-1.5 pl-2 text-slate-200">
            <span className="text-cyan-400 mt-1 text-sm">•</span>
            <div className="flex-1 leading-relaxed">{renderedParts}</div>
          </div>
        );
      }
      if (line.trim() === "") {
        return <div key={idx} className="h-2" />;
      }

      return (
        <p key={idx} className="my-1 text-slate-200 leading-relaxed">
          {renderedParts}
        </p>
      );
    });
  };

  return (
    <div className="flex h-screen w-screen bg-[#070a11] text-slate-100 overflow-hidden font-sans">
      {/* LEFT SIDEBAR */}
      <aside className="w-80 bg-[#0c121e]/90 border-r border-slate-800/80 flex flex-col justify-between backdrop-blur-xl shrink-0">
        <div>
          {/* Logo / Header */}
          <div className="p-5 border-b border-slate-800/70 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Network className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-base tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  KnowledgeAI
                </h1>
                <p className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  Hybrid Graph-RAG
                </p>
              </div>
            </div>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={clearChat}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-blue-600/90 to-indigo-600/90 hover:from-blue-600 hover:to-indigo-600 text-white font-medium text-sm transition-all shadow-md shadow-blue-600/20 hover:scale-[1.01] active:scale-[0.99]"
            >
              <Plus className="w-4 h-4" />
              <span>New Conversation</span>
            </button>
          </div>

          {/* Target Project Selector */}
          <div className="px-4 pb-2">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5 block">
              Active Repository
            </label>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-900/80 rounded-xl border border-slate-800">
              <button
                onClick={() => setActiveRepo("dhis2/dhis2-core")}
                className={`py-1.5 px-2.5 text-xs font-medium rounded-lg transition-all ${
                  activeRepo === "dhis2/dhis2-core"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                DHIS2 Core
              </button>
              <button
                onClick={() => setActiveRepo("opendaylight/transportpce")}
                className={`py-1.5 px-2.5 text-xs font-medium rounded-lg transition-all ${
                  activeRepo === "opendaylight/transportpce"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                TransportPCE
              </button>
            </div>
          </div>

          {/* Verified Connected Sources Card */}
          <div className="px-4 py-2">
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-blue-400" />
                  Verified Data Stores
                </span>
                {syncLoading && <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />}
              </div>

              {syncStatus ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-1.5 text-[11px]">
                      <GitCommit className="w-3.5 h-3.5 text-cyan-400" />
                      Git Commits
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50">
                      {syncStatus.nodes_breakdown?.commit || 0} Nodes
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-1.5 text-[11px]">
                      <Bug className="w-3.5 h-3.5 text-amber-400" />
                      Jira Issues
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/50">
                      {syncStatus.nodes_breakdown?.issue || 0} Nodes
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-1.5 text-[11px]">
                      <BookOpen className="w-3.5 h-3.5 text-purple-400" />
                      OpenAPI Spec
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800/50">
                      {syncStatus.nodes_breakdown?.endpoint || 0} Endpoints
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300 pt-1 border-t border-slate-800">
                    <span className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-medium">
                      <Network className="w-3.5 h-3.5" />
                      Knowledge Graph
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/50">
                      {syncStatus.graph_nodes_count}N / {syncStatus.graph_edges_count}E
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-[11px] text-amber-400 flex items-center gap-1">
                  <Info className="w-3 h-3" />
                  <span>Checking store verification...</span>
                </div>
              )}

              {/* Re-index button */}
              <button
                onClick={triggerReindex}
                disabled={syncLoading}
                className="mt-3 w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-slate-800 text-[11px] text-slate-300 border border-slate-700/60 transition-colors"
              >
                <RefreshCw className={`w-3 h-3 ${syncLoading ? "animate-spin text-cyan-400" : ""}`} />
                <span>{currentUser ? "Re-index Project Data" : "Sign In to Re-index"}</span>
              </button>
            </div>
          </div>
        </div>

        {/* User Profile / Auth Footer */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-900/40">
          {currentUser ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5 overflow-hidden">
                <div className="w-8 h-8 rounded-full bg-blue-600/30 border border-blue-500/50 flex items-center justify-center text-xs font-bold text-blue-300">
                  {currentUser.username[0].toUpperCase()}
                </div>
                <div className="overflow-hidden">
                  <p className="text-xs font-medium text-white truncate">{currentUser.username}</p>
                  <p className="text-[10px] text-slate-400 truncate">{currentUser.email || "Authenticated Dev"}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                title="Log Out"
                className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-950/30 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setAuthError("");
                setAuthModalOpen(true);
              }}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-slate-800/90 hover:bg-slate-800 text-xs font-medium text-slate-200 border border-slate-700/70 transition-colors"
            >
              <LogIn className="w-3.5 h-3.5 text-cyan-400" />
              <span>Sign In / Register</span>
            </button>
          )}
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col justify-between overflow-hidden relative">
        {/* TOP BAR */}
        <header className="h-14 border-b border-slate-800/80 bg-[#0c121e]/40 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Project:</span>
            <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-blue-950 text-blue-300 border border-blue-800/60">
              {activeRepo}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-full border border-emerald-800/40">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              Hybrid Graph-RAG Active
            </div>
          </div>
        </header>

        {/* CHAT MESSAGES CONTAINER */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 max-w-3xl ${msg.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"}`}
            >
              {/* Avatar */}
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-white ${
                  msg.role === "user"
                    ? "bg-gradient-to-tr from-cyan-600 to-blue-600 shadow-md shadow-blue-500/20"
                    : msg.isError
                    ? "bg-rose-600"
                    : "bg-gradient-to-tr from-indigo-600 to-blue-600 shadow-md shadow-indigo-500/20"
                }`}
              >
                {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              {/* Message Bubble */}
              <div
                className={`flex-1 rounded-2xl px-5 py-4 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/10 rounded-tr-none"
                    : msg.isError
                    ? "bg-rose-950/40 border border-rose-800/60 text-rose-200 rounded-tl-none"
                    : "bg-[#0c121e]/80 border border-slate-800/90 text-slate-100 rounded-tl-none backdrop-blur-md"
                }`}
              >
                {/* Mode & Action Bar for Assistant */}
                {msg.role === "assistant" && !msg.isError && (
                  <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800/70 text-xs">
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border flex items-center gap-1 ${
                          msg.retrievalMode === "hybrid_graph_rag"
                            ? "bg-emerald-950 text-emerald-300 border-emerald-800/60"
                            : "bg-amber-950 text-amber-300 border-amber-800/60"
                        }`}
                      >
                        {msg.retrievalMode === "hybrid_graph_rag" ? (
                          <>
                            <Network className="w-2.5 h-2.5" /> Hybrid Graph-RAG
                          </>
                        ) : (
                          "Vector-Only"
                        )}
                      </span>
                    </div>

                    <button
                      onClick={() => handleCopy(msg.id, msg.content)}
                      className="text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1 text-[11px]"
                    >
                      {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copiedId === msg.id ? "Copied" : "Copy"}</span>
                    </button>
                  </div>
                )}

                {/* Formatted Text Content */}
                <div className="text-sm">{renderFormattedContent(msg.content)}</div>

                {/* Verified Clickable Sources Shelf */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-800/70">
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
                      <Layers className="w-3 h-3 text-cyan-400" />
                      Verified Supporting Evidence ({msg.sources.length})
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((src, sIdx) => {
                        const isGithub = src.source_type === "github";
                        const isJira = src.source_type === "jira";
                        return (
                          <a
                            key={sIdx}
                            href={src.url || "#"}
                            target={src.url ? "_blank" : undefined}
                            rel="noopener noreferrer"
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs border transition-all ${
                              isGithub
                                ? "bg-cyan-950/40 text-cyan-300 border-cyan-800/60 hover:bg-cyan-900/60"
                                : isJira
                                ? "bg-amber-950/40 text-amber-300 border-amber-800/60 hover:bg-amber-900/60"
                                : "bg-purple-950/40 text-purple-300 border-purple-800/60 hover:bg-purple-900/60"
                            }`}
                          >
                            {isGithub && <GitCommit className="w-3 h-3" />}
                            {isJira && <Bug className="w-3 h-3" />}
                            {!isGithub && !isJira && <BookOpen className="w-3 h-3" />}
                            <span className="font-mono">{src.source_id}</span>
                            {src.url && <ExternalLink className="w-2.5 h-2.5 opacity-60" />}
                          </a>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Connected Graph Entities Drawer Button */}
                {msg.graphEntities && msg.graphEntities.length > 0 && (
                  <div className="mt-3">
                    <button
                      onClick={() => setActiveGraphModal(msg.graphEntities)}
                      className="text-[11px] font-medium text-emerald-400 hover:text-emerald-300 flex items-center gap-1 bg-emerald-950/40 border border-emerald-800/50 px-2.5 py-1 rounded-lg hover:bg-emerald-900/50 transition-colors"
                    >
                      <Network className="w-3 h-3" />
                      <span>View Knowledge Graph Traversal ({msg.graphEntities.length} Entities)</span>
                    </button>
                  </div>
                )}

                {/* Timestamp */}
                <div className="mt-2 text-[10px] text-slate-400/80 text-right">{msg.timestamp}</div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-3 max-w-3xl mr-auto">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-blue-600 flex items-center justify-center text-white">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-[#0c121e]/80 border border-slate-800/90 rounded-2xl rounded-tl-none px-5 py-4 flex items-center gap-3 text-slate-300 text-sm">
                <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                <span>Traversing Knowledge Graph and synthesizing grounded answer...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* STARTER PROMPTS CAROUSEL */}
        {messages.length <= 2 && (
          <div className="px-6 pb-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">Suggested Inquiries</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              {getStarterPrompts(activeRepo).map((card, idx) => {
                const Icon = card.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => sendMessage(card.prompt)}
                    className="p-3 rounded-xl bg-[#0c121e]/70 border border-slate-800/80 hover:border-slate-700 text-left transition-all hover:scale-[1.01] group"
                  >
                    <Icon className="w-4 h-4 text-cyan-400 mb-1.5" />
                    <p className="text-xs font-semibold text-slate-200 group-hover:text-white">{card.title}</p>
                    <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">{card.description}</p>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* INPUT BOX */}
        <div className="p-4 border-t border-slate-800/80 bg-[#0c121e]/60 backdrop-blur-xl shrink-0">
          <div className="max-w-4xl mx-auto flex items-end gap-2 bg-slate-900/90 border border-slate-800 focus-within:border-blue-500/80 rounded-2xl p-2 transition-all shadow-xl shadow-black/40">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={`Ask a question across ${activeRepo} commits, Jira issues, and API endpoints...`}
              rows={1}
              className="flex-1 bg-transparent px-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-32 min-h-[38px]"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className="p-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-40 text-white transition-all shadow-md shadow-blue-600/30 shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </main>

      {/* KNOWLEDGE GRAPH ENTITIES MODAL */}
      {activeGraphModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0c121e] border border-slate-800 rounded-2xl max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Network className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-sm text-white">Connected Knowledge Graph Subgraph</h3>
              </div>
              <button
                onClick={() => setActiveGraphModal(null)}
                className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded"
              >
                ✕ Close
              </button>
            </div>
            <div className="p-4 overflow-y-auto space-y-2">
              {activeGraphModal.map((entity, eIdx) => (
                <div key={eIdx} className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-cyan-300">{entity.id}</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase bg-slate-800 text-slate-300">
                      {entity.type}
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs mt-1">{entity.label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* AUTHENTICATION MODAL */}
      {authModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0c121e] border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Lock className="w-5 h-5 text-blue-400" />
                <h3 className="font-bold text-base text-white">
                  {authMode === "login" ? "Developer Sign In" : "Register Account"}
                </h3>
              </div>
              <button
                onClick={() => setAuthModalOpen(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            {authError && (
              <div className="mt-3 p-2.5 rounded-lg bg-rose-950/50 border border-rose-800 text-rose-300 text-xs">
                {authError}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="mt-4 space-y-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Username</label>
                <input
                  type="text"
                  required
                  value={authUsername}
                  onChange={(e) => setAuthUsername(e.target.value)}
                  placeholder="e.g. dev_analyst"
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              {authMode === "register" && (
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Email Address</label>
                  <input
                    type="email"
                    required
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    placeholder="dev@example.com"
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
              )}

              <div>
                <label className="text-xs text-slate-400 block mb-1">Password</label>
                <input
                  type="password"
                  required
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full py-2.5 mt-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-all shadow-md shadow-blue-600/30 flex items-center justify-center gap-2"
              >
                {authLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                <span>{authMode === "login" ? "Sign In" : "Create Account"}</span>
              </button>
            </form>

            <div className="mt-4 pt-3 border-t border-slate-800 text-center text-xs text-slate-400">
              {authMode === "login" ? (
                <span>
                  Don't have an account?{" "}
                  <button
                    onClick={() => {
                      setAuthMode("register");
                      setAuthError("");
                    }}
                    className="text-blue-400 hover:underline"
                  >
                    Register here
                  </button>
                </span>
              ) : (
                <span>
                  Already registered?{" "}
                  <button
                    onClick={() => {
                      setAuthMode("login");
                      setAuthError("");
                    }}
                    className="text-blue-400 hover:underline"
                  >
                    Sign in here
                  </button>
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
