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
} from "lucide-react";

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
      icon: ShieldCheck,
      title: "Bug Fixes & Features",
      description: `Explain recent bug fixes, features, and CI updates in ${label}`,
      prompt: `Explain recent bug fixes, features, and improvements in ${label}.`,
      color: "from-emerald-500/20 to-teal-500/20 text-emerald-400 border-emerald-500/30",
    },
    {
      icon: BookOpen,
      title: "API Documentation",
      description: `Query ${label} API specifications and endpoints`,
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
        "👋 **Welcome! I'm your AI Software Engineering Knowledge Assistant for DHIS2 Core Platform.**\n\nI combine **GitHub commits**, **Jira tickets**, and **API documentation** through a **Retrieval-Augmented Generation (RAG)** pipeline. Ask me anything about DHIS2 code, bug fixes, or architecture!",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [activeRepo, setActiveRepo] = useState("dhis2/dhis2-core");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
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
      const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000/api/chat/";
      const response = await axios.post(apiUrl, {
        message: textToSend,
        repo: activeRepo,
      });

      const aiMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "⚠️ **Could not connect to the Backend server.**\n\nPlease ensure your Django server is running on `http://localhost:8000` with `python manage.py runserver`.",
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
        content:
          "✨ Chat history cleared! Ask me anything about your project's GitHub commits, Jira issues, or API endpoints.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
  };

  // Helper to format markdown-like text into styled elements
  const renderFormattedText = (text) => {
    const lines = text.split("\n");
    return lines.map((line, idx) => {
      // Bold syntax handling
      const formattedLine = line.split(/(\*\*.*?\*\*)/g).map((chunk, cIdx) => {
        if (chunk.startsWith("**") && chunk.endsWith("**")) {
          return (
            <strong key={cIdx} className="font-semibold text-white">
              {chunk.slice(2, -2)}
            </strong>
          );
        }
        // Inline code formatting
        return chunk.split(/(`.*?`)/g).map((subChunk, sIdx) => {
          if (subChunk.startsWith("`") && subChunk.endsWith("`")) {
            return (
              <code
                key={sIdx}
                className="bg-slate-800 text-blue-300 px-1.5 py-0.5 rounded text-xs font-mono border border-slate-700/60"
              >
                {subChunk.slice(1, -1)}
              </code>
            );
          }
          return subChunk;
        });
      });

      if (line.trim().startsWith("- ") || line.trim().startsWith("• ")) {
        return (
          <li key={idx} className="ml-4 list-disc text-slate-300 my-1">
            {formattedLine}
          </li>
        );
      }
      if (/^\d+\.\s/.test(line.trim())) {
        return (
          <li key={idx} className="ml-4 list-decimal text-slate-300 my-1">
            {formattedLine}
          </li>
        );
      }
      if (line.trim() === "") {
        return <div key={idx} className="h-2" />;
      }

      return (
        <p key={idx} className="my-1 text-slate-200 leading-relaxed">
          {formattedLine}
        </p>
      );
    });
  };

  return (
    <div className="flex h-screen w-screen bg-[#070a11] text-slate-100 overflow-hidden font-sans">
      {/* LEFT SIDEBAR */}
      <aside className="w-72 bg-[#0c121e]/90 border-r border-slate-800/80 flex flex-col justify-between backdrop-blur-xl shrink-0">
        <div>
          {/* Logo / Header */}
          <div className="p-5 border-b border-slate-800/70 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-base tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  KnowledgeAI
                </h1>
                <p className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  RAG Assistant v1.0
                </p>
              </div>
            </div>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={clearChat}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-blue-600/90 to-indigo-600/90 hover:from-blue-600 hover:to-indigo-600 text-white font-medium text-sm transition-all shadow-md shadow-blue-600/20 hover:shadow-blue-600/30 hover:scale-[1.01] active:scale-[0.99]"
            >
              <Plus className="w-4 h-4" />
              <span>New Conversation</span>
            </button>
          </div>

          {/* Active Data Sources Card */}
          <div className="px-4 py-2">
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-blue-400" />
                Connected Sources
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-slate-300">
                  <span className="flex items-center gap-1.5 text-[11px]">
                    <GitCommit className="w-3.5 h-3.5 text-cyan-400" />
                    GitHub Commits
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50">
                    Synced
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span className="flex items-center gap-1.5 text-[11px]">
                    <Bug className="w-3.5 h-3.5 text-amber-400" />
                    Jira Issues
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/50">
                    Connected
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span className="flex items-center gap-1.5 text-[11px]">
                    <BookOpen className="w-3.5 h-3.5 text-purple-400" />
                    Swagger Docs
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800/50">
                    Indexed
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Repository Selector */}
          <div className="px-4 py-3">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
              Target Project
            </label>
            <select
              value={activeRepo}
              onChange={(e) => {
                const newRepo = e.target.value;
                setActiveRepo(newRepo);
                const repoLabel = newRepo === "dhis2/dhis2-core" ? "DHIS2 Core Platform" : "TransportPCE (OpenDaylight)";
                setMessages((prev) => [
                  ...prev,
                  {
                    id: Date.now().toString(),
                    role: "assistant",
                    content: `🔄 **Active project switched to ${repoLabel} (${newRepo}).**\n\nAll subsequent questions will query ${repoLabel}'s GitHub commits, Jira issues, and API endpoints.`,
                    timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                  },
                ]);
              }}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 outline-none focus:border-blue-500 cursor-pointer"
            >
              <option value="dhis2/dhis2-core">DHIS2 Core Platform</option>
              <option value="opendaylight/transportpce">TransportPCE (OpenDaylight)</option>
            </select>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-800/70 bg-slate-950/40">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="flex items-center gap-1.5 font-medium">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              Groq + ChromaDB
            </span>
            <span className="text-[11px] text-slate-500">Team 09</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col h-full bg-[#080c14] relative">
        {/* Background glow effects */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none -z-10"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none -z-10"></div>

        {/* Top Navbar */}
        <header className="h-16 border-b border-slate-800/70 px-8 flex items-center justify-between backdrop-blur-md bg-[#080c14]/70 z-10">
          <div className="flex items-center gap-3">
            <h2 className="font-semibold text-sm text-slate-200">
              Project Knowledge Workspace
            </h2>
            <span className="text-slate-600">•</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800/80 text-slate-400 border border-slate-700/50 font-mono">
              {activeRepo}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40 text-[11px] text-emerald-300">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              Backend Active (:8000)
            </div>

            <button
              onClick={clearChat}
              title="Clear conversation"
              className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800/60 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <div className="flex-1 overflow-y-auto px-6 md:px-16 py-8 space-y-6">
          {/* Welcome Banner & Quick Prompts if only 1 message */}
          {messages.length === 1 && (
            <div className="my-6 max-w-3xl mx-auto">
              <div className="text-center mb-8">
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600/30 to-indigo-600/30 border border-blue-500/30 shadow-xl shadow-blue-500/10 mb-4">
                  <Sparkles className="w-7 h-7 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">
                  What would you like to know about this repository?
                </h3>
                <p className="text-sm text-slate-400 max-w-md mx-auto">
                  Ask natural questions across code changes, commit histories, issue trackers, and API docs.
                </p>
              </div>

              {/* Quick Starter Prompt Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {getStarterPrompts(activeRepo).map((item, idx) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => sendMessage(item.prompt)}
                      className={`text-left p-4 rounded-xl border bg-gradient-to-br transition-all hover:scale-[1.02] active:scale-[0.99] group ${item.color}`}
                    >
                      <div className="flex items-center gap-2.5 font-semibold text-sm mb-1 text-slate-100">
                        <Icon className="w-4 h-4 shrink-0" />
                        <span>{item.title}</span>
                      </div>
                      <p className="text-xs text-slate-400 group-hover:text-slate-300">
                        {item.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Message List */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3.5 max-w-3xl mx-auto ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-md shadow-blue-500/20 mt-1">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              )}

              <div className="group relative max-w-[85%]">
                <div
                  className={`px-5 py-3.5 rounded-2xl text-sm ${
                    msg.role === "user"
                      ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-sm shadow-lg shadow-blue-600/15"
                      : msg.isError
                      ? "bg-rose-950/40 border border-rose-800/60 text-rose-200 rounded-bl-sm"
                      : "bg-[#111726]/90 border border-slate-800/80 text-slate-200 rounded-bl-sm shadow-xl backdrop-blur-md"
                  }`}
                >
                  <div className="prose-custom">{renderFormattedText(msg.content)}</div>
                </div>

                {/* Footer metadata & copy action */}
                <div
                  className={`flex items-center gap-2 mt-1.5 px-1 text-[11px] text-slate-500 ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <span>{msg.timestamp}</span>
                  {msg.role === "assistant" && !msg.isError && (
                    <>
                      <span>•</span>
                      <button
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="hover:text-slate-300 flex items-center gap-1 transition-colors"
                      >
                        {copiedId === msg.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-400" />
                            <span className="text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </>
                  )}
                </div>
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 mt-1">
                  <User className="w-4 h-4 text-slate-300" />
                </div>
              )}
            </div>
          ))}

          {/* Thinking / Loading State */}
          {loading && (
            <div className="flex gap-3.5 max-w-3xl mx-auto">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-md shadow-blue-500/20">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="px-5 py-3.5 rounded-2xl rounded-bl-sm bg-[#111726]/90 border border-slate-800/80 text-sm flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin text-blue-400 shrink-0" />
                <span className="text-xs text-slate-400">
                  Searching ChromaDB vector store & synthesizing answer with Groq LLM...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Floating Input Area */}
        <div className="p-6 max-w-3xl w-full mx-auto">
          <div className="relative rounded-2xl bg-[#0f1626]/90 border border-slate-800/90 shadow-2xl backdrop-blur-xl focus-within:border-blue-500/70 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all p-2 flex items-end gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask a question about commits, Jira tickets, or API endpoints..."
              rows={1}
              disabled={loading}
              className="flex-1 bg-transparent px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none resize-none max-h-32 min-h-[42px]"
            />

            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="p-2.5 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white disabled:opacity-30 disabled:hover:from-blue-600 disabled:hover:to-indigo-600 transition-all shrink-0 shadow-md shadow-blue-600/25"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>

          <p className="text-[11px] text-slate-500 text-center mt-2 flex items-center justify-center gap-2">
            <span>Press <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono text-[10px]">Enter ↵</kbd> to send</span>
            <span>•</span>
            <span>Knowledge Assistant with LangChain & Groq</span>
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
