import React, { useState, useEffect, useCallback } from 'react';
import { Search, Plus, Code, FileText, Globe, Database, Star, Tag, Download, Trash2, Grid, List, BarChart3, AlertCircle } from 'lucide-react';

const ArtifactManager = () => {
  const [artifactList, setArtifactList] = useState([]);
  const [projectList, setProjectList] = useState(['default']);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState('all');
  const [selectedProject, setSelectedProject] = useState('all');
  const [viewMode, setViewMode] = useState('grid');
  const [selectedArtifact, setSelectedArtifact] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiConnected, setApiConnected] = useState(false);

  const API_BASE_URL = 'http://localhost:5000/api';

  const apiCall = useCallback(async (endpoint, options = {}) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }, []);

  const checkApiConnection = useCallback(async () => {
    try {
      await apiCall('/health');
      setApiConnected(true);
      setError(null);
      return true;
    } catch {
      setApiConnected(false);
      setError('Cannot connect to backend API. Make sure the Python server is running on port 5000.');
      return false;
    }
  }, [apiCall]);

  const loadArtifacts = useCallback(async () => {
    if (!apiConnected) return [];

    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (searchQuery) params.append('query', searchQuery);
      if (selectedType !== 'all') params.append('type', selectedType);
      if (selectedProject !== 'all') params.append('project', selectedProject);

      const data = await apiCall(`/artifacts?${params.toString()}`);
      const loadedArtifacts = data.artifacts || [];
      setArtifactList(loadedArtifacts);
      setError(null);
      return loadedArtifacts;
    } catch (err) {
      setError(`Failed to load artifacts: ${err.message}`);
      setArtifactList([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, [searchQuery, selectedType, selectedProject, apiCall, apiConnected]);

  const loadProjects = useCallback(async () => {
    if (!apiConnected) return ['all', 'default'];

    try {
      const data = await apiCall('/projects');
      const projects = ['all', ...(data.projects || ['default'])];
      setProjectList(projects);
      return projects;
    } catch (err) {
      console.error('Failed to load projects:', err);
      setProjectList(['all', 'default']);
      return ['all', 'default'];
    }
  }, [apiCall, apiConnected]);

  const loadStatistics = useCallback(async () => {
    if (!apiConnected) return {};

    try {
      const data = await apiCall('/stats');
      setStats(data);
      return data;
    } catch (err) {
      console.error('Failed to load statistics:', err);
      setStats({});
      return {};
    }
  }, [apiCall, apiConnected]);

  const loadArtifactDetails = useCallback(async (artifactId) => {
    try {
      const data = await apiCall(`/artifacts/${artifactId}`);
      setSelectedArtifact(data);
      return data;
    } catch (err) {
      setError(`Failed to load artifact details: ${err.message}`);
      return null;
    }
  }, [apiCall]);

  const createArtifact = useCallback(async (artifactData) => {
    const data = await apiCall('/artifacts', {
      method: 'POST',
      body: JSON.stringify(artifactData),
    });
    setShowAddModal(false);
    await loadArtifacts();
    return data.id;
  }, [apiCall, loadArtifacts]);

  const deleteArtifact = useCallback(async (artifactId) => {
    await apiCall(`/artifacts/${artifactId}`, {
      method: 'DELETE',
    });

    if (selectedArtifact && selectedArtifact.id === artifactId) {
      setSelectedArtifact(null);
    }
    await loadArtifacts();
  }, [apiCall, loadArtifacts, selectedArtifact]);

  const toggleFavorite = useCallback(async (artifactId) => {
    try {
      await apiCall(`/artifacts/${artifactId}/favorite`, {
        method: 'PATCH',
      });
      await loadArtifacts();
    } catch (err) {
      console.error('Failed to toggle favorite:', err);
    }
  }, [apiCall, loadArtifacts]);

  const downloadArtifact = useCallback(async (artifactId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/artifacts/${artifactId}/download`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = response.headers.get('Content-Disposition')?.split('filename=')[1] || `artifact-${artifactId}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Failed to download artifact:', err);
    }
  }, []);

  useEffect(() => {
    const initializeConnection = async () => {
      await checkApiConnection();
    };
    initializeConnection();
  }, [checkApiConnection]);

  useEffect(() => {
    const loadData = async () => {
      if (apiConnected) {
        await Promise.all([
          loadArtifacts(),
          loadProjects(),
          loadStatistics()
        ]);
      }
    };
    loadData();
  }, [apiConnected, loadArtifacts, loadProjects, loadStatistics]);

  const getTypeIcon = (type) => {
    const icons = {
      code: <Code className="w-4 h-4" />,
      document: <FileText className="w-4 h-4" />,
      html: <Globe className="w-4 h-4" />,
      data: <Database className="w-4 h-4" />,
    };
    return icons[type] || <FileText className="w-4 h-4" />;
  };

  const getLanguageColor = (language) => {
    const colors = {
      javascript: 'bg-yellow-100 text-yellow-800',
      python: 'bg-blue-100 text-blue-800',
      html: 'bg-orange-100 text-orange-800',
      css: 'bg-purple-100 text-purple-800',
      sql: 'bg-green-100 text-green-800',
      markdown: 'bg-gray-100 text-gray-800',
      json: 'bg-indigo-100 text-indigo-800'
    };
    return colors[language] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const AddArtifactModal = () => {
    const [formData, setFormData] = useState({
      title: '',
      content: '',
      artifact_type: 'code',
      language: '',
      project: 'default',
      tags: '',
      chat_context: ''
    });
    const [saving, setSaving] = useState(false);
    const [formError, setFormError] = useState(null);

    const handleSubmit = async (e) => {
      e.preventDefault();
      setSaving(true);
      setFormError(null);

      try {
        const tagsArray = formData.tags.split(',').map(tag => tag.trim()).filter(tag => tag);
        await createArtifact({
          ...formData,
          tags: tagsArray
        });
      } catch (err) {
        setFormError(err.message);
      } finally {
        setSaving(false);
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl m-4">
          <div className="p-6 border-b">
            <h2 className="text-xl font-semibold">Add New Artifact</h2>
          </div>
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            {formError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-700">
                {formError}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-2">Title *</label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({...formData, title: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter artifact title"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Type</label>
                <select
                  value={formData.artifact_type}
                  onChange={(e) => setFormData({...formData, artifact_type: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="code">Code</option>
                  <option value="document">Document</option>
                  <option value="html">HTML</option>
                  <option value="data">Data</option>
                  <option value="text">Text</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Language</label>
                <input
                  type="text"
                  value={formData.language}
                  onChange={(e) => setFormData({...formData, language: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., javascript, python"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Project</label>
              <select
                value={formData.project}
                onChange={(e) => setFormData({...formData, project: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {projectList.filter(p => p !== 'all').map(project => (
                  <option key={project} value={project}>{project}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Tags</label>
              <input
                type="text"
                value={formData.tags}
                onChange={(e) => setFormData({...formData, tags: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Comma-separated tags"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Content *</label>
              <textarea
                value={formData.content}
                onChange={(e) => setFormData({...formData, content: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 h-40"
                placeholder="Enter your artifact content here..."
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Context</label>
              <textarea
                value={formData.chat_context}
                onChange={(e) => setFormData({...formData, chat_context: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 h-20"
                placeholder="Optional context or description"
              />
            </div>

            <div className="flex justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={() => setShowAddModal(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
                disabled={saving}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                disabled={saving}
              >
                {saving ? 'Saving...' : 'Save Artifact'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  const StatsModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl m-4">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            System Statistics
          </h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{stats.total_artifacts || 0}</div>
              <div className="text-sm text-gray-600">Total Artifacts</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{stats.total_projects || 0}</div>
              <div className="text-sm text-gray-600">Projects</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{stats.storage_mb || 0} MB</div>
              <div className="text-sm text-gray-600">Storage Used</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-3">By Type</h3>
              <div className="space-y-2">
                {Object.entries(stats.by_type || {}).map(([type, count]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="flex items-center gap-2">
                      {getTypeIcon(type)}
                      {type}
                    </span>
                    <span className="font-medium">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="font-semibold mb-3">By Language</h3>
              <div className="space-y-2">
                {Object.entries(stats.by_language || {}).map(([language, count]) => (
                  <div key={language} className="flex justify-between items-center">
                    <span className={`px-2 py-1 rounded text-xs ${getLanguageColor(language)}`}>
                      {language}
                    </span>
                    <span className="font-medium">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="p-6 border-t bg-gray-50 flex justify-end">
          <button onClick={() => setShowStatsModal(false)} className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">Close</button>
        </div>
      </div>
    </div>
  );

  const ArtifactDetailModal = () => (
    selectedArtifact && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl m-4 max-h-[90vh] overflow-hidden">
          <div className="p-6 border-b flex justify-between items-start">
            <div>
              <h2 className="text-xl font-semibold">{selectedArtifact.title}</h2>
              <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                <span className="flex items-center gap-1">
                  {getTypeIcon(selectedArtifact.artifact_type)}
                  {selectedArtifact.artifact_type}
                </span>
                {selectedArtifact.language && (
                  <span className={`px-2 py-1 rounded text-xs ${getLanguageColor(selectedArtifact.language)}`}>
                    {selectedArtifact.language}
                  </span>
                )}
                <span>Size: {formatSize(selectedArtifact.size)}</span>
                <span>Created: {formatDate(selectedArtifact.created)}</span>
              </div>
            </div>
            {selectedArtifact.favorite && <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />}
          </div>

          <div className="flex-1 overflow-auto">
            <div className="p-6">
              <div className="mb-4">
                <h3 className="font-semibold mb-2">Tags</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedArtifact.tags?.map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm flex items-center gap-1">
                      <Tag className="w-3 h-3" />
                      {tag}
                    </span>
                  )) || <span className="text-gray-500">No tags</span>}
                </div>
              </div>

              {selectedArtifact.chat_context && (
                <div className="mb-4">
                  <h3 className="font-semibold mb-2">Context</h3>
                  <p className="text-gray-700 bg-gray-50 p-3 rounded">{selectedArtifact.chat_context}</p>
                </div>
              )}

              <div>
                <h3 className="font-semibold mb-2">Content</h3>
                <pre className="bg-gray-50 p-4 rounded-lg overflow-auto text-sm font-mono border">
                  {selectedArtifact.content}
                </pre>
              </div>
            </div>
          </div>

          <div className="p-6 border-t bg-gray-50 flex justify-between">
            <div className="flex space-x-2">
              <button
                onClick={() => toggleFavorite(selectedArtifact.id)}
                className={`px-3 py-2 rounded-md flex items-center gap-2 ${
                  selectedArtifact.favorite 
                    ? 'bg-yellow-500 text-white hover:bg-yellow-600' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                <Star className="w-4 h-4" />
                {selectedArtifact.favorite ? 'Unfavorite' : 'Favorite'}
              </button>
              <button
                onClick={() => downloadArtifact(selectedArtifact.id)}
                className="px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
              <button
                onClick={async () => {
                  if (window.confirm('Are you sure you want to delete this artifact?')) {
                    try {
                      await deleteArtifact(selectedArtifact.id);
                    } catch (err) {
                      alert(`Delete failed: ${err.message}`);
                    }
                  }
                }}
                className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
            <button onClick={() => setSelectedArtifact(null)} className="px-4 py-2 text-gray-600 hover:text-gray-800">Close</button>
          </div>
        </div>
      </div>
    )
  );

  if (error && !apiConnected) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Backend Connection Failed</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <div className="bg-gray-100 p-4 rounded-lg text-left">
            <p className="text-sm font-medium text-gray-900 mb-2">To start the backend:</p>
            <code className="text-sm text-gray-700 block">cd backend</code>
            <code className="text-sm text-gray-700 block">python artifact_manager.py --web</code>
          </div>
          <button
            onClick={checkApiConnection}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading artifacts...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Artifact Manager</h1>
              <span className="ml-3 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                {artifactList.length} artifacts
              </span>
              <span className={`ml-2 px-2 py-1 text-xs rounded-full ${
                apiConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {apiConnected ? '🟢 Connected' : '🔴 Disconnected'}
              </span>
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={async () => {
                  await loadStatistics();
                  setShowStatsModal(true);
                }}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md"
                title="View Statistics"
              >
                <BarChart3 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowAddModal(true)}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center gap-2"
                disabled={!apiConnected}
              >
                <Plus className="w-4 h-4" />
                Add Artifact
              </button>
            </div>
          </div>
        </div>
      </header>

      {error && apiConnected && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <p className="ml-3 text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder="Search artifacts by title or tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={!apiConnected}
                />
              </div>
            </div>

            <div className="flex gap-4">
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={!apiConnected}
              >
                <option value="all">All Types</option>
                <option value="code">Code</option>
                <option value="document">Document</option>
                <option value="html">HTML</option>
                <option value="data">Data</option>
                <option value="text">Text</option>
              </select>

              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={!apiConnected}
              >
                <option value="all">All Projects</option>
                {projectList.filter(p => p !== 'all').map(project => (
                  <option key={project} value={project}>{project}</option>
                ))}
              </select>

              <div className="flex border border-gray-300 rounded-md">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 ${viewMode === 'grid' ? 'bg-blue-50 text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
                  disabled={!apiConnected}
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 ${viewMode === 'list' ? 'bg-blue-50 text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
                  disabled={!apiConnected}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {artifactList.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No artifacts found</h3>
            <p className="text-gray-600">
              {apiConnected
                ? "Try adjusting your search criteria or create a new artifact."
                : "Connect to the backend to view your artifacts."
              }
            </p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {artifactList.map(artifact => (
              <div
                key={artifact.id}
                className="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => loadArtifactDetails(artifact.id)}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(artifact.artifact_type)}
                      <span className="text-sm text-gray-600">{artifact.artifact_type}</span>
                    </div>
                    {artifact.favorite && <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />}
                  </div>

                  <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">{artifact.title}</h3>

                  <div className="flex flex-wrap gap-1 mb-3">
                    {artifact.language && (
                      <span className={`px-2 py-1 rounded text-xs ${getLanguageColor(artifact.language)}`}>
                        {artifact.language}
                      </span>
                    )}
                    {artifact.tags?.slice(0, 2).map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                        {tag}
                      </span>
                    ))}
                    {artifact.tags?.length > 2 && (
                      <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                        +{artifact.tags.length - 2}
                      </span>
                    )}
                  </div>

                  <div className="text-xs text-gray-500 space-y-1">
                    <div>Project: {artifact.project}</div>
                    <div>Size: {formatSize(artifact.size)}</div>
                    <div>Modified: {formatDate(artifact.modified)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-sm">
            <div className="divide-y">
              {artifactList.map(artifact => (
                <div
                  key={artifact.id}
                  className="p-4 hover:bg-gray-50 cursor-pointer"
                  onClick={() => loadArtifactDetails(artifact.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(artifact.artifact_type)}
                        {artifact.favorite && <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />}
                      </div>

                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-gray-900 truncate">{artifact.title}</h3>
                        <div className="flex items-center gap-4 text-sm text-gray-600 mt-1">
                          <span>{artifact.artifact_type}</span>
                          {artifact.language && (
                            <span className={`px-2 py-1 rounded text-xs ${getLanguageColor(artifact.language)}`}>
                              {artifact.language}
                            </span>
                          )}
                          <span>Project: {artifact.project}</span>
                          <span>Size: {formatSize(artifact.size)}</span>
                        </div>
                      </div>

                      <div className="text-sm text-gray-500">
                        {formatDate(artifact.modified)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {showAddModal && <AddArtifactModal />}
      {showStatsModal && <StatsModal />}
      {selectedArtifact && <ArtifactDetailModal />}
    </div>
  );
};

export default ArtifactManager;