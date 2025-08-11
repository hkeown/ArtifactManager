import React, { useState, useEffect } from 'react';
import { Search, Plus, Code, FileText, Globe, Database, Star, Tag, Calendar, Download, Trash2, Edit3, Eye, Filter, Grid, List, Settings, BarChart3 } from 'lucide-react';

const ArtifactManager = () => {
  const [artifacts, setArtifacts] = useState([]);
  const [projects, setProjects] = useState(['default']);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState('all');
  const [selectedProject, setSelectedProject] = useState('all');
  const [viewMode, setViewMode] = useState('grid');
  const [selectedArtifact, setSelectedArtifact] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);

  // Mock data for demonstration - replace with actual API calls
  useEffect(() => {
    const mockArtifacts = [
      {
        id: '20250810_143015_a6db7fd2e7ab',
        title: 'React Component Library',
        artifact_type: 'code',
        language: 'javascript',
        tags: ['react', 'components', 'ui'],
        created: '2025-08-10T14:30:15',
        modified: '2025-08-10T14:30:15',
        size: 2048,
        project: 'default',
        favorite: true,
        content: 'import React from "react";\n\nconst Button = ({ children, onClick }) => {\n  return (\n    <button onClick={onClick}>\n      {children}\n    </button>\n  );\n};\n\nexport default Button;'
      },
      {
        id: '20250810_143016_b7ec8gd3f8bc',
        title: 'API Documentation',
        artifact_type: 'document',
        language: 'markdown',
        tags: ['api', 'docs', 'reference'],
        created: '2025-08-10T13:45:22',
        modified: '2025-08-10T14:12:33',
        size: 1524,
        project: 'documentation',
        favorite: false,
        content: '# API Documentation\n\n## Authentication\n\nAll API requests require authentication...'
      },
      {
        id: '20250810_143017_c8fd9he4g9cd',
        title: 'Landing Page Template',
        artifact_type: 'html',
        language: 'html',
        tags: ['html', 'template', 'landing'],
        created: '2025-08-10T12:20:45',
        modified: '2025-08-10T12:20:45',
        size: 3072,
        project: 'templates',
        favorite: true,
        content: '<!DOCTYPE html>\n<html>\n<head>\n  <title>Landing Page</title>\n</head>\n<body>\n  <h1>Welcome</h1>\n</body>\n</html>'
      },
      {
        id: '20250810_143018_d9ge0if5h0de',
        title: 'Database Schema',
        artifact_type: 'data',
        language: 'sql',
        tags: ['database', 'schema', 'sql'],
        created: '2025-08-10T11:15:18',
        modified: '2025-08-10T11:15:18',
        size: 1856,
        project: 'database',
        favorite: false,
        content: 'CREATE TABLE users (\n  id SERIAL PRIMARY KEY,\n  username VARCHAR(50) UNIQUE NOT NULL,\n  email VARCHAR(100) UNIQUE NOT NULL\n);'
      }
    ];

    setArtifacts(mockArtifacts);
    setProjects(['default', 'documentation', 'templates', 'database']);
    setStats({
      total_artifacts: 4,
      total_projects: 4,
      by_type: { code: 1, document: 1, html: 1, data: 1 },
      by_language: { javascript: 1, markdown: 1, html: 1, sql: 1 },
      storage_mb: 8.5
    });
    setLoading(false);
  }, []);

  const getTypeIcon = (type) => {
    switch (type) {
      case 'code': return <Code className="w-4 h-4" />;
      case 'document': return <FileText className="w-4 h-4" />;
      case 'html': return <Globe className="w-4 h-4" />;
      case 'data': return <Database className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
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

  const filteredArtifacts = artifacts.filter(artifact => {
    const matchesSearch = artifact.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         artifact.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesType = selectedType === 'all' || artifact.artifact_type === selectedType;
    const matchesProject = selectedProject === 'all' || artifact.project === selectedProject;

    return matchesSearch && matchesType && matchesProject;
  });

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

  const AddArtifactModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl m-4">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold">Add New Artifact</h2>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Title</label>
            <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Enter artifact title" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Type</label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="code">Code</option>
                <option value="document">Document</option>
                <option value="html">HTML</option>
                <option value="data">Data</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Language</label>
              <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="e.g., javascript, python" />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Project</label>
            <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
              {projects.map(project => (
                <option key={project} value={project}>{project}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Tags</label>
            <input type="text" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Comma-separated tags" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Content</label>
            <textarea className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 h-40" placeholder="Enter your artifact content here..."></textarea>
          </div>
        </div>
        <div className="p-6 border-t bg-gray-50 flex justify-end space-x-3">
          <button onClick={() => setShowAddModal(false)} className="px-4 py-2 text-gray-600 hover:text-gray-800">Cancel</button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">Save Artifact</button>
        </div>
      </div>
    </div>
  );

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
              <div className="text-2xl font-bold text-blue-600">{stats.total_artifacts}</div>
              <div className="text-sm text-gray-600">Total Artifacts</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{stats.total_projects}</div>
              <div className="text-sm text-gray-600">Projects</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{stats.storage_mb} MB</div>
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
                  {selectedArtifact.tags.map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm flex items-center gap-1">
                      <Tag className="w-3 h-3" />
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

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
              <button className="px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2">
                <Edit3 className="w-4 h-4" />
                Edit
              </button>
              <button className="px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-2">
                <Download className="w-4 h-4" />
                Download
              </button>
              <button className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center gap-2">
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
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Claude Artifact Manager</h1>
              <span className="ml-3 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                {filteredArtifacts.length} artifacts
              </span>
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowStatsModal(true)}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md"
                title="View Statistics"
              >
                <BarChart3 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowAddModal(true)}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Add Artifact
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Filters */}
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
                />
              </div>
            </div>

            <div className="flex gap-4">
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Types</option>
                <option value="code">Code</option>
                <option value="document">Document</option>
                <option value="html">HTML</option>
                <option value="data">Data</option>
              </select>

              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Projects</option>
                {projects.map(project => (
                  <option key={project} value={project}>{project}</option>
                ))}
              </select>

              <div className="flex border border-gray-300 rounded-md">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 ${viewMode === 'grid' ? 'bg-blue-50 text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 ${viewMode === 'list' ? 'bg-blue-50 text-blue-600' : 'text-gray-600 hover:text-gray-900'}`}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Artifacts Grid/List */}
        {filteredArtifacts.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No artifacts found</h3>
            <p className="text-gray-600">Try adjusting your search criteria or create a new artifact.</p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredArtifacts.map(artifact => (
              <div
                key={artifact.id}
                className="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => setSelectedArtifact(artifact)}
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
                    {artifact.tags.slice(0, 2).map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                        {tag}
                      </span>
                    ))}
                    {artifact.tags.length > 2 && (
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
              {filteredArtifacts.map(artifact => (
                <div
                  key={artifact.id}
                  className="p-4 hover:bg-gray-50 cursor-pointer"
                  onClick={() => setSelectedArtifact(artifact)}
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

      {/* Modals */}
      {showAddModal && <AddArtifactModal />}
      {showStatsModal && <StatsModal />}
      {selectedArtifact && <ArtifactDetailModal />}
    </div>
  );
};

export default ArtifactManager;