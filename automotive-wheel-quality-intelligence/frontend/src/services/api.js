/**
 * Backend API Client Service
 * Connects frontend dashboard with FastAPI backend endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchDefects(params = {}) {
  const query = new URLSearchParams(params).toString();
  const url = `${API_BASE_URL}/api/defects${query ? `?${query}` : ''}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch defects: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchAlerts() {
  const response = await fetch(`${API_BASE_URL}/api/alerts`);
  if (!response.ok) {
    throw new Error(`Failed to fetch alerts: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchRiskSummary() {
  const response = await fetch(`${API_BASE_URL}/api/risk`);
  if (!response.ok) {
    throw new Error(`Failed to fetch risk summary: ${response.statusText}`);
  }
  return response.json();
}
