import axios from "axios";

const DJANGO_BASE = "http://localhost:8000/api";

const api = axios.create({
  baseURL: DJANGO_BASE,
  timeout: 30000,
});

// Upload files (supports multiple)
export const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const response = await api.post("/documents/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// Poll job status
export const getJobStatus = async (mlJobId) => {
  const response = await api.get(`/documents/status/${mlJobId}/`);
  return response.data;
};

// Download Excel
export const getDownloadUrl = (excelFilename) =>
  `${DJANGO_BASE}/documents/download/${excelFilename}/`;

// Save user corrections
export const saveCorrection = async (jobId, corrections, originalData) => {
  const response = await api.post("/documents/correction/", {
    job_id: jobId,
    corrections,
    original_data: originalData,
  });
  return response.data;
};

// Get training dataset stats
export const getTrainingStats = async () => {
  const response = await api.get("/documents/training/stats/");
  return response.data;
};

// Start training
export const startTraining = async (params = {}) => {
  const response = await api.post("/documents/training/start/", params);
  return response.data;
};

// Poll training status from FastAPI directly
export const getTrainingStatus = async (trainingId) => {
  const response = await axios.get(
    `http://localhost:8001/api/train/status/${trainingId}`
  );
  return response.data;
};