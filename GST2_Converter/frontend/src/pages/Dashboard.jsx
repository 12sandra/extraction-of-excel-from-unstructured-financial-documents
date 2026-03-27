import { useState, useCallback, useRef } from "react";
import { uploadFiles, getJobStatus, getDownloadUrl } from "../services/api";

const CARD_STYLE = {
  background: "white",
  borderRadius: "12px",
  padding: "24px",
  boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
  marginBottom: "24px",
};

const STATUS_COLORS = {
  queued: "#ff9800",
  processing: "#2196f3",
  complete: "#4caf50",
  failed: "#f44336",
};

const STATUS_ICONS = {
  queued: "⏳",
  processing: "⚙️",
  complete: "✅",
  failed: "❌",
};

export default function Dashboard() {
  const [dragging, setDragging] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef();

  const handleFiles = useCallback(async (files) => {
    const fileList = Array.from(files);
    const allowed = fileList.filter((f) =>
      ["application/pdf", "image/jpeg", "image/png", "image/jpg"].includes(f.type)
    );

    if (!allowed.length) {
      alert("Please upload PDF, JPG, or PNG files only.");
      return;
    }

    setUploading(true);
    try {
      const response = await uploadFiles(allowed);
      const newJobs = response.jobs.map((job) => ({
        ...job,
        status: "processing",
        percent: 0,
        step: "Starting...",
        filename: job.filename,
      }));
      setJobs((prev) => [...newJobs, ...prev]);

      // Poll each job for status
      newJobs.forEach((job) => pollJobStatus(job.ml_job_id));
    } catch (err) {
      alert("Upload failed: " + (err.response?.data?.error || err.message));
    } finally {
      setUploading(false);
    }
  }, []);

  const pollJobStatus = async (mlJobId) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(mlJobId);
        setJobs((prev) =>
          prev.map((job) =>
            job.ml_job_id === mlJobId ? { ...job, ...status } : job
          )
        );
        if (status.status !== "complete" && status.status !== "failed") {
          setTimeout(poll, 2000); // Poll every 2 seconds
        }
      } catch {
        setTimeout(poll, 3000);
      }
    };
    poll();
  };

  return (
    <div style={{ maxWidth: "960px", margin: "0 auto", padding: "32px 24px" }}>
      {/* Upload Zone */}
      <div style={CARD_STYLE}>
        <h2 style={{ margin: "0 0 8px", color: "#1a237e", fontSize: "20px" }}>
          📤 Upload GST 2A Statements
        </h2>
        <p style={{ margin: "0 0 20px", color: "#666", fontSize: "14px" }}>
          Upload PDF, JPG, PNG, or JPEG files. Multiple files supported.
        </p>

        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          style={{
            border: `2px dashed ${dragging ? "#1565c0" : "#90caf9"}`,
            borderRadius: "12px",
            padding: "48px 24px",
            textAlign: "center",
            cursor: "pointer",
            background: dragging ? "#e3f2fd" : "#f8faff",
            transition: "all 0.2s",
          }}
        >
          <div style={{ fontSize: "48px", marginBottom: "12px" }}>📁</div>
          <div style={{ fontSize: "18px", fontWeight: "600", color: "#1a237e", marginBottom: "8px" }}>
            {dragging ? "Drop files here!" : "Drag & drop files or click to browse"}
          </div>
          <div style={{ fontSize: "13px", color: "#888" }}>
            Supported: PDF, JPG, PNG, JPEG • Max 50MB per file
          </div>
          {uploading && (
            <div style={{ marginTop: "16px", color: "#1565c0", fontWeight: "600" }}>
              ⚙️ Uploading and queuing files...
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.jpg,.jpeg,.png"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {/* Jobs List */}
      {jobs.length > 0 && (
        <div style={CARD_STYLE}>
          <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>
            📋 Processing Jobs ({jobs.length})
          </h3>
          {jobs.map((job, idx) => (
            <JobCard key={job.ml_job_id || idx} job={job} />
          ))}
        </div>
      )}

      {jobs.length === 0 && (
        <div style={{
          ...CARD_STYLE,
          textAlign: "center",
          padding: "48px",
          color: "#999"
        }}>
          <div style={{ fontSize: "64px", marginBottom: "16px" }}>📊</div>
          <div style={{ fontSize: "16px" }}>
            No jobs yet. Upload GST 2A statements to get started.
          </div>
        </div>
      )}
    </div>
  );
}

function JobCard({ job }) {
  const color = STATUS_COLORS[job.status] || "#999";
  const icon = STATUS_ICONS[job.status] || "🔄";

  return (
    <div style={{
      border: `1px solid ${color}30`,
      borderLeft: `4px solid ${color}`,
      borderRadius: "8px",
      padding: "16px",
      marginBottom: "12px",
      background: `${color}08`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontWeight: "600", color: "#333", fontSize: "15px" }}>
            {icon} {job.filename}
          </div>
          <div style={{ color: "#666", fontSize: "13px", marginTop: "4px" }}>
            {job.message || job.step || job.status}
            {job.records_extracted ? ` • ${job.records_extracted} records` : ""}
            {job.confidence ? ` • ${(job.confidence * 100).toFixed(1)}% confidence` : ""}
            {job.processing_time ? ` • ${job.processing_time}s` : ""}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
          <span style={{
            background: color,
            color: "white",
            padding: "4px 12px",
            borderRadius: "20px",
            fontSize: "12px",
            fontWeight: "600",
            textTransform: "uppercase"
          }}>
            {job.status}
          </span>
          {job.status === "complete" && job.excel_filename && (
            <a
              href={getDownloadUrl(job.excel_filename)}
              download
              style={{
                background: "#1565c0",
                color: "white",
                padding: "6px 16px",
                borderRadius: "6px",
                textDecoration: "none",
                fontSize: "13px",
                fontWeight: "600",
              }}
            >
              ⬇️ Download Excel
            </a>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {job.status === "processing" && (
        <div style={{
          marginTop: "12px",
          background: "#e3f2fd",
          borderRadius: "4px",
          height: "6px",
          overflow: "hidden"
        }}>
          <div style={{
            width: `${job.percent || 0}%`,
            height: "100%",
            background: "#1565c0",
            borderRadius: "4px",
            transition: "width 0.5s ease",
          }} />
        </div>
      )}
    </div>
  );
}