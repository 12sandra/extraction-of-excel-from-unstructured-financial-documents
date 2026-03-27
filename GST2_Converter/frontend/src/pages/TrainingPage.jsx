import { useState, useEffect } from "react";
import { getTrainingStats, startTraining, getTrainingStatus } from "../services/api";

const CARD_STYLE = {
  background: "white",
  borderRadius: "12px",
  padding: "24px",
  boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
  marginBottom: "24px",
};

export default function TrainingPage() {
  const [stats, setStats] = useState(null);
  const [training, setTraining] = useState(null);
  const [trainingId, setTrainingId] = useState(null);
  const [epochs, setEpochs] = useState(3);
  const [annotatedOnly, setAnnotatedOnly] = useState(true);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  useEffect(() => {
    if (!trainingId) return;
    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(trainingId);
        setTraining(status);
        if (status.status === "complete" || status.status === "failed") {
          clearInterval(interval);
          loadStats();
        }
      } catch {}
    }, 3000);
    return () => clearInterval(interval);
  }, [trainingId]);

  const loadStats = async () => {
    try {
      const data = await getTrainingStats();
      setStats(data);
    } catch {}
  };

  const handleStartTraining = async () => {
    if (!stats?.ready_for_training) {
      alert("Not enough training data yet. Need at least 5 annotated documents.");
      return;
    }
    setLoading(true);
    try {
      const result = await startTraining({ epochs, annotated_only: annotatedOnly });
      setTrainingId(result.training_id);
      setTraining({ status: "queued", ...result });
    } catch (err) {
      alert("Failed to start training: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "32px 24px" }}>
      <div style={{ marginBottom: "24px" }}>
        <h1 style={{ color: "#1a237e", margin: "0 0 8px", fontSize: "24px" }}>
          🧠 Model Training Center
        </h1>
        <p style={{ color: "#666", margin: 0, fontSize: "14px" }}>
          Fine-tune LayoutLMv3 on your GST 2A data to improve extraction accuracy over time.
        </p>
      </div>

      {/* Dataset Stats */}
      <div style={CARD_STYLE}>
        <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>📊 Training Dataset</h3>
        {stats ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
            {[
              { label: "Raw Documents", value: stats.raw_documents, color: "#2196f3" },
              { label: "Annotated Documents", value: stats.annotated_documents, color: "#4caf50" },
              { label: "Total Documents", value: stats.total_documents, color: "#9c27b0" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: `${color}15`,
                border: `2px solid ${color}30`,
                borderRadius: "10px",
                padding: "16px",
                textAlign: "center"
              }}>
                <div style={{ fontSize: "32px", fontWeight: "700", color }}>{value}</div>
                <div style={{ fontSize: "13px", color: "#666", marginTop: "4px" }}>{label}</div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: "#999", textAlign: "center", padding: "20px" }}>
            Loading statistics...
          </div>
        )}

        {stats && !stats.ready_for_training && (
          <div style={{
            marginTop: "16px",
            background: "#fff3e0",
            border: "1px solid #ffb74d",
            borderRadius: "8px",
            padding: "12px 16px",
            fontSize: "14px",
            color: "#e65100"
          }}>
            ⚠️ Need at least <strong>5 annotated documents</strong> to start training.
            Keep uploading and correcting documents to build your dataset.
            Currently: {stats.annotated_documents}/5
          </div>
        )}
      </div>

      {/* Training Config */}
      <div style={CARD_STYLE}>
        <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>⚙️ Training Configuration</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <label style={{ display: "block", fontWeight: "600", marginBottom: "6px", color: "#333" }}>
              Training Epochs: {epochs}
            </label>
            <input
              type="range"
              min="1" max="10" value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "12px", color: "#888" }}>
              More epochs = more learning, but takes longer. Start with 3.
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <input
              type="checkbox"
              id="annotated"
              checked={annotatedOnly}
              onChange={(e) => setAnnotatedOnly(e.target.checked)}
            />
            <label htmlFor="annotated" style={{ fontSize: "14px", color: "#333" }}>
              Use only annotated (user-corrected) data — Recommended
            </label>
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={loading || !stats?.ready_for_training || training?.status === "training"}
          style={{
            marginTop: "20px",
            background: stats?.ready_for_training ? "#1565c0" : "#ccc",
            color: "white",
            border: "none",
            padding: "12px 28px",
            borderRadius: "8px",
            fontSize: "15px",
            fontWeight: "600",
            cursor: stats?.ready_for_training ? "pointer" : "not-allowed",
            width: "100%",
          }}
        >
          {loading ? "Starting..." :
           training?.status === "training" ? "⚙️ Training in progress..." :
           training?.status === "complete" ? "✅ Start New Training Run" :
           "🚀 Start Fine-Tuning"}
        </button>
      </div>

      {/* Training Progress */}
      {training && (
        <div style={CARD_STYLE}>
          <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>📈 Training Progress</h3>
          <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "12px" }}>
            <span style={{
              background: training.status === "complete" ? "#4caf50" :
                         training.status === "failed" ? "#f44336" : "#2196f3",
              color: "white",
              padding: "4px 12px",
              borderRadius: "20px",
              fontSize: "12px",
              fontWeight: "600",
              textTransform: "uppercase"
            }}>
              {training.status}
            </span>
            <span style={{ color: "#666", fontSize: "14px" }}>
              {training.message || `${training.data_files} documents, ${training.epochs} epochs`}
            </span>
          </div>
          {training.status === "training" && (
            <div style={{
              background: "#e3f2fd",
              borderRadius: "4px",
              height: "8px",
              overflow: "hidden",
              marginBottom: "8px"
            }}>
              <div style={{
                width: `${training.progress || 0}%`,
                height: "100%",
                background: "#1565c0",
                borderRadius: "4px",
                transition: "width 0.5s ease",
                animation: "pulse 1.5s infinite"
              }} />
            </div>
          )}
          {training.status === "complete" && (
            <div style={{
              background: "#e8f5e9",
              border: "1px solid #81c784",
              borderRadius: "8px",
              padding: "12px",
              color: "#2e7d32",
              fontSize: "14px"
            }}>
              ✅ Training complete! The model has been updated.
              New document extractions will use the improved model.
              {training.accuracy && ` Accuracy: ${(training.accuracy * 100).toFixed(1)}%`}
            </div>
          )}
        </div>
      )}

      {/* How It Works */}
      <div style={{ ...CARD_STYLE, background: "#f8faff" }}>
        <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>💡 How Continuous Learning Works</h3>
        <div style={{ fontSize: "14px", color: "#555", lineHeight: "1.8" }}>
          <p>1. <strong>Upload</strong> GST 2A statements → system extracts data automatically</p>
          <p>2. <strong>Review</strong> the extracted Excel file and note any errors</p>
          <p>3. <strong>Correct</strong> any wrong values (corrections are saved automatically)</p>
          <p>4. <strong>Train</strong> — use this page to fine-tune LayoutLMv3 on your corrections</p>
          <p>5. Future extractions <strong>improve automatically</strong> as the model learns your document patterns</p>
        </div>
      </div>
    </div>
  );
}