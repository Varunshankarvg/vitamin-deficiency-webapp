import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [reportUrl, setReportUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [details, setDetails] = useState(null); // prediction + severity + image
  const [allowDownload, setAllowDownload] = useState(false); // NEW: allow download after view prediction

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setReportUrl(null);
    setSuccess(false);
    setDetails(null);
    setAllowDownload(false);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      setSuccess(false);
      const response = await axios.post('http://localhost:5000/predict', formData, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      setReportUrl(url);
      setSuccess(true);
    } catch (error) {
      console.error(error);
      alert('Error generating report');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictDetails = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/predict-details', formData);
      setDetails(response.data);
      setAllowDownload(true); // ✅ Only after prediction, allow report download
    } catch (error) {
      console.error(error);
      alert("Failed to get prediction details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Vitamin Deficiency Detection</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />

      <div style={{ marginTop: '20px' }}>
        <button onClick={handlePredictDetails} className="upload-button secondary" disabled={loading}>
          {loading ? 'Processing...' : 'View Prediction'}
        </button>
      </div>

      {details && (
        <div className="result-box">
          <h2>🎯 Prediction Results</h2>
          {details.predictions.map((pred, index) => (
            <p key={index}>- {pred.label}: {(pred.confidence * 100).toFixed(2)}%</p>
          ))}
          <p><strong>🔥 Severity:</strong> {details.severity_percent.toFixed(2)}% ({details.severity_level})</p>

          <h3>🧠 Model GradCAM Focus</h3>
          <img
            src={`http://localhost:5000${details.gradcam_url}`}
            alt="GradCAM"
            className="gradcam-image"
          />
        </div>
      )}

      {allowDownload && (
        <>
          <div style={{ marginTop: '20px' }}>
            <button onClick={handleUpload} className="upload-button">
              {loading ? 'Generating PDF...' : 'Generate Report'}
            </button>
          </div>

          {success && (
            <a href={reportUrl} download="vitamin_deficiency_report.pdf">
              <button className="download-button" style={{ marginTop: '10px' }}>
                Download Your Report
              </button>
            </a>
          )}
        </>
      )}
    </div>
  );
}

export default App;
