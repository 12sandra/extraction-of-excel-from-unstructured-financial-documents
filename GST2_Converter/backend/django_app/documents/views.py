import os
import requests
import json
from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import FileResponse
from pathlib import Path

from .models import UploadJob, ExtractedRecord, UserCorrection


class UploadView(APIView):
    """Handle file uploads — forwards to FastAPI ML service."""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        files = request.FILES.getlist("files")
        
        if not files:
            return Response({"error": "No files provided"}, status=400)
        
        results = []
        
        for file in files:
            # Save file and create job record in Django
            job = UploadJob.objects.create(
                original_filename=file.name,
                uploaded_file=file,
                status="queued"
            )
            
            # Forward to FastAPI ML service
            try:
                file.seek(0)  # Reset file pointer
                ml_response = requests.post(
                    f"{settings.FASTAPI_ML_URL}/api/extract/upload",
                    files={"files": (file.name, file, file.content_type)},
                    timeout=10
                )
                
                if ml_response.status_code == 200:
                    ml_data = ml_response.json()
                    ml_job = ml_data["jobs"][0]
                    
                    # Update Django job with ML service job ID
                    job.status = "processing"
                    job.save()
                    
                    results.append({
                        "django_job_id": str(job.id),
                        "ml_job_id": ml_job["job_id"],
                        "filename": file.name,
                        "status": "processing"
                    })
                else:
                    job.status = "failed"
                    job.error_message = f"ML service error: {ml_response.status_code}"
                    job.save()
                    results.append({
                        "django_job_id": str(job.id),
                        "filename": file.name,
                        "status": "failed",
                        "error": "ML service unavailable"
                    })
                    
            except requests.exceptions.ConnectionError:
                job.status = "failed"
                job.error_message = "Cannot connect to ML service. Is FastAPI running?"
                job.save()
                results.append({
                    "django_job_id": str(job.id),
                    "filename": file.name,
                    "status": "failed",
                    "error": "ML service not running. Start FastAPI first."
                })
        
        return Response({"jobs": results, "total": len(results)})


class JobStatusView(APIView):
    """Get status of a job by polling FastAPI ML service."""
    
    def get(self, request, ml_job_id):
        try:
            response = requests.get(
                f"{settings.FASTAPI_ML_URL}/api/extract/status/{ml_job_id}",
                timeout=5
            )
            if response.status_code == 200:
                return Response(response.json())
            return Response({"error": "Job not found"}, status=404)
        except:
            return Response({"error": "ML service unavailable"}, status=503)


class DownloadExcelView(APIView):
    """Download Excel file."""
    
    def get(self, request, excel_filename):
        excel_path = Path(os.getenv("MEDIA_ROOT", "media")) / "outputs" / excel_filename
        
        if not excel_path.exists():
            return Response({"error": "File not found"}, status=404)
        
        response = FileResponse(
            open(str(excel_path), "rb"),
            as_attachment=True,
            filename=excel_filename,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        return response


class SaveCorrectionView(APIView):
    """Save user corrections for continuous learning."""
    
    def post(self, request):
        job_id = request.data.get("job_id")
        corrections = request.data.get("corrections", {})
        original_data = request.data.get("original_data", {})
        
        if not job_id or not corrections:
            return Response({"error": "job_id and corrections are required"}, status=400)
        
        # Save to training data folder for future fine-tuning
        training_path = Path(os.getenv("TRAINING_DATA_PATH", "./training_data")) / "annotated"
        training_path.mkdir(parents=True, exist_ok=True)
        
        correction_entry = {
            "job_id": job_id,
            "corrections": corrections,
            "original_data": original_data,
            "timestamp": timezone.now().isoformat()
        }
        
        import uuid
        save_path = training_path / f"correction_{uuid.uuid4()}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(correction_entry, f, ensure_ascii=False, indent=2)
        
        return Response({
            "success": True,
            "message": "Correction saved. Thank you! This will improve future extractions.",
            "file_saved": save_path.name
        })


class TrainingStatsView(APIView):
    """Get training dataset statistics."""
    
    def get(self, request):
        try:
            response = requests.get(
                f"{settings.FASTAPI_ML_URL}/api/train/dataset/stats",
                timeout=5
            )
            return Response(response.json())
        except:
            return Response({"error": "ML service unavailable"}, status=503)


class StartTrainingView(APIView):
    """Trigger bulk training/fine-tuning."""
    
    def post(self, request):
        try:
            response = requests.post(
                f"{settings.FASTAPI_ML_URL}/api/train/start",
                json={
                    "epochs": request.data.get("epochs", 3),
                    "batch_size": request.data.get("batch_size", 4),
                    "annotated_only": request.data.get("annotated_only", True)
                },
                timeout=10
            )
            return Response(response.json(), status=response.status_code)
        except requests.exceptions.ConnectionError:
            return Response({"error": "ML service not running"}, status=503)