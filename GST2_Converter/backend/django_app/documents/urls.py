from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.UploadView.as_view(), name="upload"),
    path("status/<str:ml_job_id>/", views.JobStatusView.as_view(), name="job-status"),
    path("download/<str:excel_filename>/", views.DownloadExcelView.as_view(), name="download"),
    path("correction/", views.SaveCorrectionView.as_view(), name="save-correction"),
    path("training/stats/", views.TrainingStatsView.as_view(), name="training-stats"),
    path("training/start/", views.StartTrainingView.as_view(), name="start-training"),
]