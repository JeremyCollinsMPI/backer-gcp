gcloud builds submit --tag gcr.io/backer-1585550181964/backergcp --timeout 3600
gcloud run deploy demo10 --image gcr.io/backer-1585550181964/backergcp --platform managed --memory 8Gi --cpu 2 --region asia-east1 --allow-unauthenticated --timeout 600
