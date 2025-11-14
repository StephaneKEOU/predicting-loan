################################
### DOCKER LOCAL
################################

build_container_local:
	docker build --tag=$$IMAGE:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 $$IMAGE:dev

################################
### DOCKER DEPLOYMENT
################################

# Step 1 (Only the first time)
allow_docker_push:
	gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

# Step 2 (Only the first time)
create_artifacts_repo:
	gcloud artifacts repositories create $$ARTIFACTSREPO --repository-format=docker \
		--location=$$GCP_REGION --description="Docker repository"

# Step 3
build_for_production:
	docker build -t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

# Step 3 ( M1 M2 M3 M4 M5 Chips )
m_chip_build_image_production:
	docker build --platform linux/amd64 --tag=$$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

# Step 4
push_image_production:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod

# Step 5
deploy_to_cloud_run:
	gcloud run deploy --image=$$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod \
		--platform=managed --region=$$GCP_REGION --allow-unauthenticated --port=8000 $$CLOUDRUN_SERVICE
