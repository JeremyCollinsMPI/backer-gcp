docker build -t jeremycollinsmpi/backer .
docker run -it --rm --name gcp -p 8080:8080 -v $PWD/cache:/root/.cache jeremycollinsmpi/backer python download_models.py