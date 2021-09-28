docker build -t jeremycollinsmpi/backer .
docker run -it --rm --name gcp -v $PWD:/app -p 8080:8080 -v $PWD/cache:/root/.cache jeremycollinsmpi/backer /bin/bash