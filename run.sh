docker build -t jeremycollinsmpi/backer .
docker run -it --rm --name backer -p 8080:8080 jeremycollinsmpi/backer 