
# 
docker build -t my-fastapi-app .

docker run -d --name my-fastapi-container -p 8000:8000 my-fastapi-app

docker start my-fastapi-container