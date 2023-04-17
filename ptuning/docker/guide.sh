#!/usr/bin/env bash

# build docker for chatglm-6b training
docker build

docker tag
docker login
docker push

docker run -it -
