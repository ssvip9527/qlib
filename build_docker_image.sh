#!/bin/bash

docker_user="your_dockerhub_username"

read -p "Do you want to build the nightly version of the qlib image? (default is stable) (yes/no): " answer;
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

if [ "$answer" = "yes" ]; then
    # 构建 qlib 镜像的 nightly 版本
    docker build --build-arg IS_STABLE=no -t qlib_image -f ./Dockerfile .
    image_tag="nightly"
else
    # 构建 qlib 镜像的 stable 版本
    docker build -t qlib_image -f ./Dockerfile .
    image_tag="stable"
fi

read -p "Is it uploaded to docker hub? (default is no) (yes/no): " answer;
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

if [ "$answer" = "yes" ]; then
    # 登录 Docker Hub
    # 如果你是新注册的 docker hub 用户，请在执行此步骤前先验证你的邮箱地址。
    docker login
    # 标记 Docker 镜像
    docker tag qlib_image "$docker_user/qlib_image:$image_tag"
    # 推送 Docker 镜像到 Docker Hub
    docker push "$docker_user/qlib_image:$image_tag"
else
    echo "Not uploaded to docker hub."
fi
