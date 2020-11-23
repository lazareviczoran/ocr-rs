set -e
REPO="lazareviczoran/ocr-rs"

# Clear existing tag from semaphore cache
cache delete $SEMAPHORE_GIT_BRANCH-image-tag

git remote add ocr-remote https://github.com/lazareviczoran/ocr-rs
git fetch ocr-remote

if [[ ( "$SEMAPHORE_GIT_BRANCH" == "master" ) \
        && ( -n "$(git diff $SEMAPHORE_GIT_COMMIT_RANGE --name-only|grep Dockerfile)" ) \
    || ( "$SEMAPHORE_GIT_BRANCH" != "master" ) \
        && ( -n "$(git diff ocr-remote/master $SEMAPHORE_GIT_BRANCH --name-only|grep Dockerfile)" ) ]]; then
    IMAGE_TAG=$SEMAPHORE_GIT_BRANCH
    echo $DOCKER_HUB_ACCESS_TOKEN | docker login --username $DOCKER_USERNAME --password-stdin
    docker build -t $REPO:$IMAGE_TAG .
    docker push $REPO:$IMAGE_TAG
else
    IMAGE_TAG=latest
fi

echo $IMAGE_TAG > image-tag.txt

# Store image tag in semaphore cache
cache store $SEMAPHORE_GIT_BRANCH-image-tag image-tag.txt