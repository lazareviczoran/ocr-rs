set -e
REPO="lazareviczoran/ocr-rs"

if [[ ( "$SEMAPHORE_GIT_BRANCH" == "master" ) \
        && ( -z "$(git diff $SEMAPHORE_GIT_COMMIT_RANGE --name-only|grep Dockerfile)" ) ]]; then
    echo 'Preparing to push to docker hub'

    # Load tag name from semaphore cache
    cache restore $SEMAPHORE_GIT_BRANCH-image-tag
    TAG=$(cat image-tag.txt)

    # install required libs
    apt-get install jq -y

    # Fetch existing tags, and use latest that matches pattern (e.g. 0.0.1) if exists
    LATEST_VERSION=curl "https://registry.hub.docker.com/v2/repositories/$($REPO)/tags?ordering=last_updated" \
        |jq '."results"[]["name"]' \
        |grep -E -m 1 '\d+\.\d+\.\d+'
    if [ -z $LATEST_VERSION ]; then
        LATEST_VERSION="0.0.1"
    fi
    VALUES=($(echo $LATEST_VERSION|grep -Eo '\d+'))
    VALUES[3]=$((VALUES[3] + 1))
    NEW_VERSION=$(join . ${VALUES[@]})

    echo $DOCKER_HUB_ACCESS_TOKEN | docker login --username $DOCKER_USERNAME --password-stdin
    docker pull $REPO:$TAG
    docker tag $REPO:$TAG $REPO:$NEW_VERSION
    docker push $REPO:$NEW_VERSION
    docker tag $REPO:$TAG $REPO:latest
    docker push $REPO:latest

    echo 'Successfully uploaded new image version'
else
    echo 'Not on main branch (master), skipping push'
fi