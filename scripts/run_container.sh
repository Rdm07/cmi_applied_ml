docker run \
    --shm-size=2G \
    -p 5000:80 \
    --name spam-flask-app -it \
    -d img_spam_flask_app
exit 0