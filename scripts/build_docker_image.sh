count=`ls -1 *Dockerfile 2>/dev/null | wc -l`
if [ $count != 0 ]
then 
echo go ahead
else
cd ..
fi 

echo "$PWD"
docker build --network=host -t img_spam_flask_app .