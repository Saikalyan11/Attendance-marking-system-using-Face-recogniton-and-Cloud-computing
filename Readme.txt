We always know that Facerecogntion is used as a biometric security scanner to scan our faces to take attendance of 
each person and those person's photos should be stored in a secure place and they also need storage. So if we use 
Hardware storage it's gonna cost more. Then After all that I thought of using Cloud instead of hardware so in this 
project we used face recognition using Cloud computng. I know that their is a service called AWS Rekognition but I 
wanted to do using my own machine learning model....

To design this kind of system make sire you follow these below steps:


1.Create an AWS account if you do not already have one.

2. Create an Amazon S3 bucket to store the model or images and any
associated files.

3. Allow Public access to each and every image stored in S3 bucket.

4.Go to IAM dashboard and create a IAM user with administration permission and download those AMI access and secret keys.

5.Now open terminal and install the boto3 library and configure the IAM user with access keys.

6. And then write code for retrieving images from s3 and merge the face recognition code with it