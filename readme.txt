This application was built using a 64bit windows 10 pc.
Python version = 3.9.1

To run this application there are a few steps.

First Clone this repository to your PC using git clone https://github.com/tonymorris21/interimproject.git
Navigate to the project folder by typing cd inteirmproject/project
Next run python setup.py build followed by python setup.py install * You must have administrative permissions to run this command*
Now all of the dependencies should be installed
Run python __init__.py 
After running this command you should see "Running on http://127.0.0.1:5000/" in the output. 

To use this application go to http://127.0.0.1:5000/
You can log in with an account already made with email as admin@admin.com and password as admin
There is already a project created. You can click upload to upload a dataset. There are datasets already in the upload folder(one of these can be selected). 
The water potability csv can be uploaded and the three features with missing values can be imputed using the median. The potability feature can be encoded using label encoding.
The target feature must be set to potability before clicking train. On the training configuration apge you can select any algorithm without filling in the additional parameters and click train. 
This will load a model evaluation page. 
