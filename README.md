## SETUP-INSTRUCTIONS
1. Install Anaconda to create environment.
2. Run anaconda environment.
3. Run `pip install -r requirements.txt` to install the required dependencies. If there is a problem, the requirements file contains the names of the dependencies you can install them one by one.
4. Run `python manage.py runserver` to run the server.
5. Go to `http://localhost:8000/soil/train-page` to train the model. Click the button and wait for training to be complete. Notifications will show. This will train the model and create a model called img_model.p.
6. Go to `http://localhost:8000/soil/test-page` to try out different pictures. Load a picture and click submit, the predicted result willl show.

