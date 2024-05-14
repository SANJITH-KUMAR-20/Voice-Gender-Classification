**Simple platform for audio voice classification**

A simple project that classifies a voice based on gender.

To run this project you'll first need to create a new environment, and install all dependecies.

` python -m venv voiceenv `

You will need to install the dependecies in the new environment, for that you can use the following commands.

` source voiceenv/Scripts/activate ` - bash

` voiceenv/Scripts/activate.ps1 ` - powershell

` pip install -r requirements.txt `


After this move into the project directory and start the server and the streamlit app
before that put the model in the main directory

` uvicorn app:app --reload `

` streamlit run page.py `





