# Gender Detection Project

A Machine Learning web application developed using Flask that predicts gender based on names.

## Features

* Predict gender from names
* Flask-based web application
* Machine Learning integration
* Simple and beginner-friendly UI

## Technologies Used

* Python
* Flask
* Scikit-learn
* HTML/CSS
* Pandas

## Project Structure

gender-detection-project/
│
├── app.py
├── train_model.py
├── gender.csv
├── model.pkl
├── vectorizer.pkl
│
├── templates/
│ └── index.html

## How to Run

Install required libraries:

pip install flask pandas scikit-learn numpy

Train model:

python train_model.py

Run Flask application:

python app.py

Open browser:

http://127.0.0.1:5000

## Output

User enters a name and the application predicts whether the gender is Male or Female.
