# shark_tank

Predicting if a Shark Tank Pitch Will Get a Deal or Not

## By Joseph Cohen

***

# README

***

# Goals

Welcome to my project! The goal of this project is to determine what types of pitches are most likely to make a deal on the TV show Shark Tank. As one may or may not know, Shark Tank is an exciting reality show in which hopeful and budding entrepreneurs are given a unique opportunity to present their business to a group of proven and succesful entrepreneurs, known as the sharks. The sharks are a group of five (chosen from a constant pool of 6) who will spend time talking to each entrepreneur to get an understanding of all the things that matter when going into business with someone. If the sharks are impressed they will make an offer to join the business. At this point, the entrepreneurs looking for a deal have the option to either accept that deal or push for a better deal. Often times, entrepreneurs push their requests beyond what the sharks will commit to and may leave without a deal. So now that we know how this whole idea works, the goal of my project is to investigate 6 seasons worth of data to build a model and that will predict the success of an entrepreneur on the show in terms of getting a deal, and also identify leading indicators as to whether one may get a deal or not.

***

# Statistical Methods Leveraged

Machine Learning Classification

Predictive Modeling

Data Visualization

Machine Learning Pipelines

Feature Engineering

***

# Summary of Files

### shark_tank.csv
This is my primary data source and contains information on 6 seasons worth of data from the show.

### cleaned_shark.csv
This is a CSV file I created to save my work and read it into pandas for exploratory data analysis.

### eda_plan.txt
This is just some "scrap" paper where I write down some notes about how I intend to preform exploratory data analysis (I wouldn't spend a lot of time with that document, it's mainly for me).

### first_model.ipynb
This is my first notebook where I clean data and build a model.

### EDA.ipynb
This is where I perform exploratory data analysis and hypothesis testing.

### sharks_to_clean.csv
This is a quick csv I created that helped me sort out some categorical data.

### shark_dummy.ipynb
Here I did a little work just to decide how to handle categorical data representing sharks present.

### helper.py
This was a backend module I work to help automate data cleaning and build shorter models.

### second_model
Here, I applied my helper module to quickly run through data pre-processing and modeling.