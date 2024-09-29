# Football Positional Analysis Tool

Inspired by xG and Possession Models, this project aims to utilize Computer Vision (CV) to directly translate in-real-life players' movements/actions into tangible data that can be computed to *"Datify and Prettify"* statistics (Under Data Visualization).
<p> This work started with the creation and annotation of a large dataset of football images, from which I gathered through an existing collection of football clips provided by the Bundesliga ©, in a Kaggle competition. My dataset -- composed of 2306 images divided into 85% Train | 10% Valid | 5% Test. Dataset features imagery changes in orientation, saturation, and brightness levels -- can be found in the following <b><a href=https://app.roboflow.com/alberto-alves-n6pue/footballanalytics/5> link </a></b>. Trained Model is currently unavailable in this repository because of its sizing (>100MB), which will later be made downloadable. </p>

<p>Training and Validation was conducted on Kaggle Notebooks.</p>

### Objective
Achieving a positive detection and tracking rate into a model which allows for the correct analysis of player/ball movement, as well as, the creation of pass maps and 2D (maybe 3D) representations of the pitch.

## Features
- Dataset Creation and Annotation.
- Detection of Players, Goalkeepers, Referees, and Ball.
- Stabilization of Ball Detection.
- Unique Object Tracking.

## Results (In Progress)

Currently, the model's inference results look like the following:

https://github.com/user-attachments/assets/27f2b526-ef66-4609-99f0-448eaab71f27

Ball detection remains a problem. Accuracy could probably be improved further, but alternatives are needed.

## Contributions
