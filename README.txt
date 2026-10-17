# Netflix Movie Recommendation System using SVD
## First download data file (combined_data_1.txt) from link--> https://drive.google.com/file/d/1nUAwFYxnGiPhFOgP4tiXwOkaL8BgCMQG/view
## Required data
Place these files in the `data` folder:
- `combined_data_1.txt`: Netflix Prize ratings format (`movie_id:` header followed by `user_id,rating,date` rows).
- `movie_titles.txt`: matching Netflix Prize metadata, three fields per row: `movie_id,year,title`, with no header.

**Do not use a MovieLens file with `movieId,title,genres` as a substitute.** Matching numeric IDs alone does not establish compatibility.

## Run
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Put both matching dataset files into `data/`.
3. Open `netflix_project.py` in Python IDLE and press F5.

## Outputs
The script writes the sampled ratings, three EDA charts, model metrics (including a global-mean baseline), and top-10 recommendations into `output/`.

## Method
Reservoir sampling reads the entire ratings file but retains a reproducible sample of 100,000 ratings. The sample is split 80/20 for training/testing. SVD uses 50 factors, 20 epochs, learning rate 0.005, regularization 0.02, and random seed 42.
