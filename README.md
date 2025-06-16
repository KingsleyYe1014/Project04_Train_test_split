# Project 4: Census Data Analysis and Modeling

This project analyzes census data using machine learning models to predict household income based on various demographic factors.

## Setup

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Get a Census API key:
   - Go to https://api.census.gov/data/key_signup.html
   - Sign up for an API key
   - Create a file named `config.py` in the project root
   - API key as: `CENSUS_API_KEY = "a52e53ddb8b40f8b0a7b15c459ce26c5c902313c"`

## Project Structure

- `main.py`: Main analysis script
- `config.py`: API key configuration (gitignored)
- `requirements.txt`: Required Python packages
- `report.md`: Project report and findings

## Data Source

This project uses data from the U.S. Census Bureau API, specifically:
- American Community Survey (ACS) 5-year estimates
- Variables include: household income, education level, employment status, etc.

## Model

The project implements both regression and decision tree models to predict household income based on demographic factors. 
