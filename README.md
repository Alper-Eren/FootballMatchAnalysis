# Football Match Analysis

This repository contains code for analyzing football matches using a Random Forest classifier and regressor. The code predicts the match outcome and the total number of goals based on historical match data.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- pandas
- scikit-learn
- matplotlib
- numpy

You can install these dependencies by running the following command:
```
pip install pandas scikit-learn matplotlib numpy
```

## Installation

To use this code, follow these steps:

1. Clone this repository to your local machine using `git clone https://github.com/your-username/football-match-analysis.git`.
2. Navigate to the repository's directory: `cd football-match-analysis`.

## Usage

1. Place your dataset file named `dataset_results.csv` in the repository's directory.
2. Open the Python file `football_analysis.py` in your preferred development environment.
3. Modify the code as needed (e.g., change the number of estimators, add additional features).
4. Run the code: `python football_analysis.py`.
5. Follow the prompts to enter the home team and away team for analysis.
6. The code will predict the match outcome probabilities and the total number of goals.
7. The predicted probabilities will be displayed as a bar chart.
8. The predicted number of goals will be displayed as a bar chart.

## Results

The code will display the following results:

- Match Outcome Probabilities: The predicted probabilities for different match outcomes (e.g., home win, away win, draw).
- Total Number of Goals Prediction: The average goals per match for the given teams and the predicted number of goals for the analyzed match.

## Contributing

If you want to contribute to this project, you can follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-feature`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push the changes to your fork: `git push origin feature/my-feature`.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to customize the README file based on your needs and preferences.

