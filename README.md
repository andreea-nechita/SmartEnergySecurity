# SmartEnergySecurity
**SmartEnergySecurity** is a dual-purpose tool, developed as part of a coursework project. Simple to use, yet effective, it (currently) provides two of the must-have functionalities of your smart meter:
* Smart scheduling of your daily electricity-wise tasks,
* Detection of manipulated pricing.

**SmartEnergySecurity** is 100% built in Python and run via the command-line interface (CLI), which means the installation and deployment are straightforward. Plus, it comes with all the required resources (apart from external dependencies). Set it up in a matter of seconds and you're ready to use it!

## Installation

## Usage
The main runner script is [`sec.py`](smart-energy-security/sec.py) (abbreviation of SmartEnergySecurity), which can be used in `--detect` (`-D`), `--schedule`(`-S`) or both of them.

For more detailed information about the accepted script parameters, run `python sec.py -h` or `python sec.py --help`. This provides the entire list of arguments and a descriptive text for each of them.

### `--detect`
This mode of operation is used for detecting abnormal pricing curves. It takes a file(path) as argument and creates a CSV-formatted file named `TestingResults.txt`, containing the input price data and a final column with the predicted label (`0` for normal and `1` for abnormal). It also accepts an external Keras model and an external `sklearn.preprocessing` data scaler (saved as a `pickle` file) to replace the default ones.

A sample [input file](resources/TestingData.txt) can be found in the `resources` directory.

#### Example of using detection mode
`python sec.py --detect -d TestingData.txt` or `python sec.py -D --data TestingData.txt`

### `--schedule`
The scheduling mode implements task scheduling based on the guideline price and scheduling requirements (i.e. ready time, deadline, maximum scheduled energy per hour, energy demand). For each set of guideline prices in the input file, a bar chart figure is saved in the `out` directory, which corresponds to the computed scheduling plan. The indeces of the pricing curves (their row numbers in the input file) are employed in naming the figure files.

Adding the optional argument `--solutions` enables saving the scheduling results in `.csv` files (one output file for each guideline price in the input).

If detection is performed during the same run, the pricing curves used as input in the classification are used in scheduling. In this case, mentioning another input file for pricing information through the `--pricing` (`-p`) argument has no effect.

The `--label` (`-l`) argument can be used in order to specify which pricing curves to be considered in scheduling the tasks: `normal`, `abnormal`, `all`, or `none` (if the pricing curves are not labelled). The default option is `all`, which implies that scheduling is performed for all pricing curves and these guideline prices ARE LABELLED.

A sample [file](resources/COMP3217CW2Input.xlsx) containing the scheduling requirements is in the `resources` directory. The file of pricing guidelines must have the same format as the input file (if unlabelled) or output file (if labelled) for detection mode 

#### Example of using scheduling mode
* `python sec.py --schedule -r input.xlsx --pricing TestingData.txt` (in this case, the guideline prices must be labelled)
* `python sec.py -S --requirements input.xlsx -p TestingData.txt --label none` (but here, `input.xlsx` does not have a `label` column)

If both `--detect` and `--scheduling` options are passed, the input price curves are first classified, then used for scheduling. The `--label` (`-l`) argument can be used in order to specify which pricing curves to be considered in scheduling the tasks: `normal`, `abnormal`, `all`, or `none` (if the pricing curves are not labelled).
