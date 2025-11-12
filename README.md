# adv-csv-parser
An Advanced CSV Analysis, Paresing and Extraction Utility

## Description

An Advanced CSV analyzer/parser/extractor with an extensible CLI. Features:
- Efficient parsing of arbitrary CSV files (handles dialects, large files)
- Streaming processing to minimize memory usage
- Modular CLI using argparse with subcommands
- Type hints, comprehensive docstrings, PEP 8 compliant
- Robust error handling and resource management
- Extensible design for future processors

## Prerequisites

Requires Python 3.x (preferrably 3.11+) and uses the following libraries:
* annotations (future)
* argparse
* csv
* sys
* abc
* pathlib
* typing
* collections.abc

## How to Use

To analyze a target CSV file:
```
# ./python adv-csv-parser.py info path/to/file.csv
```

To perform a recond count on a target CSV file:
```
# ./python adv-csv-parser.py count path/to/file.csv
```

To see examplar data within a target CSV file:
```
# ./python adv-csv-parser.py head path/to/file.csv --lines 10
```

To selectively display data by field name from a target CSV file:
```
# ./python adv-csv-parser.py select path/to/file.csv Name Age --delimiter ','
```

## Built With

* [Python](https://www.python.org) designed by Guido van Rossum

## Author

**Rick Pelletier** - [Gannett Co., Inc. (USA Today Network)](https://www.usatodayco.com/)
