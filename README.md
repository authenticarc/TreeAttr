## TreeAttr

TreeAttr leverages LightGBM to attribute changes in data, facilitating the understanding of underlying factors driving variations in datasets.

### Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Swap Data Attribution](#swap-data-attribution)
    - [Futures User Attribution](#futures-user-attribution)
    - [New Device Attribution](#new-device-attribution)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)

### Overview
TreeAttr uses LightGBM to attribute changes in various data sets, aiding in identifying key factors that contribute to data variations in scenarios like swaps, futures trading, and new device registrations.

### Installation
To install TreeAttr, clone the repository and install the dependencies:

```bash
git clone https://github.com/authenticarc/TreeAttr.git
cd TreeAttr
pip install -r requirements.txt
```

### Usage
TreeAttr provides scripts for different attribution tasks. Below are examples of how to run these tasks.

#### Swap Data Attribution
For swap data attribution:

```bash
python3 scripts/reg.py --config config/swap_amt.yaml | grep -v "LightGBM"
python3 scripts/cls.py --config config/swap_user.yaml | grep -v "LightGBM"
```

#### Futures User Attribution
For futures user attribution:

```bash
python3 scripts/cls.py --config config/futures_user.yaml | grep -v "LightGBM"
python3 scripts/reg.py --config config/futures_amt.yaml | grep -v "LightGBM"
```

#### New Device Attribution
For new device attribution:

```bash
python3 scripts/cls.py --config config/new_imei.yaml | grep -v "LightGBM"
```

### Configuration
Configurations are provided in YAML files within the `config` directory. Each task has a corresponding YAML configuration file that specifies the parameters for the task, such as data paths, feature columns, and target variables.

### Contributing
Contributions are welcome. Please fork the repository and submit pull requests for any improvements.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For more details, visit the [TreeAttr GitHub repository](https://github.com/authenticarc/TreeAttr).
