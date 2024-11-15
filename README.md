# Score-Deep

## Overview
`score-deep` is a project focused on deep learning techniques for synthetic data generation and evaluation. It integrates advanced diffusion models and deep architectures for benchmarking on imbalanced credit scoring datasets.

## Project Structure
- **Path**: `C:\Users\ASUS\Documents\University\MSc\__Thesis__\__Project__\score-deep`
- **Environment**: Activate the required Python environment from the following path:
  ```
  source "C:\Users\ASUS\Documents\Extra Courses\ML-DL\env\Scripts\activate"
  ```

## Scripts and Usage
### Custom TabDDPM Implementation
Run the following commands to execute the pipeline for different datasets:

#### Dataset preparation
```
python src/data_prep.py
```

#### UCI German Dataset
```
python src/myTabddpm/pipeline.py --config configuration/datasets/uci_german/config.toml --train --sample
```

#### HMEQ Dataset
```
python src/myTabddpm/pipeline.py --config configuration/datasets/hmeq/config.toml --train --sample
```

#### GMSC Dataset
```
python src/myTabddpm/pipeline.py --config configuration/datasets/gmsc/config.toml --train --sample
```

#### PAKDD Dataset
```
python src/myTabddpm/pipeline.py --config configuration/datasets/pakdd/config.toml --train --sample
```

#### UCI Taiwan Dataset
```
python src/myTabddpm/pipeline.py --config configuration/datasets/uci_taiwan/config.toml --train --sample
```

