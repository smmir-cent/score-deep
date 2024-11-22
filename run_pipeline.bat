@echo off

REM Dataset preparation
python src\data_prep.py

REM UCI German Dataset
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_german\config.toml --train --sample

REM HMEQ Dataset
python src\myTabddpm\pipeline.py --config configuration\datasets\hmeq\config.toml --train --sample

REM GMSC Dataset
python src\myTabddpm\pipeline.py --config configuration\datasets\gmsc\config.toml --train --sample

REM PAKDD Dataset
python src\myTabddpm\pipeline.py --config configuration\datasets\pakdd\config.toml --train --sample

REM UCI Taiwan Dataset
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_taiwan\config.toml --train --sample

echo All tasks completed!
pause
