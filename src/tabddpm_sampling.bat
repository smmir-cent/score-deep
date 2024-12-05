@echo off

echo Starting script at: %date% %time%

python src\data_prep.py
echo Finished data prep at: %date% %time%

echo ######################### myTabddpm (identity): uci_german #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_german\identity.toml --train --sample
echo Finished uci_german (identity) at: %date% %time%

echo ######################### myTabddpm (identity): uci_taiwan #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_taiwan\identity.toml --train --sample
echo Finished uci_taiwan (identity) at: %date% %time%

echo ######################### myTabddpm (identity): hmeq #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\hmeq\identity.toml --train --sample
echo Finished hmeq (identity) at: %date% %time%

echo ######################### myTabddpm (identity): gmsc #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\gmsc\identity.toml --train --sample
echo Finished gmsc (identity) at: %date% %time%

echo ######################### myTabddpm (identity): pakdd #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\pakdd\identity.toml --train --sample

echo Finished pakdd (identity) at: %date% %time%

echo ######################### myTabddpm (bgm): uci_german #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_german\bgm.toml --train --sample
echo Finished uci_german (bgm) at: %date% %time%

echo ######################### myTabddpm (bgm): uci_taiwan #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_taiwan\bgm.toml --train --sample
echo Finished uci_taiwan (bgm) at: %date% %time%

echo ######################### myTabddpm (bgm): hmeq #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\hmeq\bgm.toml --train --sample
echo Finished hmeq (bgm) at: %date% %time%

echo ######################### myTabddpm (bgm): gmsc #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\gmsc\bgm.toml --train --sample
echo Finished gmsc (bgm) at: %date% %time%

echo ######################### myTabddpm (bgm): pakdd #########################
python src\myTabddpm\pipeline.py --config configuration\datasets\pakdd\bgm.toml --train --sample
echo Finished pakdd (bgm) at: %date% %time%

echo ######################### PYTHON SRC/MAIN.PY #########################
python src\main.py
echo Finished python src\main.py at: %date% %time%


echo All tasks completed at: %date% %time%
pause
