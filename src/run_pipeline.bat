@echo off

python src\data_prep.py

echo ############ myTabddpm (identity): uci_german ############
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_german\identity.toml --train --sample
echo ############ myTabddpm (identity): uci_taiwan ############
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_taiwan\identity.toml --train --sample
echo ############ myTabddpm (identity): hmeq ############
python src\myTabddpm\pipeline.py --config configuration\datasets\hmeq\identity.toml --train --sample

echo ############ myTabddpm (identity): gmsc ############
@REM python src\myTabddpm\pipeline.py --config configuration\datasets\gmsc\identity.toml --train --sample
echo ############ myTabddpm (identity): pakdd ############
@REM python src\myTabddpm\pipeline.py --config configuration\datasets\pakdd\identity.toml --train --sample


echo ############ myTabddpm (bgm): uci_german ############
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_german\bgm.toml --train --sample
echo ############ myTabddpm (bgm): uci_taiwan ############
python src\myTabddpm\pipeline.py --config configuration\datasets\uci_taiwan\bgm.toml --train --sample
echo ############ myTabddpm (bgm): hmeq ############
python src\myTabddpm\pipeline.py --config configuration\datasets\hmeq\bgm.toml --train --sample

echo ############ myTabddpm (bgm): gmsc ############
@REM python src\myTabddpm\pipeline.py --config configuration\datasets\gmsc\bgm.toml --train --sample
echo ############ myTabddpm (bgm): pakdd ############
@REM python src\myTabddpm\pipeline.py --config configuration\datasets\pakdd\bgm.toml --train --sample


echo All tasks completed!
pause
