python3 scripts/reg.py --config config/swap_amt.yaml | grep -v "LightGBM" >> results/20240512.md
python3 scripts/cls.py --config config/swap_user.yaml | grep -v "LightGBM" >> results/20240512.md
python3 scripts/reg.py --config config/futures_amt.yaml | grep -v "LightGBM" >> results/20240512.md
python3 scripts/cls.py --config config/futures_user.yaml | grep -v "LightGBM" >> results/20240512.md
python3 scripts/cls.py --config config/new_imei.yaml | grep -v "LightGBM" >> results/20240512.md