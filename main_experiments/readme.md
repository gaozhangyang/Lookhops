1. use `generate_json.ipynb` to get `config.json`
2. use `nnictl create -c config.yml` to run experiments
3. use `nnictl experiment export $experiment_name$ -f ./csv/XXX.csv --type "csv"` to save results.