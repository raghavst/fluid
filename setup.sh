python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
rsync -a flwr_changed_files/ ./venv/lib/python3.10/site-packages/flwr/server/strategy/