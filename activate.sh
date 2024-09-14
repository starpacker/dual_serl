# Initialize Conda
echo "Initializing Conda..."
source ~/Miniconda3/etc/profile.d/conda.sh

sleep 1

# Activate the environment and run leaphand_node.py
echo "Activating the base Conda environment..."
conda activate base

sleep 1

# Activate the 'serl' environment and run the server
echo "Activating the 'dual_serl' Conda environment..."
conda activate dual_serl

sleep 1

# activate robot server
echo "Running robot_server flexiv_server.py..."
python serl_robot_infra/robot_servers/flexiv_server.py




