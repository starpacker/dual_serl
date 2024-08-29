# start ros and kill prev processes
pkill -SIGINT -f roscore
sleep 3
roscore &

# Debugging information
echo "Stopping previous leaphand_node.py process..."
pkill -2 -f leaphand_node.py

# Initialize Conda
echo "Initializing Conda..."
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the environment and run leaphand_node.py
echo "Activating the base Conda environment..."
conda activate base


# Wait a few seconds to ensure pkill ends
sleep 2

echo "Running leaphand_node.py..."
rosrun leap_hand leaphand_node.py &

# Wait a few seconds to ensure leaphand_node.py starts
sleep 2

# Activate the 'serl' environment and run the server
echo "Activating the 'serl' Conda environment..."
conda activate serl

# activate robot server
echo "Running robot_server/server.py..."
python robot_server/server.py




