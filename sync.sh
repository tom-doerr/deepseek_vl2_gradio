#!/bin/bash

# Sync files
rsync -avz --progress . conic:~/${PWD##*/}/

# Set up port forwarding in the background
ssh -N -L 7860:localhost:7860 conic &

# Store the SSH process ID
echo $! > .port_forward.pid

echo "Port forwarding set up. Access the Gradio interface at http://localhost:7860"
echo "To stop port forwarding, run: kill \$(cat .port_forward.pid) && rm .port_forward.pid"
