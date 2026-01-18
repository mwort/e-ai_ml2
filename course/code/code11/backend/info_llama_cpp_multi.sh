echo "Infos about llama cpp instances"
printf '%*s\n' 80 '' | tr ' ' '-'

ps -ef | grep llama-server | grep -v grep

#!/bin/bash
printf '%*s\n' 80 '' | tr ' ' '-'

echo "🔍 Checking llama-server CPU usage per instance:"
echo

# List all llama-server PIDs
pids=$(pgrep -f "llama-server")

if [ -z "$pids" ]; then
    echo "❌ No running llama-server instances found."
    exit 1
fi

for pid in $pids; do
    # Get the port from the command line
    cmd=$(ps -p $pid -o args=)
    port=$(echo "$cmd" | grep -oP '(?<=--port )\d+')

    # Get CPU affinity
    cpus=$(taskset -cp "$pid" | awk -F: '{print $2}' | xargs)

    echo "🧠 PID $pid is running on port $port with CPUs: $cpus"
done

printf '%*s\n' 80 '' | tr ' ' '-'
