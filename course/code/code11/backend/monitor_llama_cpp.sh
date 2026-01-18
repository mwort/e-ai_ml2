#!/usr/bin/env python3
import time
import psutil

monitored_ports = [8080, 8081, 8082, 8083, 8084, 8085]

def find_llama_processes():
    llama_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if not cmdline:
                continue
            cmd = " ".join(cmdline).lower()
            if "llama" in cmd:
                for port in monitored_ports:
                    if str(port) in cmd:
                        llama_processes.append((port, proc))
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return llama_processes

def monitor_loop(interval=2):
    header = f"{'':<2}{'Port':<6} {'PID':<6} {'CPU %':<6} {'Memory MB':<10}"
    blink = True

    while True:
        llama_procs = find_llama_processes()
        num_lines = len(llama_procs) + 3  # header + separator + process lines

        # Move cursor up to overwrite all lines
        for _ in range(num_lines):
            print("\x1b[F", end='')

        print(header)
        print("-" * len(header))

        for port, proc in llama_procs:
            try:
                cpu = proc.cpu_percent(interval=0.1)
                mem = proc.memory_info().rss / (1024 * 1024)
                dot = "●" if blink else " "
                line = f"{dot:<2}{port:<6} {proc.pid:<6} {cpu:<6.1f} {mem:<10.1f}"
            except psutil.NoSuchProcess:
                dot = "●" if blink else " "
                line = f"{dot:<2}{port:<6} {'-':<6} {'-':<6} {'-':<10}"
            print(f"\r{line}")

        print("-" * len(header), end='\n', flush=True)
        blink = not blink
        time.sleep(interval)

if __name__ == "__main__":
    # Print initial empty space so the first update has room to move up
    print("\n" * 8)
    monitor_loop()
