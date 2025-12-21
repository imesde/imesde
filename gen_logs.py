import time
import random
import sys

templates = [
    "INFO: User {user} logged in from {ip}",
    "ERROR: Connection reset by peer on port {port}",
    "WARN: Disk usage at {percent}% on /dev/sda1",
    "DEBUG: Memory heap size at {mem}MB",
    "CRITICAL: Service {service} is unreachable",
    "INFO: File {file}.txt uploaded successfully by {user}",
    "ERROR: Database query failed: syntax error at {port}"
]

users = ["admin", "guest", "root", "dev_user"]
services = ["auth_db", "api_gateway", "worker_pool"]

def generate():
    try:
        while True:
            t = random.choice(templates).format(
                user=random.choice(users),
                ip=f"192.168.1.{random.randint(1, 254)}",
                port=random.randint(1024, 65535),
                percent=random.randint(80, 99),
                mem=random.randint(100, 4000),
                service=random.choice(services),
                file=random.randint(100, 999)
            )
            print(t, flush=True)
            time.sleep(0.1) # Un log ogni 100ms
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    generate()