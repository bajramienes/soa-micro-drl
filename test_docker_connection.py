import docker

try:
    client = docker.from_env()
    containers = client.containers.list(all=True)
    print(f"Docker is accessible. Found {len(containers)} containers.")
    for c in containers:
        print(f" - {c.name} (Status: {c.status})")
except Exception as e:
    print(f"Docker access failed: {e}")
