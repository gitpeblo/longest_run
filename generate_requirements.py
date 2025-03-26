import subprocess

def get_user_installed_packages():
    result = subprocess.run(
        ["pip", "list", "--not-required", "--format=freeze"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip().splitlines()

if __name__ == "__main__":
    pkgs = get_user_installed_packages()
    with open("requirements.txt", "w") as f:
        f.write("\n".join(pkgs) + "\n")
    print(f"✅ requirements.txt written with {len(pkgs)} user-installed packages.")

