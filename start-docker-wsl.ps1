# start-docker-wsl.ps1
Write-Host "🚀 Starting Ubuntu and Docker WSL distros..."

# Start Ubuntu
wsl -d Ubuntu --exec echo "Ubuntu started ✅"

# Start docker-desktop backend
wsl -d docker-desktop --exec echo "Docker Desktop backend started ✅"

# Small wait to let Docker Engine initialize
Start-Sleep -Seconds 5

# Check docker status
docker version