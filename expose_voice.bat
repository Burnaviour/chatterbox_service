@echo off
echo ==================================================
echo       Exposing Voice Service to Network
echo ==================================================
echo.
echo Your IP Address(es):
ipconfig | findstr "IPv4"
echo.
echo --------------------------------------------------
echo Target: Local Port 8000 (Voice Service)
echo Listen: Port 8001 (Exposed)
echo.
echo Access URL: http://<YOUR_IP>:8001
echo --------------------------------------------------
echo.
"C:\Users\Burnaviour\.local\bin\caddy.exe" reverse-proxy --from :8001 --to :8000
pause
