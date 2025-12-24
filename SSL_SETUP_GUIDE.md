# SSL Setup Guide (Windows Server + Certbot)

This guide explains how to enable **HTTPS** for your Streamlit app using a free Let's Encrypt certificate.

## Prerequisites
- Domain Name: `insurancebot.scubeenterprise.com`
- Access to the Windows Server (Remote Desktop).
- Port 80 open (for verification) and Port 443 open (for secure traffic).

## Step 1: Open Ports in AWS and Windows Firewall

You need to open **Port 80** and **Port 443** just like you did for 8501.

### 1. AWS Security Group
1.  Go to AWS Console -> EC2 -> Security Groups.
2.  Edit Inbound Rules.
3.  Add Rule: **HTTP** (Port 80) -> Source: Anywhere (`0.0.0.0/0`).
4.  Add Rule: **HTTPS** (Port 443) -> Source: Anywhere (`0.0.0.0/0`).
5.  Save.

### 2. Windows Firewall (Inside the Server)
Open PowerShell as Administrator and run:

```powershell
New-NetFirewallRule -DisplayName "Allow HTTP" -Direction Inbound -LocalPort 80 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Allow HTTPS" -Direction Inbound -LocalPort 443 -Protocol TCP -Action Allow
```

## Step 2: Install Certbot on Windows

1.  Download the **Certbot installer** for Windows: [https://dl.eff.org/certbot-beta-installer-win32.exe](https://dl.eff.org/certbot-beta-installer-win32.exe)
2.  Run the installer. Default settings are fine.
3.  Open a **NEW** PowerShell window as Administrator (to load the new command).

## Step 3: Generate the Certificate

Run this command to get a certificate. Certbot will temporarily spin up a web server on Port 80 to prove you own the domain.

```powershell
certbot certonly --standalone -d insurancebot.scubeenterprise.com
```

*   Enter your email when asked.
*   Agree to terms (`Y`).
*   If successful, it will save keys to `C:\Certbot\live\insurancebot.scubeenterprise.com\`.

## Step 4: Configure Streamlit to use SSL

Streamlit needs to know where these files are.

1.  **Stop your running Streamlit app** (Ctrl+C).
2.  Create a folder `.streamlit` in your `ChatSQL` directory if it doesn't exist.
3.  Create a file `.streamlit/config.toml` (or edit it).
4.  Add these lines to `config.toml`:

```toml
[server]
sslCertFile = "C:\\Certbot\\live\\insurancebot.scubeenterprise.com\\fullchain.pem"
sslKeyFile = "C:\\Certbot\\live\\insurancebot.scubeenterprise.com\\privkey.pem"
port = 443
address = "0.0.0.0"
```

*Note: We are changing the port to 443 so users don't have to type :8501 anymore.*

## Step 5: Run the App

Run it as usual. Streamlit will pick up the config file automatically.

```powershell
streamlit run underwriting_ui_UseThis.py
```

Now accesses your app at: **https://insurancebot.scubeenterprise.com/** (No port needed!)

## Renewal
Let's Encrypt certs expire every 90 days. To renew, stop streamlit and run:
```powershell
certbot renew
```
Then start streamlit again.
