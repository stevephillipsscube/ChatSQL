# AWS Hosting Guide for ChatSQL (Windows Server)

This guide explains how to host your ChatSQL application on an AWS **Windows Server** EC2 instance.

## Prerequisites
- An AWS Account
- Your code pushed to GitHub (https://github.com/stevephillipsscube/ChatSQL)

## Step 1: Launch an EC2 Instance

1.  **Log in to AWS Console** and navigate to **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: `ChatSQL-Server`
4.  **OS Images**: Select **Windows** (Microsoft Windows Server 2022 Base).
5.  **Instance Type**: `t3.medium` (Recommended minimum).
6.  **Key Pair**: Create a new key pair (e.g., `chatsql-key-win`), or use an existing one. **Save the .pem file**, you will need it to retrieve your administrator password.
7.  **Network Settings**:
    *   Create security group.
    *   Allow **RDP** traffic from **My IP** (or Anywhere, but My IP is safer).
    *   Allow **HTTP/HTTPS** traffic from Internet.
8.  **Launch Instance**.

## Step 2: Configure Security Group (Open Port 8501)

1.  Go to your instance in the dashboard.
2.  Click the **Security (tab)** -> click the **Security Group** link.
3.  **Edit inbound rules**.
4.  Add Rule:
    *   **Type**: Custom TCP
    *   **Port range**: `8501`
    *   **Source**: `0.0.0.0/0` (Anywhere)
5.  Save rules.

## Step 3: Connect to your Instance

1.  Wait for the instance to initialize (Status Check: 2/2 passed).
2.  Select the instance and click **Connect**.
3.  Go to the **RDP client** tab.
4.  Click **Get password**.
5.  Upload your `.pem` key file and click **Decrypt Password**.
6.  Copy the **Administrator** password.
7.  Open **Remote Desktop Connection** on your local computer.
8.  Connect to the **Public DNS** of your instance.
9.  Log in as `Administrator` with the decrypted password.

## Step 4: Install Dependencies (On the Server)

*Once logged into the Remote Desktop:*

1.  **Install Python**:
    *   Open Edge or PowerShell.
    *   Download Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/).
    *   **IMPORTANT**: In the installer, check the box **"Add Python to PATH"**.
    *   Click **Install Now**.
    *   (Optionally disable path length limit if prompted).

2.  **Install Git**:
    *   Download Git for Windows from [git-scm.com/download/win](https://git-scm.com/download/win).
    *   Run the installer (Next, Next, Next...).

## Step 5: Deploy Application

1.  Open **PowerShell** (Search "PowerShell" in Start Menu).

2.  **Clone the Repository**:
    ```powershell
    cd C:\Users\Administrator\Documents
    git clone https://github.com/stevephillipsscube/ChatSQL.git
    cd ChatSQL
    ```

3.  **Install Python Libraries**:
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    *   Create a `.env` file in the folder.
    *   *Tip: You can install Notepad++ or VS Code for easier editing, or just use regular Notepad.*
    *   Example: `notepad .env` -> Paste keys -> Save.

## Step 6: Configure Windows Firewall

**Crucial Step**: Even though you opened port 8501 in AWS, **Windows Firewall** inside the server blocks it by default.

1.  Open **PowerShell** as Administrator.
2.  Run this command to allow traffic on port 8501:
    ```powershell
    New-NetFirewallRule -DisplayName "Allow Streamlit" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
    ```

## Step 7: Run the Application

### Option A: Direct Run
```powershell
streamlit run underwriting_ui_UseThis.py
```
*Visit `http://<YOUR_INSTANCE_IP>:8501` in your browser.*

### Option B: Keep Running (Background)
PowerShell doesn't have `nohup` like Linux. To keep it running:

1.  Run the command as usual.
2.  **Do not sign out**. Disconnect the RDP session (click X) – your session stays active.
3.  For a robust solution, consider using **NSSM** (Non-Sucking Service Manager) to run it as a Windows Service.
