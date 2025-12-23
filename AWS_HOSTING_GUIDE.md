# AWS Hosting Guide for ChatSQL

This guide explains how to host your ChatSQL application on an AWS EC2 instance.

## Prerequisites
- An AWS Account
- Your code pushed to GitHub (https://github.com/stevephillipsscube/ChatSQL)

## Step 1: Launch an EC2 Instance

1.  **Log in to AWS Console** and navigate to **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: `ChatSQL-Server`
4.  **OS Images**: Select **Ubuntu** (Ubuntu Server 24.04 LTS or 22.04 LTS).
5.  **Instance Type**: `t3.medium` (Recommended: t2.micro might be too small for chroma/duckdb/streamlit).
6.  **Key Pair**: Create a new key pair (e.g., `chatsql-key`), download the `.pem` file.
7.  **Network Settings**:
    *   Create security group.
    *   Allow SSH traffic from **My IP**.
    *   Allow HTTP/HTTPS traffic from Internet.
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

1.  Open your terminal (or PowerShell).
2.  Move your key file to a safe place.
3.  Connect using SSH:
    ```bash
    ssh -i "path/to/chatsql-key.pem" ubuntu@<YOUR_INSTANCE_PUBLIC_IP>
    ```

## Step 4: Install Dependencies

Run these commands on the server to install Python, pip, and Git:

```bash
# Update package list
sudo apt-get update

# Install Python and pip
sudo apt-get install -y python3-pip python3-venv git

# (Optional) Install Docker if you want to use the Dockerfile
# sudo apt-get install -y docker.io
```

## Step 5: Deploy Application

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/stevephillipsscube/ChatSQL.git
    cd ChatSQL
    ```

2.  **Create Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create the `.env` file on the server.
    ```bash
    nano .env
    ```
    *Paste your `.env` content (API Keys, etc) here. Press Ctrl+O, Enter, Ctrl+X to save.*

## Step 6: Run the Application

### Option A: Direct Run (Testing)
```bash
streamlit run underwriting_ui_UseThis.py
```
*Visit `http://<YOUR_INSTANCE_IP>:8501` in your browser.*

### Option B: Run in Background (Production)
To keep the app running after you close SSH:

```bash
nohup streamlit run underwriting_ui_UseThis.py --server.port 8501 &
```

*To stop it later:* `pkill streamlit`
