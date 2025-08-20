# Deployment Guide

This guide provides instructions on how to deploy the Mango Tree Location API to a production environment. We will cover two common scenarios:
1.  **Deploying to a Virtual Private Server (VPS)**
2.  **Deploying to Heroku (PaaS)**

## General Considerations

Before deploying, ensure you have:
-   A production-ready database (e.g., MongoDB Atlas).
-   Your Cloudinary credentials.
-   Set all required environment variables in your production environment. **Do not hardcode them.**

For production, it is recommended to use a robust WSGI server like Gunicorn to run the FastAPI application.

## 1. Deploying to a Virtual Private Server (VPS)

This guide assumes you have a VPS running a Linux distribution like Ubuntu, with root or sudo access.

### Step 1: Install Gunicorn

First, add Gunicorn to your `requirements.txt` file and install it in your virtual environment.

```bash
# Add this line to requirements.txt
gunicorn

# Install it
pip install -r requirements.txt
```

### Step 2: Run the App with Gunicorn

You can run your FastAPI application using Gunicorn with Uvicorn's worker class. This provides a battle-tested, production-ready server setup.

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4 -b 0.0.0.0:8000
```
-   `-k uvicorn.workers.UvicornWorker`: Specifies the Uvicorn worker class.
-   `main:app`: Tells Gunicorn where to find the FastAPI `app` instance.
-   `-w 4`: The number of worker processes. A good starting point is `(2 * number_of_cpu_cores) + 1`.
-   `-b 0.0.0.0:8000`: The address and port to bind to.

### Step 3: Set up a Process Manager (systemd)

To ensure your application runs continuously and restarts automatically on failure or server reboot, use a process manager like `systemd`.

1.  Create a service file:
    ```bash
    sudo nano /etc/systemd/system/mango_api.service
    ```

2.  Add the following content. Make sure to replace placeholders like `/path/to/your/project` and `your_user` with your actual paths and username.

    ```ini
    [Unit]
    Description=Mango Tree Location API
    After=network.target

    [Service]
    User=your_user
    Group=www-data
    WorkingDirectory=/path/to/your/project
    EnvironmentFile=/path/to/your/project/.env
    ExecStart=/path/to/your/project/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4 -b 0.0.0.0:8000

    [Install]
    WantedBy=multi-user.target
    ```

3.  Enable and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl start mango_api
    sudo systemctl enable mango_api
    ```

### Step 4: Set up a Reverse Proxy (Nginx)

It's a best practice to run a web server like Nginx in front of your application. Nginx can handle incoming traffic, serve static files, and manage SSL/TLS certificates.

1.  Install Nginx:
    ```bash
    sudo apt update
    sudo apt install nginx
    ```

2.  Create a new Nginx configuration file:
    ```bash
    sudo nano /etc/nginx/sites-available/mango_api
    ```

3.  Add the following configuration, replacing `your_domain.com` with your actual domain name.

    ```nginx
    server {
        listen 80;
        server_name your_domain.com;

        location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
    ```

4.  Enable the site and restart Nginx:
    ```bash
    sudo ln -s /etc/nginx/sites-available/mango_api /etc/nginx/sites-enabled
    sudo nginx -t  # Test configuration
    sudo systemctl restart nginx
    ```
You can also use Certbot to easily set up HTTPS for your domain.

## 2. Deploying to Heroku

Heroku is a Platform-as-a-Service (PaaS) that simplifies deployment.

### Step 1: Create a `Procfile`

Create a file named `Procfile` (with no extension) in the root of your project directory. This file tells Heroku how to run your application.

```
web: gunicorn -k uvicorn.workers.UvicornWorker main:app
```
Make sure you have added `gunicorn` to your `requirements.txt` file.

### Step 2: Install the Heroku CLI and Log In

Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli). Then, log in to your Heroku account:
```bash
heroku login
```

### Step 3: Create a Heroku App

Create a new app on Heroku from your terminal:
```bash
heroku create your-app-name
```
If you don't provide a name, Heroku will generate one for you.

### Step 4: Set Environment Variables

Set your environment variables (MongoDB URI, Cloudinary credentials) in the Heroku app's settings. You can do this via the Heroku dashboard or the CLI:

```bash
heroku config:set MONGODB_URI="your_mongodb_connection_string"
heroku config:set CLOUDINARY_CLOUD_NAME="your_cloudinary_cloud_name"
heroku config:set CLOUDINARY_API_KEY="your_cloudinary_api_key"
heroku config:set CLOUDINARY_API_SECRET="your_cloudinary_api_secret"
```

### Step 5: Deploy to Heroku

Commit your changes (including the `Procfile`) and push your code to Heroku:

```bash
git add .
git commit -m "docs: Add Procfile for Heroku deployment"
git push heroku main
```
(Note: Depending on your local branch name, you might need to use `git push heroku your-branch-name:main`).

Your application is now deployed! You can open it in your browser with `heroku open`.
