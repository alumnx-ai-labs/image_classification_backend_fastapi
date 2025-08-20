# Mango Tree Location API

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Cloudinary](https://img.shields.io/badge/Cloudinary-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)

A FastAPI-based backend for classifying and managing mango tree images and their geographical locations. This API allows for image uploads, metadata storage, and proximity analysis to identify duplicate entries.

This project is designed to be beginner-friendly and easy to deploy. For more detailed guides, please see our [documentation](#documentation).

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Application](#running-the-application)
- [Features](#features)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [Deployment](#deployment)
- [License](#license)

## Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Pip](https://pip.pypa.io/en/stable/installation/)
- A [MongoDB](https://www.mongodb.com/try/download/community) database instance (local or cloud-hosted).
- A [Cloudinary](https://cloudinary.com/users/register/free) account for image hosting.

### Installation

1.  **Fork the repository**
    Click the "Fork" button at the top-right of this page to create your own copy of this repository.

2.  **Clone your fork**
    ```bash
    git clone https://github.com/YOUR_USERNAME/image_classification_backend_fastapi.git
    cd image_classification_backend_fastapi
    ```

3.  **Create a virtual environment and activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

This project uses environment variables for configuration. Create a `.env` file in the root of the project directory and add the following variables:

```
# .env file
MONGODB_URI="your_mongodb_connection_string"
CLOUDINARY_CLOUD_NAME="your_cloudinary_cloud_name"
CLOUDINARY_API_KEY="your_cloudinary_api_key"
CLOUDINARY_API_SECRET="your_cloudinary_api_secret"
```

Replace the placeholder values with your actual credentials from MongoDB and Cloudinary.

### Running the Application

Once the dependencies are installed and the environment variables are set, you can run the application using Uvicorn:

```bash
uvicorn main:app --reload
```

The `--reload` flag enables hot-reloading, which is useful for development. The API will be available at `http://127.0.0.1:8000`.

## Features

-   **Image Upload**: Upload images directly to the backend or to Cloudinary.
-   **Metadata Storage**: Save image metadata, including location and predictions, to MongoDB.
-   **Proximity Analysis**: Check for duplicate images based on GPS coordinates.
-   **Farm Management**: Group plants into farms and manage their data.
-   **Secure Endpoints**: Generate signatures for secure client-side uploads to Cloudinary.

## API Documentation

Once the application is running, you can access the interactive API documentation (provided by FastAPI and Swagger UI) at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

This interface allows you to explore and test all the API endpoints directly from your browser.

## Documentation

For more detailed guides, please refer to the following documents:
- [**Contributing Guide**](./CONTRIBUTING.md)
- [**Deployment Guide**](./docs/DEPLOYMENT.md)

## Contributing

We welcome contributions from everyone! If you're a first-time contributor, this repository is a great place to start. Please read our [**CONTRIBUTING.md**](./CONTRIBUTING.md) for detailed instructions on how to submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
