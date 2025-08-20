# Contributing to the Mango Tree Location API

First off, thank you for considering contributing to this project! We welcome contributions from everyone, especially those new to open source. This guide will walk you through the process of making your first contribution.

## How to Contribute

We use the standard GitHub Fork & Pull Request workflow. If you're new to this, don't worry! Here's a step-by-step guide.

### 1. Fork the Repository

-   Go to the project's GitHub page: [https://github.com/alumnx-ai-labs/image_classification_backend_fastapi](https://github.com/alumnx-ai-labs/image_classification_backend_fastapi)
-   Click the **Fork** button in the top-right corner. This will create a copy of the repository in your own GitHub account.

### 2. Clone Your Fork

-   On your GitHub account, navigate to your forked repository.
-   Click the **Code** button and copy the HTTPS or SSH URL.
-   Open your terminal and run the following command to clone the repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/image_classification_backend_fastapi.git
    cd image_classification_backend_fastapi
    ```
    (Replace `YOUR_USERNAME` with your actual GitHub username.)

### 3. Create a New Branch

It's important to create a new branch for each feature or bug fix you work on. This keeps your changes organized and separate from the main codebase.

-   Make sure you are on the `dev` branch and have the latest changes from the main repository.
    ```bash
    # Optional: Configure a remote for the original repository
    git remote add upstream https://github.com/alumnx-ai-labs/image_classification_backend_fastapi.git

    # Fetch the latest changes
    git fetch upstream

    # Checkout the dev branch
    git checkout dev

    # Make sure your dev branch is up-to-date
    git pull upstream dev
    ```

-   Now, create your new branch. Choose a descriptive name, like `feature/add-new-endpoint` or `fix/typo-in-readme`.
    ```bash
    git checkout -b your-branch-name
    ```

### 4. Make Your Changes

Now you can start making your changes to the code! Add new features, fix bugs, or improve the documentation.

### 5. Commit Your Changes

Once you're happy with your changes, you need to commit them. A good commit message is important. We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

-   Stage your changes:
    ```bash
    git add .
    ```
-   Commit them with a descriptive message:
    ```bash
    git commit -m "feat: Add a new feature that does something"
    # or
    git commit -m "fix: Correct a bug in the proximity check"
    # or
    git commit -m "docs: Update the README with new instructions"
    ```

### 6. Push Your Changes

Push your new branch and its commits to your forked repository on GitHub.

```bash
git push origin your-branch-name
```

### 7. Open a Pull Request

You're now ready to open a pull request!

-   Go to your forked repository on GitHub.
-   You should see a prompt to create a pull request from your new branch. Click on it.
-   Make sure the **base repository** is `alumnx-ai-labs/image_classification_backend_fastapi` and the **base branch** is `dev`.
-   The **head repository** should be your fork, and the **compare branch** should be `your-branch-name`.
-   Give your pull request a clear title and a detailed description of the changes you've made.
-   Click **Create pull request**.

Our team will review your pull request, provide feedback, and merge it once it's ready. Congratulations on making your first contribution!

## Questions?

If you have any questions or get stuck, feel free to open an issue on the repository. We're here to help!
