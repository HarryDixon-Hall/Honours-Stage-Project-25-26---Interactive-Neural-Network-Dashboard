# Deployment Guide

## Recommended environment

Use a Linux container host for production. The best fit for this project as it stands is Google Cloud Run running the Docker image built from this repository.

Why this is the best fit:

- Dash and Flask deploy cleanly on Linux with Gunicorn.
- The app is effectively stateless today, so it does not need Kubernetes or a more complex platform yet.
- Cloud Run stays simple for one public web service, but still leaves room for extra services, jobs, secrets, and databases later.
- GitHub Actions integrates cleanly with Google Cloud through Workload Identity Federation, so you do not need to store a long-lived JSON key in GitHub.
- The same Docker image can later move to GKE, Azure, AWS, Fly.io, or Render if the project grows.

## Identity and services model

Google Cloud and Azure can both scale to roughly the same level of system complexity. The ceiling is not the main difference. The main difference is how each cloud names and groups its building blocks.

Rough service mapping:

- GitHub OIDC trust: Google Cloud Workload Identity Federation, Microsoft Entra federated credential
- CI/CD runtime identity: Google Cloud service account, Azure service principal or managed identity
- Container registry: Artifact Registry, Azure Container Registry
- Public container host: Cloud Run, Azure Container Apps or App Service
- Managed PostgreSQL: Cloud SQL for PostgreSQL, Azure Database for PostgreSQL
- Secrets store: Secret Manager, Key Vault
- Background work: Cloud Run Jobs or Pub/Sub workers, Container Apps jobs or Azure messaging workers

In practice, both clouds can support the same growth path for this app:

- one public containerized web app now
- a managed database later
- background jobs for longer training tasks
- secret storage for API keys or credentials
- multiple services if you split the app into UI, API, and workers

For this repository, Google Cloud Run is simpler than moving straight to a larger platform because it keeps the first deployment as a single HTTP container service.

## System architecture

```text
Developer push to main
        |
        v
GitHub repository
        |
        v
GitHub Actions workflow
  - run tests
  - build Docker image
        - exchange GitHub OIDC for short-lived Google Cloud credentials
        - push image to Artifact Registry
        - deploy image to Cloud Run
        |
        v
Artifact Registry
                                |
                                v
Cloud Run
        - pulls the image
  - starts the container
        - injects PORT for the web process
  - exposes HTTPS endpoint
        |
        v
User browser

Future extension:
Cloud SQL for PostgreSQL
  - stores user progress, accounts, or experiment history
```

## What Docker and the container do

- `Dockerfile`: the recipe for the runtime image. It defines the OS base image, installs Python dependencies, copies your source code, and starts the app with Gunicorn.
- Docker image: the packaged application artifact produced from the `Dockerfile`. It is the deployable unit that moves from GitHub Actions to your hosting platform.
- Container: a running instance of that image. Cloud Run starts the container on demand, scales instances up and down, and routes web traffic into it.

For this project, the container gives you a stable and repeatable production environment. That matters because the app depends on scientific Python packages, Dash, Flask, Matplotlib, NumPy, Pandas, and scikit-learn. The container makes production match the environment that was tested in CI.

## Why not deploy the PyInstaller build?

`app.spec` is useful for packaging a local executable, but it is not the right production artifact for a cloud deployment. Your real deployed application is the web server defined in `app.py`. Cloud platforms expect a long-running HTTP process, not a desktop executable.

## GitHub Actions deployment flow

The workflow in `.github/workflows/deploy.yml` does this on every push to `main`:

1. Runs the existing Python test suite on Windows, matching your current CI setup.
2. Authenticates GitHub Actions to Google Cloud using Workload Identity Federation.
3. Builds a Linux Docker image on Ubuntu.
4. Pushes the image to Google Artifact Registry as `latest` and a commit-specific SHA tag.
5. Deploys the image to Google Cloud Run.

## One-time Google Cloud setup

1. Create a Google Cloud project.
2. Enable Cloud Run, Artifact Registry, and the required IAM-related APIs for GitHub OIDC authentication.
3. Create an Artifact Registry Docker repository in your chosen region.
4. Create a Google Cloud service account for deployment.
5. Grant the deployment service account these roles:
         - `roles/run.admin`
         - `roles/artifactregistry.writer`
         - `roles/iam.serviceAccountUser`
6. Create a Workload Identity Pool and Provider that trust this GitHub repository.
7. Allow the GitHub principal set to impersonate the deployment service account.
8. Add these GitHub repository variables:
         - `GCP_PROJECT_ID`
         - `GCP_REGION`
         - `GCP_ARTIFACT_REGISTRY_REPOSITORY`
         - `GCP_CLOUD_RUN_SERVICE`
9. Add these GitHub repository secrets:
         - `GCP_WORKLOAD_IDENTITY_PROVIDER`
         - `GCP_SERVICE_ACCOUNT`

The workflow is designed to use short-lived Google credentials issued at runtime. That is more secure than storing a JSON service account key in GitHub.

## Detailed first deployment walkthrough

If this is your first time using Google Cloud, follow these steps in this order. The order matters because Cloud Run cannot run an image until Artifact Registry exists, and GitHub Actions cannot push or deploy anything until identity federation is configured.

### Step 1: Choose the names you will use everywhere

Pick these four values before you click through the console. Reusing the same values across Google Cloud and GitHub avoids confusion later.

- `PROJECT_ID`: your Google Cloud project ID
- `GCP_REGION`: a single region for both Artifact Registry and Cloud Run, for example `europe-west2`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`: a Docker repository name, for example `dashboard-images`
- `GCP_CLOUD_RUN_SERVICE`: the Cloud Run service name, for example `interactive-dashboard`

These values appear in the image URL that the workflow builds and deploys:

`REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/SERVICE:TAG`

### Step 2: Enable billing and the required APIs

In the Google Cloud console:

1. Select or create your project.
2. Open Billing and make sure billing is attached to the project.
3. Open APIs & Services, then Library.
4. Enable these APIs:
        - Cloud Run API
        - Artifact Registry API
        - IAM API
        - IAM Credentials API
        - Security Token Service API
        - Resource Manager API

Why this matters:

- Cloud Run hosts the app.
- Artifact Registry stores the built container image.
- IAM, IAM Credentials, and STS are required for GitHub Actions to exchange its OIDC token for short-lived Google Cloud credentials.

### Step 3: Create Artifact Registry

Artifact Registry is not your GitHub repository and it is not Cloud Run. It is the storage location for the built Docker image. Think of it as Google Cloud's private Docker image library.

In the Google Cloud console:

1. Search for Artifact Registry.
2. Open Repositories.
3. Click Create Repository.
4. Set Repository name to your chosen value, for example `dashboard-images`.
5. Set Format to `Docker`.
6. If repository mode appears, choose `Standard`.
7. Under Location type, choose `Region`.
8. Choose the same region you plan to use for Cloud Run, for example `europe-west2`.
9. Add a description if you want. This is optional.
10. For Encryption, leave the default Google-managed key unless you already know you need a customer-managed key.
11. If you see the Immutable image tags setting, leave it disabled for now.

Why leave immutable tags disabled:

The workflow in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) pushes both a commit SHA tag and a moving `latest` tag. If immutable tags are enabled, the first push works, but later pushes to `latest` will fail.

12. Ignore cleanup policies for now.
13. If vulnerability scanning is offered, you can leave it off for the first deployment if you want the simplest setup.
14. Click Create.

What you should expect after this step:

- The repository exists.
- It will probably be empty. That is normal.
- You are not expected to see an image yet because GitHub Actions has not pushed one yet.

### Step 4: Create the deployment service account

This service account is the Google Cloud identity that GitHub Actions will impersonate during the workflow.

In the Google Cloud console:

1. Search for Service Accounts.
2. Open IAM & Admin, then Service Accounts.
3. Click Create Service Account.
4. Set the name to something clear, for example `github-cloudrun-deployer`.
5. Leave the generated email address as it is.
6. Continue to roles.
7. Grant these roles to the service account:
        - Cloud Run Admin
        - Artifact Registry Writer
8. Finish creating the service account.

What this account does:

- It deploys new revisions to Cloud Run.
- It pushes built images into Artifact Registry.

### Step 5: Allow the deployer to use the runtime service identity

When Cloud Run runs your app, it runs it as a Google service account. If you do not choose a custom runtime account, Cloud Run typically uses the Compute Engine default service account.

Your GitHub deployer account needs permission to tell Cloud Run which runtime service account to use.

In the Google Cloud console:

1. Stay in IAM & Admin.
2. Open IAM.
3. Find the Compute Engine default service account. It usually looks like this:

        PROJECT_NUMBER-compute@developer.gserviceaccount.com

4. Open its permissions or grant access to it.
5. Grant the role Service Account User to your deployer service account.

If you later create a dedicated runtime service account for the app, grant Service Account User on that account instead.

### Step 6: Create Workload Identity Federation for GitHub Actions

This is the trust bridge between GitHub Actions and Google Cloud. GitHub does not get a long-lived password or JSON key. Instead, GitHub presents a short-lived OIDC token, and Google accepts it only if it matches the repository rules you define here.

In the Google Cloud console:

1. Search for Workload Identity Federation.
2. Open Workload Identity Pools.
3. Click Create pool.
4. Give the pool a name such as `github-pool`.
5. Continue.
6. Choose `OpenID Connect` as the provider type.
7. Set Provider name to something like `github-provider`.
8. Set Issuer URL to:

        `https://token.actions.githubusercontent.com`

9. Continue to attribute mapping.
10. Add these attribute mappings:
         - `google.subject` -> `assertion.sub`
         - `attribute.repository` -> `assertion.repository`
         - `attribute.repository_owner` -> `assertion.repository_owner`
         - `attribute.ref` -> `assertion.ref`
         - `attribute.actor` -> `assertion.actor`
11. In Attribute condition, restrict the provider to your repository. Use this shape:

        `assertion.repository == "OWNER/REPO"`

Replace `OWNER/REPO` with your real GitHub repository, for example:

        `assertion.repository == "HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard"`

12. Save the pool and provider.
13. Copy the full provider resource name. It will look like this:

        `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`

Why the attribute condition matters:

GitHub uses a shared issuer URL across all repositories. Without the repository restriction, another repository could try to present a token from the same issuer. The condition narrows trust to your repository only.

### Step 7: Allow GitHub to impersonate the deployer service account

Creating the pool and provider is only half the setup. You must also allow the matching GitHub principal to act as the deployer service account.

In the Google Cloud console:

1. Go back to IAM & Admin, then Service Accounts.
2. Open your `github-cloudrun-deployer` service account.
3. Open the Permissions tab.
4. Click Grant Access.
5. In New principals, enter this principal set, replacing the placeholders:

        `principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/attribute.repository/OWNER/REPO`

Example shape:

        `principalSet://iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/github-pool/attribute.repository/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard`

6. Grant these roles on the service account:
        - Workload Identity User
        - Service Account Token Creator

Why both are needed here:

- Workload Identity User allows GitHub's federated principal to impersonate the service account.
- Service Account Token Creator is needed because the workflow in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) generates an access token for Docker login when it pushes the image to Artifact Registry.

### Step 8: Add GitHub repository variables

In your GitHub repository:

1. Open Settings.
2. Open Secrets and variables, then Actions.
3. Open the Variables tab.
4. Add these variables:
        - `GCP_PROJECT_ID` = your project ID
        - `GCP_REGION` = your region, for example `europe-west2`
        - `GCP_ARTIFACT_REGISTRY_REPOSITORY` = your Artifact Registry repository name, for example `dashboard-images`
        - `GCP_CLOUD_RUN_SERVICE` = your Cloud Run service name, for example `interactive-dashboard`

### Step 9: Add GitHub repository secrets

In the same GitHub Actions settings area:

1. Open the Secrets tab.
2. Add these secrets:
        - `GCP_WORKLOAD_IDENTITY_PROVIDER` = the full provider resource name you copied earlier
        - `GCP_SERVICE_ACCOUNT` = the deployer service account email

Example values:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`: `projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
- `GCP_SERVICE_ACCOUNT`: `github-cloudrun-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com`

### Step 10: Start the first deployment

Once the Google Cloud setup and GitHub configuration are done, the easiest first deployment is through GitHub Actions rather than through the Cloud Run console.

In GitHub:

1. Open the Actions tab.
2. Open the workflow named Deploy To Cloud Run.
3. Click Run workflow.
4. Choose the `main` branch if prompted.
5. Start the run.

What the workflow does:

1. Runs the tests.
2. Authenticates to Google Cloud.
3. Builds the Docker image from [Dockerfile](Dockerfile).
4. Pushes the image to Artifact Registry.
5. Creates or updates the Cloud Run service.

### Step 11: Check that the image was actually pushed

In Google Cloud:

1. Open Artifact Registry.
2. Open your Docker repository.
3. You should now see an image path under the repository.
4. You should see at least two tags after a successful run:
        - a commit SHA tag
        - `latest`

If the repository is still empty, the workflow did not finish the image push step successfully.

### Step 12: Check that Cloud Run actually deployed

In Google Cloud:

1. Open Cloud Run.
2. Open Services.
3. You should see the service named with your `GCP_CLOUD_RUN_SERVICE` value.
4. Open it.
5. Check that there is at least one revision.
6. Check that traffic is routed to that revision.
7. Open the service URL shown at the top of the service page.

That URL is the proof that the deployment worked.

### Step 13: If the deployment fails, check the failure by layer

Use this order when debugging:

1. If the workflow fails before Docker build, the problem is usually a missing GitHub variable or secret.
2. If the workflow fails during Google authentication, the problem is usually the Workload Identity Provider value, the attribute condition, or the principal set binding on the service account.
3. If the workflow fails when pushing the image, the problem is usually Artifact Registry Writer access or a wrong repository name or region.
4. If the workflow pushes the image but Cloud Run fails, the problem is usually Cloud Run IAM, service account usage rights, or container startup issues.
5. If the deployment succeeds but the app does not load, open the Cloud Run service logs and inspect the container startup output.

## GitHub configuration values

- `GCP_PROJECT_ID`: your Google Cloud project ID
- `GCP_REGION`: the region for Artifact Registry and Cloud Run, for example `europe-west2`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`: the Docker repository name inside Artifact Registry
- `GCP_CLOUD_RUN_SERVICE`: the Cloud Run service name to deploy
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: the full provider resource name, such as `projects/123456789/locations/global/workloadIdentityPools/github/providers/main`
- `GCP_SERVICE_ACCOUNT`: the deployment service account email

## Environment choice summary

- Best current environment: Google Cloud Run with a Docker container.
- Best if you later add multiple services or background workers in Google Cloud: Cloud Run plus Cloud Run Jobs, Pub/Sub, and Cloud SQL.
- Not recommended for production: deploying the Windows PyInstaller executable.

## Future feature growth

As long as the app stays mostly stateless, Cloud Run is enough. If you add persistent user progress, experiment history, or user accounts, add a managed database instead of writing to the container filesystem.

Recommended future additions:

- Cloud SQL for PostgreSQL for saved progress and user data.
- Secret Manager for application secrets.
- Cloud Run Jobs or Pub/Sub-backed workers if you add long-running training tasks.
- Redis if you later need caching or shared session state.

## Local container test

```powershell
docker build -t dashboard-app .
docker run --rm -p 8000:8000 dashboard-app
```

Then open `http://127.0.0.1:8000/`.