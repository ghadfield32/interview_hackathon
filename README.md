# FastAPI + React • Railway-ready Template

This repo contains a FastAPI back-end (`/api`) and a React (Vite) front-end (`/web`) with **JAX/NumPyro backend** for Bayesian machine learning models.  
Follow the steps below to run everything locally with **Railway CLI** and then deploy the two services on railway.com.

## 🧠 Machine Learning Features

- **JAX/NumPyro Backend**: No C compilation required - works seamlessly on Windows, Linux, and macOS
- **Hierarchical Bayesian Models**: Breast cancer diagnosis with varying intercepts by texture quintiles
- **Random Forest & Logistic Regression**: Iris species classification
- **Self-healing Model Service**: Automatically trains missing models in the background
- **MLflow Integration**: Model versioning and deployment tracking
- **Rate Limiting**: Redis-backed token bucket rate limiting with configurable limits per endpoint type
- **Prediction Caching**: Redis-based caching of ML model results for improved performance
- **Automated Garbage Collection**: Keeps Railway volumes tidy by pruning old runs and artifacts

### Why JAX/NumPyro?

- **Cross-platform**: No compiler dependencies - pure Python + JIT compilation
- **Fast sampling**: NumPyro NUTS sampler is significantly faster than traditional PyMC
- **Windows-friendly**: Eliminates MSVC/GCC compilation issues
- **Production-ready**: Stable JAX 0.4.28 LTS with NumPyro 0.14.0

---

## ⚙️ Configuration System

The repository uses a centralized `config.yaml` system that controls all aspects of the application:

### Configuration Hierarchy

1. **Built-in safe defaults** (hardcoded fallbacks)
2. **config.yaml: default block** (base configuration)
3. **config.yaml: environment block** (dev/staging/prod overlays)
4. **Environment variables** (12-factor app compliance - highest precedence)

### Key Configuration Areas

#### 🔒 Rate Limiting Configuration
```yaml
# Rate Limiting (requests per window)
RATE_LIMIT_DEFAULT: 60        # General API endpoints
RATE_LIMIT_CANCER: 30         # Cancer prediction (heavier computation)
RATE_LIMIT_LOGIN: 3           # Login attempts (security)
RATE_LIMIT_TRAINING: 2        # Model training (resource intensive)
RATE_LIMIT_WINDOW: 60         # Window in seconds for default/cancer/training
RATE_LIMIT_WINDOW_LIGHT: 300  # Window for light endpoints (iris/predict)
RATE_LIMIT_LOGIN_WINDOW: 20   # Window for login attempts
```

#### 💾 Prediction Caching (Redis)
```yaml
# Prediction caching (Redis)
CACHE_ENABLED: 0              # 0 = disabled, 1 = enabled
CACHE_TTL_MINUTES: 60         # How long to cache predictions (in minutes)
```

#### 🧠 MLflow & Model Management
```yaml
# MLflow Configuration
MLFLOW_EXPERIMENT: "ml_fullstack_models"
MLFLOW_TRACKING_URI: "file:api/mlruns_local"  # Local dev
MLFLOW_REGISTRY_URI: "file:api/mlruns_local"  # Local dev
RETAIN_RUNS_PER_MODEL: 5      # Keep N latest runs per model
MLFLOW_GC_AFTER_TRAIN: 1      # Run garbage collection after training

# Model Training Flags
SKIP_BACKGROUND_TRAINING: 0   # 0 = train on startup, 1 = skip
AUTO_TRAIN_MISSING: 1         # 0 = manual training, 1 = auto-train missing models
UNIT_TESTING: 0               # 0 = normal mode, 1 = test mode
```

#### 🎯 MLOps Quality Gates
```yaml
# Quality Gate Thresholds
QUALITY_GATE_ACCURACY_THRESHOLD: 0.85
QUALITY_GATE_F1_THRESHOLD: 0.85

# MLOps Configuration
ENVIRONMENT: "development"
REQUIRE_MODEL_APPROVAL: 0     # 0 = auto-deploy, 1 = manual approval
AUTO_PROMOTE_TO_PRODUCTION: 0 # 0 = manual promotion, 1 = auto-promote
ENABLE_MODEL_COMPARISON: 1    # 0 = disable, 1 = enable model comparison
MODEL_AUDIT_ENFORCEMENT: "warn" # "warn", "fail", "ignore"
MAX_MODEL_VERSIONS_PER_MODEL: 10
```

#### 🔧 JAX/XLA Configuration
```yaml
# JAX/XLA Configuration
XLA_FLAGS: "--xla_force_host_platform_device_count=1"
PYTENSOR_FLAGS: "device=cpu,floatX=float32"
```

### Environment-Specific Settings

#### Development (`dev`)
- **Relaxed rate limits**: Higher limits for testing
- **Caching enabled**: Short TTL (5 minutes) for testing
- **Background training**: Enabled for rapid iteration
- **Debug flags**: Enabled for troubleshooting

#### Staging (`staging`)
- **Stricter rate limits**: Production-like limits
- **Caching disabled**: No caching by default
- **Background training**: Disabled (pre-trained models only)
- **Quality gates**: Higher thresholds (0.90)

#### Production (`prod`)
- **Strictest rate limits**: Conservative limits
- **Caching disabled**: Can be enabled via environment variables
- **Background training**: Disabled (pre-trained models only)
- **Quality gates**: Highest thresholds (0.92)
- **Audit enforcement**: "fail" (strict compliance)

### How to Adjust Configuration

#### 1. Modify config.yaml
Edit the appropriate environment block in `config.yaml`:

```yaml
dev:
  # Increase rate limits for development
  RATE_LIMIT_DEFAULT: 200
  RATE_LIMIT_CANCER: 100
  
  # Enable caching with longer TTL
  CACHE_ENABLED: 1
  CACHE_TTL_MINUTES: 30
  
  # Relax quality gates
  QUALITY_GATE_ACCURACY_THRESHOLD: 0.80
  QUALITY_GATE_F1_THRESHOLD: 0.80
```

#### 2. Override with Environment Variables
Environment variables take highest precedence:

```bash
# Override specific settings
export RATE_LIMIT_DEFAULT=100
export CACHE_ENABLED=1
export CACHE_TTL_MINUTES=30
export QUALITY_GATE_ACCURACY_THRESHOLD=0.85
```

#### 3. Railway Environment Variables
In Railway dashboard, add environment variables to override config.yaml:

```bash
# Enable caching in production
CACHE_ENABLED=1
CACHE_TTL_MINUTES=60

# Adjust rate limits
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_CANCER=50

# Modify quality gates
QUALITY_GATE_ACCURACY_THRESHOLD=0.90
QUALITY_GATE_F1_THRESHOLD=0.90
```

### Configuration Validation

Check your effective configuration at runtime:

```bash
# View effective configuration (sensitive fields redacted)
curl http://localhost:8000/api/v1/debug/effective-config
```

### Common Configuration Adjustments

#### Enable Prediction Caching
```yaml
# In config.yaml dev block
CACHE_ENABLED: 1
CACHE_TTL_MINUTES: 30
```

#### Adjust Rate Limits
```yaml
# More permissive development limits
RATE_LIMIT_DEFAULT: 200
RATE_LIMIT_CANCER: 100
RATE_LIMIT_LOGIN: 20
RATE_LIMIT_TRAINING: 10
```

#### Modify Quality Gates
```yaml
# Stricter quality requirements
QUALITY_GATE_ACCURACY_THRESHOLD: 0.90
QUALITY_GATE_F1_THRESHOLD: 0.90
```

#### Enable Model Training
```yaml
# Allow background training
SKIP_BACKGROUND_TRAINING: 0
AUTO_TRAIN_MISSING: 1
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ghadfield32/cancerbayes_irisrf_ex_fastapi_react
cd cancerbayes_irisrf_ex_fastapi_react
```

# Dev Env***
### 2. Set Up Environment Variables

Copy the environment templates to create your local configuration:

```bash
# Root level environment
cp env.template .env

# API environment
cp api/env.template api/.env

# Web environment  
cp web/env.template web/.env
```

**Important**: Update the following variables in your `.env` files:
- `USERNAME_KEY` and `USER_PASSWORD` - Set your desired login credentials
- `VITE_API_URL` - For local development: `http://127.0.0.1:8000/api/v1`
- `SECRET_KEY` - Generate a secure key for production

### 3. Local Development Setup

Install all dependencies and set up the development environment:

```bash
# use if you want to edit code locally
uv sync

# Install all dependencies (Python venv, uv, and Node modules)
npm run install:all

# Start the backend development server
npm run backend:dev

# In a new terminal, seed the database with credentials
npm run seed

# In another terminal, start the frontend
cd web && npm run dev
```

### 4. Test Local Setup

Verify everything is working:

```bash
# Test backend API
curl -s http://127.0.0.1:8000/docs

# Test frontend
curl -s http://127.0.0.1:5173

# Check effective configuration
curl -s http://127.0.0.1:8000/api/v1/debug/effective-config
```

# Setup in Railway for staging env
### 3. Create Railway Project

1. Go to [Railway Dashboard](https://railway.app)
2. Create a new project
3. Note your project ID for linking

### 4. Set Up Railway Services

Create **three services** in your Railway project:

#### Service 1: API Backend
1. Create a new service from GitHub
2. Select your repository
3. Go to **Settings** → **Root Directory** → Set to `api`
4. Go to **Variables** → **Raw Editor** and add:

REDIS_URL="${{Redis.REDIS_URL}}"
SECRET_KEY="dev-secret-key-change-in-production"
DATABASE_URL="sqlite+aiosqlite:///./app.db"
USERNAME_KEY="alice"
USER_PASSWORD="supersecretvalue"
APP_ENV="production"



#### Service 2: Web Frontend
1. Create another service from GitHub
2. Select your repository  
3. Go to **Settings** → **Root Directory** → Set to `web`
4. Go to **Variables** → **Raw Editor** and add:
4a. Get the Vite api url from the API service in the Railway dashboard by making an external domain for it

```bash
# Copy from web/env.template and update:
VITE_API_URL=https://your-api-service-url.up.railway.app/api/v1
USERNAME_KEY=your-username
USER_PASSWORD=your-password
```

#### Service 3: Redis (for Rate Limiting & Caching)
1. Add **Redis** plugin from Railway's plugin marketplace
2. In Data Tab, Press Connect to the database redis Connect button. Note the Redis external URL for railway cli testing (staging env.staging file)and use the Internal URL for production. Update it in your API service variables

### 5. Create Railway Volume

1. Right-click on the background of your Railway workspace
2. Select **"Create Volume"**
3. Name it `mlruns`
4. Mount path: `/data/mlruns`
5. Attach to your **API service**
6. Set size to 1GB (sufficient for model artifacts)

---

# Staging Env***

## 🚂 Railway Deployment

### 1. Install Railway CLI

```bash
# macOS/Linux/WSL
curl -fsSL https://railway.com/install.sh | sh

# Windows (PowerShell)
# Download from https://railway.com/cli
```

### 2. Railway CLI Authentication

#### 2.a. Get a profile level token from Railway dashboard and set it
### **Setup in RAILWAY_API_TOKEN in .env**
#### skip steps 2.b and 2.c if you run the command below and successful
 
Steps: 
1) goto railway profile dashboard > tokens > create new token
2) copy the token
3) paste it in the root .env file
4) save the file
```bash
npm run railway:auth # rerun each new window
```


#### 2.b. run this if the npm run railway:auth fails
```bash
railway login
```

**If login fails on Windows:**
```powershell
# Clear existing configuration
Remove-Item -Force "$Env:USERPROFILE\.railway\config.json"
Test-Path "$Env:USERPROFILE\.railway\config.json"  # Should return False

# Clear environment variables
Remove-Item Env:RAILWAY_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:RAILWAY_API_TOKEN -ErrorAction SilentlyContinue

# Get a profile level token from Railway dashboard and set it
$Env:RAILWAY_API_TOKEN = '<your-token>'

# Re-authenticate
railway logout
railway whoami
railway login
```



### 6. Link Services with Railway CLI

```bash
# Link API service
cd api
railway link
# Select your workspace and API service

# Test API locally with Railway environment in the staging env
railway run npm run backend:staging

# Link Web service  
cd ../web
railway link
# Select your workspace and Web service

# Test Web locally with Railway environment
railway run npm run dev
```

# Production Env***

### 1. update models/env/config.yaml to production
```bash
npm run backend:prod
```

### 7. Deploy to Production

```bash
# Commit and push to GitHub
git add .
git commit -m "feat: ready for deployment"
git push

# Railway will automatically deploy both services
```

### 8. Configure Production URLs

After deployment:

1. Go to your **API service** in Railway dashboard
2. Copy the generated domain (e.g., `https://api-123456.up.railway.app`)
3. Go to your **Web service** → **Variables**
4. Update `VITE_API_URL` to: `https://api-123456.up.railway.app/api/v1`
5. Redeploy the web service

### 9. Update Credentials

In both API and Web services, update:
- `USERNAME_KEY` - Your desired username
- `USER_PASSWORD` - Your desired password

---

## 🔧 Development Workflow

### Local Development Commands

```bash
# Full environment reset (if needed)
npm run install:all

# Start backend with hot reload
npm run backend:dev

# Seed database with credentials
npm run seed

# Start frontend
cd web && npm run dev # use ; instead of && for powershell

# Run both simultaneously
npm run dev
```

### Railway Development Commands

```bash
# Test API with Railway environment
cd api && railway run npm run backend

# Test Web with Railway environment  
cd web && railway run npm run dev

# Seed database on Railway
cd api && railway run npm run seed
```

### Testing

```bash
# Test API endpoints
npm run test:api:endpoints

# Test all API functionality
npm run test:api

# Validate setup
npm run validate:setup
```

---

## 🗂️ Project Structure

```
cancerbayes_irisrf_ex_fastapi_react/
├── config.yaml                 # Centralized configuration
├── api/                        # FastAPI Backend
│   ├── app/
│   │   ├── core/               # Configuration
│   │   ├── ml/                 # Machine Learning
│   │   ├── routers/            # API Routes
│   │   ├── schemas/            # Pydantic Models
│   │   ├── services/           # Business Logic
│   │   └── middleware/         # Rate Limiting
│   ├── scripts/                # Database seeding
│   ├── tests/                  # Backend tests
│   └── env.template            # API environment
├── web/                        # React Frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── Login.jsx
│   │   │   ├── Layout.jsx
│   │   │   ├── CancerForm.jsx
│   │   │   ├── IrisForm.jsx
│   │   │   ├── ModelTraining.jsx
│   │   │   └── ResultsDisplay.jsx
│   │   └── services/           # API client
│   └── env.template            # Frontend environment
├── scripts/                    # Build scripts
├── package.json                # Root scripts
└── env.template                # Root environment
```

---

## 🧹 MLflow Volume Management

### Automatic Cleanup

The system includes automatic garbage collection:

- **Retention Policy**: Keeps latest 5 runs per model (configurable)
- **Background Cleanup**: Runs after each training session
- **Volume Monitoring**: Tracks disk usage and cleans old artifacts

### Manual Cleanup

```bash
# Local cleanup
cd api
python -c "
from app.services.ml.model_service import model_service
import asyncio
asyncio.run(model_service.vacuum_store())
"

# Railway cleanup (via Cron Job)
# Add daily cron job in Railway dashboard:
# Schedule: 0 2 * * * (daily at 2 AM)
# Command: python -c "from app.services.ml.model_service import model_service; import asyncio; asyncio.run(model_service.vacuum_store())"
```

---

## 🔍 Troubleshooting

### Common Issues

**401 Token Expired**
- Refresh token in localStorage or log out/in
- FastAPI will return helpful hints

**Wrong Root Directory**
- Ensure API service root is set to `api/`
- Ensure Web service root is set to `web/`
- Check build logs for dependency installation errors

**Railway CLI Issues**
- Clear configuration: `Remove-Item -Force "$Env:USERPROFILE\.railway\config.json"`
- Re-authenticate with fresh token

**Model Training Issues**
- Check MLflow volume is properly mounted at `/data/mlruns`
- Verify `MLFLOW_TRACKING_URI=file:/data/mlruns` in API variables
- Monitor logs for PyMC convergence warnings

**Rate Limiting Issues**
- Ensure Redis service is properly configured
- Check `REDIS_URL` in API service variables
- Verify rate limit settings in environment variables

**Caching Issues**
- Verify `CACHE_ENABLED=1` in configuration
- Check `CACHE_TTL_MINUTES` setting
- Ensure Redis is accessible and working

### Validation Commands

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check model status
curl http://localhost:8000/api/v1/ready/full

# Test predictions
curl -X POST http://localhost:8000/api/v1/cancer/predict \
  -H "Content-Type: application/json" \
  -d '{"texture_mean": 17.5, "area_mean": 1000}'

# View effective configuration
curl http://localhost:8000/api/v1/debug/effective-config
```

---

## 🚀 Production Checklist

- [ ] Environment variables configured in Railway
- [ ] Redis service added and linked
- [ ] MLflow volume created and mounted
- [ ] API service root directory set to `api/`
- [ ] Web service root directory set to `web/`
- [ ] `VITE_API_URL` updated with production API domain
- [ ] `USERNAME_KEY` and `USER_PASSWORD` set
- [ ] Both services deployed successfully
- [ ] Frontend accessible and functional
- [ ] API endpoints responding correctly
- [ ] Model predictions working
- [ ] Rate limiting functional
- [ ] Caching working (if enabled)
- [ ] Configuration validated via `/api/v1/debug/effective-config`

Happy shipping! 🚂
