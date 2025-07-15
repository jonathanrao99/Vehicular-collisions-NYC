# 🚀 Deployment Guide

This guide covers various deployment options for the NYC Motor Vehicle Collisions Analysis Dashboard.

## 📋 Table of Contents

- [Local Development](#-local-development)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [Production Considerations](#-production-considerations)
- [Monitoring and Logging](#-monitoring-and-logging)
- [Troubleshooting](#-troubleshooting)

## 💻 Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nyc-collisions-analysis.git
   cd nyc-collisions-analysis
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   ```bash
   # Extract the RAR file or download from NYC Open Data
   # Ensure Motor_Vehicle_Collisions_-_Crashes.csv is in the project root
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   - Open your browser
   - Navigate to `http://localhost:8501`

### Using Make Commands

```bash
# Full setup
make full-setup

# Run application
make run

# Clean up
make clean
```

## 🐳 Docker Deployment

### Prerequisites

- Docker
- Docker Compose

### Quick Start with Docker

1. **Build and run with Docker Compose**
   ```bash
   # Build and start
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop
   docker-compose down
   ```

2. **Using Make commands**
   ```bash
   # Build Docker image
   make docker-build
   
   # Run with Docker Compose
   make docker-run
   
   # Stop containers
   make docker-stop
   ```

### Manual Docker Commands

```bash
# Build image
docker build -t nyc-collisions-analysis .

# Run container
docker run -d \
  --name nyc-collisions-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  nyc-collisions-analysis

# View logs
docker logs -f nyc-collisions-app

# Stop container
docker stop nyc-collisions-app
docker rm nyc-collisions-app
```

### Docker Configuration

The application includes:
- **Dockerfile**: Multi-stage build for production
- **docker-compose.yml**: Orchestration with volumes and networking
- **Health checks**: Automatic health monitoring
- **Security**: Non-root user execution

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Set environment variables**
   ```bash
   heroku config:set ENVIRONMENT=production
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Open app**
   ```bash
   heroku open
   ```

### AWS Deployment

#### Using AWS ECS

1. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name nyc-collisions-analysis
   ```

2. **Build and push image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   docker build -t nyc-collisions-analysis .
   docker tag nyc-collisions-analysis:latest your-account.dkr.ecr.us-east-1.amazonaws.com/nyc-collisions-analysis:latest
   docker push your-account.dkr.ecr.us-east-1.amazonaws.com/nyc-collisions-analysis:latest
   ```

3. **Create ECS cluster and service**
   ```bash
   # Use AWS Console or AWS CLI to create ECS cluster
   # Deploy using the pushed image
   ```

#### Using AWS EC2

1. **Launch EC2 instance**
   ```bash
   # Launch Ubuntu instance
   # Configure security groups for port 8501
   ```

2. **Install Docker**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

3. **Deploy application**
   ```bash
   git clone https://github.com/yourusername/nyc-collisions-analysis.git
   cd nyc-collisions-analysis
   docker-compose up -d
   ```

### Google Cloud Platform

#### Using Google Cloud Run

1. **Enable Cloud Run API**
   ```bash
   gcloud services enable run.googleapis.com
   ```

2. **Build and deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/nyc-collisions-analysis
   gcloud run deploy nyc-collisions-analysis \
     --image gcr.io/PROJECT-ID/nyc-collisions-analysis \
     --platform managed \
     --allow-unauthenticated \
     --port 8501
   ```

### Azure Deployment

#### Using Azure Container Instances

1. **Build and push to Azure Container Registry**
   ```bash
   az acr build --registry your-registry --image nyc-collisions-analysis .
   ```

2. **Deploy to Container Instances**
   ```bash
   az container create \
     --resource-group your-rg \
     --name nyc-collisions-app \
     --image your-registry.azurecr.io/nyc-collisions-analysis:latest \
     --ports 8501 \
     --dns-name-label your-app-name
   ```

## 🏭 Production Considerations

### Environment Variables

```bash
# Production environment
ENVIRONMENT=production
LOG_LEVEL=WARNING
MAX_ROWS=500000
CACHE_TTL=7200

# Development environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
MAX_ROWS=50000
CACHE_TTL=300
```

### Security Best Practices

1. **Use HTTPS in production**
   ```bash
   # Configure SSL certificates
   # Use reverse proxy (nginx) for SSL termination
   ```

2. **Implement authentication** (if needed)
   ```bash
   # Add authentication middleware
   # Use environment variables for secrets
   ```

3. **Regular security updates**
   ```bash
   # Keep dependencies updated
   # Monitor for security vulnerabilities
   ```

### Performance Optimization

1. **Data caching**
   ```bash
   # Enable Redis for session storage
   # Implement data caching strategies
   ```

2. **Load balancing**
   ```bash
   # Use multiple instances
   # Implement health checks
   ```

3. **Resource limits**
   ```bash
   # Set memory and CPU limits
   # Monitor resource usage
   ```

## 📊 Monitoring and Logging

### Application Logs

```bash
# View application logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f nyc-collisions-app

# Access logs directory
ls -la logs/
```

### Health Monitoring

```bash
# Health check endpoint
curl http://localhost:8501/_stcore/health

# Application status
docker-compose ps
```

### Performance Monitoring

1. **Resource usage**
   ```bash
   # Monitor CPU and memory
   docker stats nyc-collisions-app
   ```

2. **Application metrics**
   ```bash
   # Monitor response times
   # Track user interactions
   ```

## 🔧 Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check if port is available
netstat -tulpn | grep 8501

# Check Docker logs
docker-compose logs

# Verify data file exists
ls -la Motor_Vehicle_Collisions_-_Crashes.csv
```

#### Data Loading Issues

```bash
# Check file permissions
ls -la *.csv

# Verify file format
head -5 Motor_Vehicle_Collisions_-_Crashes.csv

# Check file size
du -h Motor_Vehicle_Collisions_-_Crashes.csv
```

#### Memory Issues

```bash
# Reduce data size
# Edit config.py: DATA_CONFIG['max_rows'] = 50000

# Monitor memory usage
docker stats

# Restart with more memory
docker-compose down
docker-compose up -d
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Support

For additional support:

1. **Check documentation**: README.md
2. **Review logs**: `logs/app.log`
3. **Run tests**: `python test_app.py`
4. **Create issue**: GitHub Issues

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale with Docker Compose
docker-compose up -d --scale nyc-collisions-app=3

# Use load balancer
# Configure nginx or HAProxy
```

### Vertical Scaling

```bash
# Increase resources in docker-compose.yml
services:
  nyc-collisions-app:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 🔄 Updates and Maintenance

### Application Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Data Updates

```bash
# Download latest data
# Replace CSV file
# Restart application
docker-compose restart
```

### Dependency Updates

```bash
# Update requirements
pip install --upgrade -r requirements.txt

# Rebuild Docker image
docker-compose build --no-cache
```

---

**For additional deployment options or custom configurations, please refer to the Streamlit documentation or contact the development team.**