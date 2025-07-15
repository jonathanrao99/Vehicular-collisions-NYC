.PHONY: help install run test clean docker-build docker-run docker-stop lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run the application locally"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run the application locally
run:
	@echo "Starting NYC Collisions Analysis Dashboard..."
	streamlit run app.py

# Run tests (placeholder for future test implementation)
test:
	@echo "Running tests..."
	@echo "No tests implemented yet."

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	@echo "Cleanup completed!"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t nyc-collisions-analysis .
	@echo "Docker image built successfully!"

# Run with Docker Compose
docker-run:
	@echo "Starting application with Docker Compose..."
	docker-compose up -d
	@echo "Application is running at http://localhost:8501"

# Stop Docker containers
docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "Docker containers stopped!"

# Run linting
lint:
	@echo "Running linting..."
	@echo "Install flake8 and black for linting: pip install flake8 black"
	@echo "No linting configured yet."

# Format code
format:
	@echo "Formatting code..."
	@echo "Install black for code formatting: pip install black"
	@echo "No formatting configured yet."

# Development setup
dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the application"

# Production setup
prod-setup: docker-build
	@echo "Production environment setup complete!"
	@echo "Run 'make docker-run' to start the application"

# Show logs
logs:
	@echo "Showing application logs..."
	docker-compose logs -f

# Restart application
restart: docker-stop docker-run
	@echo "Application restarted!"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	@echo "Dependencies updated!"

# Create data directory
setup-data:
	@echo "Creating data directory..."
	mkdir -p data
	@echo "Data directory created!"

# Full setup for new environment
full-setup: setup-data install
	@echo "Full setup completed!"
	@echo "Place your CSV data file in the data/ directory"
	@echo "Run 'make run' to start the application"