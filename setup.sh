#!/bin/bash
# This script sets up the project by installing dependencies, checking for a poetry environment,
# and installing pre-commit hooks.

# Add color and formatting variables at the top
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Initialize error tracking
ERRORS_FOUND=0

# Process command line arguments
INSTALL_DEV=false
for arg in "$@"; do
  case $arg in
    --dev)
      INSTALL_DEV=true
      shift
      ;;
  esac
done

# Function for section headers
print_section() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

# Function for success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function for warnings
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ---- POETRY SETUP ---- #
print_section "Poetry Setup"

# First check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    
    # Run the installation command
    curl -sSL https://install.python-poetry.org | python3 -
    POETRY_INSTALL_STATUS=$?
    
    if [ $POETRY_INSTALL_STATUS -ne 0 ]; then
        print_warning "Poetry installation failed!"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    else
        export PATH="$HOME/.local/bin:$PATH"
        
        # Verify installation succeeded
        if ! command -v poetry &> /dev/null; then
            print_warning "Poetry was installed but cannot be found in PATH!"
            echo -e "${YELLOW}    Try adding this to your shell profile:${NC}"
            echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
            ERRORS_FOUND=$((ERRORS_FOUND + 1))
        else
            print_success "Poetry installed successfully"
        fi
    fi
else
    print_success "Poetry already installed"
fi

# Configure poetry to create venvs in project
poetry config virtualenvs.in-project true
print_success "Poetry configured to create virtual environments in project directory"

# Then check for virtual environment
if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Creating one..."
    
    # Create virtual environment and install dependencies based on the --dev flag
    if [ "$INSTALL_DEV" = true ]; then
        echo "Installing with development dependencies..."
        poetry install --with dev
    else
        echo "Installing without development dependencies..."
        poetry install
    fi
    
    POETRY_VENV_STATUS=$?
    
    if [ $POETRY_VENV_STATUS -ne 0 ]; then
        print_warning "Failed to create Poetry virtual environment!"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    else
        print_success "Poetry environment created successfully"
    fi
else
    print_success "Poetry environment already exists"
    # Update dependencies based on the --dev flag
    if [ "$INSTALL_DEV" = true ]; then
        echo "Updating with development dependencies..."
        poetry cache clear --all pypi
        poetry install --with dev
    else
        echo "Updating without development dependencies..."
        poetry cache clear --all pypi
        poetry install
    fi
fi

# --- Final Status Message --- #
print_section "Setup Status"
if [ $ERRORS_FOUND -eq 0 ]; then
    if [ "$INSTALL_DEV" = true ]; then
        print_success "Setup Complete with DEV dependencies! 🎉"
    else
        print_success "Setup Complete with regular dependencies! 🎉"
    fi
    print_success "To activate the virtual environment, run: poetry env activate"
    print_success "Or use: source .venv/bin/activate"
else
    print_warning "Setup completed with warnings and errors! Please check the messages above."
    echo -e "${YELLOW}    ${ERRORS_FOUND} issue(s) were detected that may affect functionality.${NC}"
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}    You can still activate the environment with: source .venv/bin/activate${NC}"
    else
        echo -e "${RED}    The virtual environment setup failed. Fix the issues before proceeding.${NC}"
    fi
fi