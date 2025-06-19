#!/bin/bash
# This script sets up the project by installing dependencies, checking for a poetry environment,
# and installing pre-commit hooks with improved Ubuntu compatibility.

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

# Function for error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to try multiple Poetry installation methods
install_poetry() {
    print_section "Poetry Installation"
    
    # Method 1: Try pipx installation
    echo "Attempting Poetry installation via pipx..."
    if command -v pipx &> /dev/null; then
        pipx install poetry
        if command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully via pipx"
            return 0
        fi
    else
        echo "pipx not found, trying to install it..."
        python3 -m pip install --user pipx
        if [ $? -eq 0 ]; then
            python3 -m pipx ensurepath
            export PATH="$HOME/.local/bin:$PATH"
            pipx install poetry
            if command -v poetry &> /dev/null; then
                print_success "Poetry installed successfully via pipx"
                return 0
            fi
        fi
    fi
    
    # Method 2: Try official installer with Ubuntu fix
    echo "Attempting Poetry installation via official installer with Ubuntu compatibility..."
    export DEB_PYTHON_INSTALL_LAYOUT=deb
    curl -sSL https://install.python-poetry.org | python3 -
    POETRY_INSTALL_STATUS=$?
    
    if [ $POETRY_INSTALL_STATUS -eq 0 ]; then
        export PATH="$HOME/.local/bin:$PATH"
        hash -r
        
        if [ -f "$HOME/.local/bin/poetry" ]; then
            print_success "Poetry installed successfully via official installer"
            return 0
        elif command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully via official installer"
            return 0
        fi
    fi
    
    # Method 3: Manual installation
    echo "Attempting manual Poetry installation..."
    POETRY_VENV="$HOME/.poetry-venv"
    python3 -m venv "$POETRY_VENV"
    "$POETRY_VENV/bin/pip" install -U pip setuptools
    "$POETRY_VENV/bin/pip" install poetry
    
    if [ $? -eq 0 ]; then
        # Create symlink or add to PATH
        if [ ! -d "$HOME/.local/bin" ]; then
            mkdir -p "$HOME/.local/bin"
        fi
        ln -sf "$POETRY_VENV/bin/poetry" "$HOME/.local/bin/poetry"
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully via manual method"
            echo -e "${YELLOW}    Add this to your shell profile for persistence:${NC}"
            echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
            return 0
        fi
    fi
    
    print_error "All Poetry installation methods failed!"
    return 1
}

# ---- POETRY SETUP ---- #
print_section "Poetry Setup"

# Check if Poetry is already installed
if command -v poetry &> /dev/null; then
    print_success "Poetry already installed ($(poetry --version))"
else
    # Try to install Poetry using multiple methods
    if ! install_poetry; then
        print_warning "Poetry installation failed!"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
fi

# Only proceed if Poetry is available
if command -v poetry &> /dev/null; then
    # Configure poetry to create venvs in project
    poetry config virtualenvs.in-project true
    print_success "Poetry configured to create virtual environments in project directory"

    # Check for virtual environment
    if [ ! -d ".venv" ]; then
        echo "No virtual environment found. Creating one..."
        
        # Create virtual environment and install dependencies based on the --dev flag
        if [ "$INSTALL_DEV" = true ]; then
            echo "Installing with development dependencies..."
            poetry install --with dev
        else
            echo "Installing without development dependencies..."
            poetry install --only main
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
else
    print_error "Poetry is not available - cannot create virtual environment"
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
fi

# --- Final Status Message --- #
print_section "Setup Status"
if [ $ERRORS_FOUND -eq 0 ]; then
    if [ "$INSTALL_DEV" = true ]; then
        print_success "Setup Complete with DEV dependencies! 🎉"
    else
        print_success "Setup Complete with regular dependencies! 🎉"
    fi
    print_success "To activate the virtual environment, run: poetry shell"
    print_success "Or use: source .venv/bin/activate"
    print_success "To run commands in the environment: poetry run <command>"
else
    print_warning "Setup completed with warnings and errors! Please check the messages above."
    echo -e "${YELLOW}    ${ERRORS_FOUND} issue(s) were detected that may affect functionality.${NC}"
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}    You can still activate the environment with: source .venv/bin/activate${NC}"
    else
        echo -e "${RED}    The virtual environment setup failed. Fix the issues before proceeding.${NC}"
    fi
fi