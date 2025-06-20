#!/bin/bash
# This script sets up the project by installing Python 3.12, Poetry, and dependencies.
# Asks for user permission before making system-level changes.

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
SKIP_PROMPTS=false
for arg in "$@"; do
  case $arg in
    --dev)
      INSTALL_DEV=true
      shift
      ;;
    --yes|-y)
      SKIP_PROMPTS=true
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

# Function to ask for user confirmation
ask_permission() {
    if [ "$SKIP_PROMPTS" = true ]; then
        return 0
    fi
    
    echo -e "${YELLOW}$1${NC}"
    echo -e "${BOLD}Do you want to proceed? (y/N)${NC}"
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" ]] || [[ "$ID_LIKE" == *"ubuntu"* ]] || [[ "$ID_LIKE" == *"debian"* ]]; then
                echo "ubuntu"
            elif [[ "$ID" == "centos"* ]] || [[ "$ID" == "rhel"* ]] || [[ "$ID_LIKE" == *"rhel"* ]]; then
                echo "centos"
            elif [[ "$ID" == "fedora"* ]]; then
                echo "fedora"
            else
                echo "linux"
            fi
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_section "System Dependencies Check"
    
    OS=$(detect_os)
    echo "Detected OS: $OS"
    
    case $OS in
        "ubuntu")
            echo "Checking for required system packages..."
            MISSING_PACKAGES=""
            
            # Check for essential packages
            for pkg in software-properties-common curl wget build-essential libssl-dev libffi-dev python3-dev python3-pip git; do
                if ! dpkg -l | grep -q "^ii  $pkg "; then
                    MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
                fi
            done
            
            if [ -n "$MISSING_PACKAGES" ]; then
                echo "Missing packages:$MISSING_PACKAGES"
                if ask_permission "This will install system packages using sudo. This is generally safe but requires admin privileges."; then
                    sudo apt update
                    sudo apt install -y $MISSING_PACKAGES
                    if [ $? -eq 0 ]; then
                        print_success "System dependencies installed"
                    else
                        print_error "Failed to install system dependencies"
                        ERRORS_FOUND=$((ERRORS_FOUND + 1))
                    fi
                else
                    print_warning "Skipping system dependencies - some features may not work"
                fi
            else
                print_success "All system dependencies already installed"
            fi
            ;;
        "centos"|"fedora")
            echo "System package installation required for CentOS/RHEL/Fedora"
            if ask_permission "This will install development tools and libraries using sudo."; then
                if [[ "$OS" == "centos" ]]; then
                    sudo yum update -y
                    sudo yum groupinstall -y "Development Tools"
                    sudo yum install -y openssl-devel libffi-devel python3-devel python3-pip curl wget git
                else
                    sudo dnf update -y
                    sudo dnf groupinstall -y "Development Tools"
                    sudo dnf install -y openssl-devel libffi-devel python3-devel python3-pip curl wget git
                fi
                
                if [ $? -eq 0 ]; then
                    print_success "System dependencies installed"
                else
                    print_error "Failed to install system dependencies"
                    ERRORS_FOUND=$((ERRORS_FOUND + 1))
                fi
            else
                print_warning "Skipping system dependencies - Python compilation will likely fail"
            fi
            ;;
        "macos")
            echo "Checking for Homebrew on macOS..."
            if ! command -v brew &> /dev/null; then
                if ask_permission "Homebrew is not installed. Install it? (Recommended for macOS development)"; then
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                else
                    print_warning "Skipping Homebrew installation"
                fi
            fi
            if command -v brew &> /dev/null; then
                print_success "Homebrew available"
            fi
            ;;
        *)
            print_warning "Unknown OS - you may need to install dependencies manually"
            ;;
    esac
}

# Function to install Python 3.12
install_python312() {
    print_section "Python 3.12 Installation"
    
    # Check if Python 3.12 is already available
    if command -v python3.12 &> /dev/null; then
        PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
        print_success "Python 3.12 already installed: $PYTHON_VERSION"
        export PYTHON_EXECUTABLE="python3.12"
        return 0
    fi
    
    OS=$(detect_os)
    
    case $OS in
        "ubuntu")
            echo "Python 3.12 not found. Need to install via deadsnakes PPA."
            if ask_permission "This will add the deadsnakes PPA and install Python 3.12. This is safe and commonly used."; then
                sudo add-apt-repository ppa:deadsnakes/ppa -y
                sudo apt update
                sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils
                
                # Install pip for Python 3.12
                if ! python3.12 -m pip --version &> /dev/null; then
                    echo "Installing pip for Python 3.12..."
                    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
                fi
            else
                print_warning "Skipping Python 3.12 installation"
                return 1
            fi
            ;;
        "centos"|"fedora")
            echo "Python 3.12 not found. Need to compile from source."
            echo "${RED}WARNING: This will take 10-20 minutes and use significant CPU/disk space.${NC}"
            if ask_permission "Compile Python 3.12 from source? (Required for this project)"; then
                echo "Installing Python 3.12 from source (this may take a while)..."
                cd /tmp
                wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
                tar xzf Python-3.12.0.tgz
                cd Python-3.12.0
                ./configure --enable-optimizations --with-ensurepip=install
                make -j$(nproc)
                sudo make altinstall  # altinstall is safer than install
                cd - > /dev/null
                
                # Create symlink if it doesn't exist
                if [ ! -f /usr/local/bin/python3.12 ] && [ -f /usr/local/bin/python3.12 ]; then
                    sudo ln -sf /usr/local/bin/python3.12 /usr/bin/python3.12
                fi
            else
                print_warning "Skipping Python 3.12 compilation"
                return 1
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                if ask_permission "Install Python 3.12 via Homebrew?"; then
                    brew install python@3.12
                    
                    # Add to PATH if needed
                    if ! command -v python3.12 &> /dev/null; then
                        export PATH="/opt/homebrew/bin:$PATH"
                        export PATH="/usr/local/bin:$PATH"
                    fi
                else
                    print_warning "Skipping Python 3.12 installation"
                    return 1
                fi
            else
                print_error "Homebrew not available - cannot install Python 3.12"
                return 1
            fi
            ;;
        *)
            print_error "Cannot automatically install Python 3.12 on this OS"
            print_warning "Please install Python 3.12 manually and re-run this script"
            ERRORS_FOUND=$((ERRORS_FOUND + 1))
            return 1
            ;;
    esac
    
    # Verify installation
    if command -v python3.12 &> /dev/null; then
        PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
        print_success "Python 3.12 installed successfully: $PYTHON_VERSION"
        export PYTHON_EXECUTABLE="python3.12"
        return 0
    else
        print_error "Python 3.12 installation failed"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
        return 1
    fi
}

# Function to check Python version compatibility
check_python_version() {
    print_section "Python Version Check"
    
    # Check if python3.12 is available
    if command -v python3.12 &> /dev/null; then
        PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
        print_success "Python 3.12 found: $PYTHON_VERSION"
        export PYTHON_EXECUTABLE="python3.12"
        return 0
    fi
    
    # Check current python3 version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        echo "Current Python version: $PYTHON_VERSION"
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
            print_success "Python version is compatible"
            export PYTHON_EXECUTABLE="python3"
            return 0
        else
            print_warning "Python version $PYTHON_VERSION is not compatible (requires >=3.12)"
            return 1
        fi
    else
        print_error "Python 3 not found!"
        return 1
    fi
}

# Function to try multiple Poetry installation methods
install_poetry() {
    print_section "Poetry Installation"
    
    echo "Poetry will be installed to your user directory (~/.local/bin)"
    echo "This is safe and won't affect system Python."
    
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
        $PYTHON_EXECUTABLE -m pip install --user pipx
        if [ $? -eq 0 ]; then
            $PYTHON_EXECUTABLE -m pipx ensurepath
            export PATH="$HOME/.local/bin:$PATH"
            pipx install poetry
            if command -v poetry &> /dev/null; then
                print_success "Poetry installed successfully via pipx"
                return 0
            fi
        fi
    fi
    
    # Method 2: Try official installer
    echo "Attempting Poetry installation via official installer..."
    export DEB_PYTHON_INSTALL_LAYOUT=deb
    curl -sSL https://install.python-poetry.org | $PYTHON_EXECUTABLE -
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
    $PYTHON_EXECUTABLE -m venv "$POETRY_VENV"
    "$POETRY_VENV/bin/pip" install -U pip setuptools
    "$POETRY_VENV/bin/pip" install poetry
    
    if [ $? -eq 0 ]; then
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

# ---- MAIN EXECUTION ---- #

print_section "Self-Contained Project Setup"
echo "This script will set up your Python development environment."
echo -e "${BLUE}Use --yes to skip all prompts${NC}"
echo -e "${BLUE}Use --dev to install development dependencies${NC}"
echo

# Install system dependencies first
install_system_deps

# Check for Python 3.12, install if needed
check_python_version
if [ $? -ne 0 ]; then
    echo "Python 3.12 not found. Installation required for this project."
    install_python312
    if [ $? -ne 0 ]; then
        print_error "Cannot proceed without Python 3.12"
        echo -e "${YELLOW}You can manually install Python 3.12 and re-run this script.${NC}"
        exit 1
    fi
fi

# ---- POETRY SETUP ---- #
print_section "Poetry Setup"

# Check if Poetry is already installed
if command -v poetry &> /dev/null; then
    print_success "Poetry already installed ($(poetry --version))"
else
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

    # Use specific Python version if available
    if [ -n "$PYTHON_EXECUTABLE" ]; then
        echo "Setting Poetry to use $PYTHON_EXECUTABLE"
        poetry env use "$PYTHON_EXECUTABLE"
    fi

    # Check for virtual environment
    if [ ! -d ".venv" ]; then
        echo "No virtual environment found. Creating one..."
        
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
        if [ "$INSTALL_DEV" = true ]; then
            echo "Updating with development dependencies..."
            poetry install --with dev
        else
            echo "Updating dependencies..."
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
    print_success "Setup Complete! 🎉"
    echo
    print_success "Environment is ready! Here's how to use it:"
    echo -e "${GREEN}    Activate: ${BOLD}poetry shell${NC}${GREEN} or ${BOLD}source .venv/bin/activate${NC}"
    echo -e "${GREEN}    Run commands: ${BOLD}poetry run <command>${NC}"
    echo -e "${GREEN}    Install packages: ${BOLD}poetry add <package>${NC}"
    
    if [ -n "$PYTHON_EXECUTABLE" ]; then
        echo
        echo -e "${BLUE}    Python version: ${BOLD}$($PYTHON_EXECUTABLE --version)${NC}"
    fi
else
    print_warning "Setup completed with ${ERRORS_FOUND} issue(s)."
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}    You can still try: source .venv/bin/activate${NC}"
    fi
fi