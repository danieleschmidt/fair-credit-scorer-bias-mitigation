#!/bin/bash
# Setup SBOM Generation Tools for MATURING repositories
# Install Syft and Grype for comprehensive supply chain security

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and architecture
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case $arch in
        x86_64) arch="amd64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) 
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    case $os in
        linux) platform="linux_${arch}" ;;
        darwin) platform="darwin_${arch}" ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    echo $platform
}

# Install Syft
install_syft() {
    log_info "Installing Syft SBOM generator..."
    
    if command -v syft &> /dev/null; then
        local current_version=$(syft version | head -n1 | awk '{print $2}')
        log_info "Syft already installed (version: $current_version)"
        
        read -p "Update to latest version? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    local platform=$(detect_platform)
    local latest_version=$(curl -s https://api.github.com/repos/anchore/syft/releases/latest | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4 | sed 's/^v//')
    local download_url="https://github.com/anchore/syft/releases/download/v${latest_version}/syft_${latest_version}_${platform}.tar.gz"
    
    log_info "Downloading Syft v${latest_version} for ${platform}..."
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Download and extract
    curl -L -o "$temp_dir/syft.tar.gz" "$download_url"
    tar -xzf "$temp_dir/syft.tar.gz" -C "$temp_dir"
    
    # Install to /usr/local/bin or user bin
    if [[ -w /usr/local/bin ]]; then
        sudo mv "$temp_dir/syft" /usr/local/bin/
        log_success "Syft installed to /usr/local/bin/syft"
    else
        mkdir -p "$HOME/.local/bin"
        mv "$temp_dir/syft" "$HOME/.local/bin/"
        log_success "Syft installed to $HOME/.local/bin/syft"
        log_warning "Make sure $HOME/.local/bin is in your PATH"
    fi
    
    # Verify installation
    if command -v syft &> /dev/null; then
        local installed_version=$(syft version | head -n1 | awk '{print $2}')
        log_success "Syft v${installed_version} successfully installed"
    else
        log_error "Syft installation failed"
        return 1
    fi
}

# Install Grype
install_grype() {
    log_info "Installing Grype vulnerability scanner..."
    
    if command -v grype &> /dev/null; then
        local current_version=$(grype version | head -n1 | awk '{print $2}')
        log_info "Grype already installed (version: $current_version)"
        
        read -p "Update to latest version? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    local platform=$(detect_platform)
    local latest_version=$(curl -s https://api.github.com/repos/anchore/grype/releases/latest | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4 | sed 's/^v//')
    local download_url="https://github.com/anchore/grype/releases/download/v${latest_version}/grype_${latest_version}_${platform}.tar.gz"
    
    log_info "Downloading Grype v${latest_version} for ${platform}..."
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Download and extract
    curl -L -o "$temp_dir/grype.tar.gz" "$download_url"
    tar -xzf "$temp_dir/grype.tar.gz" -C "$temp_dir"
    
    # Install to /usr/local/bin or user bin
    if [[ -w /usr/local/bin ]]; then
        sudo mv "$temp_dir/grype" /usr/local/bin/
        log_success "Grype installed to /usr/local/bin/grype"
    else
        mkdir -p "$HOME/.local/bin"
        mv "$temp_dir/grype" "$HOME/.local/bin/"
        log_success "Grype installed to $HOME/.local/bin/grype"
        log_warning "Make sure $HOME/.local/bin is in your PATH"
    fi
    
    # Verify installation
    if command -v grype &> /dev/null; then
        local installed_version=$(grype version | head -n1 | awk '{print $2}')
        log_success "Grype v${installed_version} successfully installed"
    else
        log_error "Grype installation failed"
        return 1
    fi
}

# Update vulnerability database
update_grype_db() {
    log_info "Updating Grype vulnerability database..."
    
    if command -v grype &> /dev/null; then
        grype db update
        log_success "Grype database updated"
    else
        log_warning "Grype not found, skipping database update"
    fi
}

# Verify tools installation
verify_installation() {
    log_info "Verifying SBOM tools installation..."
    
    local all_good=true
    
    # Check Syft
    if command -v syft &> /dev/null; then
        local syft_version=$(syft version | head -n1 | awk '{print $2}')
        log_success "Syft v${syft_version} is available"
    else
        log_error "Syft is not available in PATH"
        all_good=false
    fi
    
    # Check Grype
    if command -v grype &> /dev/null; then
        local grype_version=$(grype version | head -n1 | awk '{print $2}')
        log_success "Grype v${grype_version} is available"
    else
        log_error "Grype is not available in PATH"
        all_good=false
    fi
    
    if $all_good; then
        log_success "All SBOM tools are properly installed!"
        
        # Test with a simple command
        log_info "Testing tools with current directory..."
        syft packages . -o table | head -5
        echo ""
        
        return 0
    else
        log_error "Some tools are missing or not properly installed"
        return 1
    fi
}

# Create SBOM configuration
create_config() {
    log_info "Creating SBOM configuration files..."
    
    # Create .syft.yaml configuration
    cat > .syft.yaml << 'EOF'
# Syft configuration for comprehensive SBOM generation
catalogers:
  enabled:
    - python-installed-package-cataloger
    - python-package-cataloger  
    - python-poetry-lock-cataloger
    - python-pipfile-lock-cataloger
    - python-pip-requirements-cataloger

package:
  search-unindexed-archives: true
  search-indexed-archives: true

file-metadata:
  enabled: true
  digests: ["sha256", "sha1"]

output:
  - "json"
  - "spdx-json" 
  - "cyclonedx-json"

log:
  level: "info"
EOF
    
    # Create .grype.yaml configuration  
    cat > .grype.yaml << 'EOF'
# Grype configuration for vulnerability scanning
db:
  validate-by-hash-on-start: true
  validate-age: true
  max-allowed-built-age: 120h

match:
  python:
    using-cpes: true

output:
  - "json"
  - "table"

log:
  level: "info"

fail-on-severity: "high"
EOF
    
    log_success "Configuration files created (.syft.yaml, .grype.yaml)"
}

# Show usage information
show_usage() {
    cat << EOF
SBOM Tools Setup Script for MATURING Repositories

This script installs and configures tools for Software Bill of Materials (SBOM)
generation and vulnerability scanning.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -s, --syft-only         Install only Syft
    -g, --grype-only        Install only Grype  
    -u, --update-db         Update Grype vulnerability database
    -c, --config            Create configuration files only
    -v, --verify            Verify installation only

TOOLS:
    - Syft: SBOM generation tool by Anchore
    - Grype: Vulnerability scanner by Anchore

EXAMPLES:
    $0                      # Install all tools
    $0 --syft-only          # Install only Syft
    $0 --update-db          # Update vulnerability database
    $0 --verify             # Check if tools are installed

EOF
}

# Main execution
main() {
    local install_syft=true
    local install_grype=true
    local update_db=false
    local create_config_only=false
    local verify_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -s|--syft-only)
                install_grype=false
                shift
                ;;
            -g|--grype-only)
                install_syft=false
                shift
                ;;
            -u|--update-db)
                update_db=true
                shift
                ;;
            -c|--config)
                create_config_only=true
                shift
                ;;
            -v|--verify)
                verify_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Setting up SBOM tools for MATURING repository..."
    
    # Handle specific actions
    if $verify_only; then
        verify_installation
        exit $?
    fi
    
    if $create_config_only; then
        create_config
        exit 0
    fi
    
    if $update_db; then
        update_grype_db
        exit 0
    fi
    
    # Install tools
    local success=true
    
    if $install_syft; then
        install_syft || success=false
    fi
    
    if $install_grype; then
        install_grype || success=false
        
        # Update database after installation
        if $success; then
            update_grype_db
        fi
    fi
    
    # Create configuration
    if $success; then
        create_config
        verify_installation
    fi
    
    if $success; then
        log_success "🎉 SBOM tools setup completed successfully!"
        echo ""
        log_info "Next steps:"
        echo "  1. Run 'python sbom/generate_sbom.py' to generate SBOM"
        echo "  2. Use 'make sbom' if Makefile target exists"
        echo "  3. Integrate SBOM generation into CI/CD pipeline"
    else
        log_error "💥 SBOM tools setup failed!"
        exit 1
    fi
}

# Run main function
main "$@"