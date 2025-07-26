#!/bin/bash
"""
Wrapper script to run the Repository Hygiene Bot

This script provides a convenient way to run the hygiene bot with
proper environment setup and error handling.
"""

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
DRY_RUN=false
VERBOSE=false
SINGLE_REPO=""
CONFIG_FILE="$PROJECT_ROOT/config/repo-hygiene.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Print usage information
usage() {
    cat << EOF
Repository Hygiene Bot Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -d, --dry-run       Run in dry-run mode (no changes made)
    -v, --verbose       Enable verbose logging
    -r, --repo REPO     Process only a single repository
    -c, --config FILE   Path to configuration file
    -u, --user USER     GitHub username (or set GITHUB_USER env var)
    -t, --token TOKEN   GitHub token (or set GITHUB_TOKEN env var)

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN        GitHub personal access token (required)
    GITHUB_USER         GitHub username (required)

EXAMPLES:
    # Run for all repositories
    $0 --user myuser --token \$GITHUB_TOKEN

    # Dry run with verbose output
    $0 --dry-run --verbose

    # Process single repository
    $0 --repo my-special-repo

    # Use custom config
    $0 --config /path/to/custom-config.yaml

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -r|--repo)
                SINGLE_REPO="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -u|--user)
                export GITHUB_USER="$2"
                shift 2
                ;;
            -t|--token)
                export GITHUB_TOKEN="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment and requirements
validate_environment() {
    log_info "Validating environment..."
    
    # Check required environment variables
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_error "GITHUB_TOKEN environment variable is required"
        log_info "Get a token from: https://github.com/settings/tokens"
        exit 1
    fi
    
    if [[ -z "${GITHUB_USER:-}" ]]; then
        log_error "GITHUB_USER environment variable is required"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not found"
        exit 1
    fi
    
    # Check if required Python packages are available
    cd "$PROJECT_ROOT"
    if ! python3 -c "import requests, yaml" &> /dev/null; then
        log_warn "Required Python packages not found, installing..."
        pip3 install requests PyYAML || {
            log_error "Failed to install required packages"
            exit 1
        }
    fi
    
    # Check if config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Setup logging directory
setup_logging() {
    local log_dir="$PROJECT_ROOT/logs"
    mkdir -p "$log_dir"
    
    # Create timestamped log file
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    export LOG_FILE="$log_dir/hygiene_bot_$timestamp.log"
    
    log_info "Logs will be written to: $LOG_FILE"
}

# Run the hygiene bot
run_bot() {
    log_info "Starting Repository Hygiene Bot..."
    
    cd "$PROJECT_ROOT"
    
    # Build command arguments
    local args=()
    
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=(--dry-run)
        log_warn "Running in DRY-RUN mode - no changes will be made"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=(--verbose)
    fi
    
    if [[ -n "$SINGLE_REPO" ]]; then
        args+=(--single-repo "$SINGLE_REPO")
        log_info "Processing single repository: $SINGLE_REPO"
    fi
    
    args+=(--config "$CONFIG_FILE")
    
    # Run the bot
    log_info "Executing: python3 -m src.repo_hygiene_cli ${args[*]}"
    
    if python3 -m src.repo_hygiene_cli "${args[@]}" 2>&1 | tee "$LOG_FILE"; then
        log_success "Repository hygiene bot completed successfully"
        
        # Show metrics if available
        if [[ -f "$PROJECT_ROOT/metrics/profile_hygiene.json" ]]; then
            local repo_count=$(jq length "$PROJECT_ROOT/metrics/profile_hygiene.json" 2>/dev/null || echo "unknown")
            log_success "Metrics collected for $repo_count repositories"
        fi
        
    else
        log_error "Repository hygiene bot failed"
        log_info "Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup tasks here
}

# Main execution
main() {
    log_info "Repository Hygiene Bot Runner"
    log_info "=============================="
    
    # Set up signal handlers
    trap cleanup EXIT
    trap 'log_error "Script interrupted"; exit 1' INT TERM
    
    # Parse arguments
    parse_args "$@"
    
    # Setup and validate
    setup_logging
    validate_environment
    
    # Run the bot
    run_bot
    
    log_success "All operations completed successfully"
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi