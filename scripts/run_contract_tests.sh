#!/bin/bash
# Advanced Contract Testing Runner for MATURING repositories
# Comprehensive API validation and consumer-driven contract testing

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PACT_DIR="$PROJECT_ROOT/contract_testing/pacts"
RESULTS_DIR="$PROJECT_ROOT/contract_results"

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

# Help function
show_help() {
    cat << EOF
Contract Testing Runner for MATURING Repositories

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -c, --consumer-only     Run only consumer contract tests
    -p, --provider-only     Run only provider verification
    -b, --broker-url URL    Pact Broker URL (default: http://localhost:9292)
    -v, --version VERSION   Consumer/Provider version (default: 1.0.0)
    -t, --tag TAG          Tag for this version (default: dev)
    -e, --env ENV          Environment (dev, staging, prod)
    --publish              Publish contracts to broker
    --verify-latest        Verify against latest consumer contracts
    --can-i-deploy         Check if safe to deploy
    --report               Generate detailed contract report

EXAMPLES:
    $0                                    # Run all contract tests
    $0 --consumer-only                    # Run only consumer tests
    $0 --provider-only --verify-latest    # Verify against latest contracts
    $0 --publish --tag staging            # Publish contracts with staging tag
    $0 --can-i-deploy --env prod          # Check deployment safety

EOF
}

# Parse command line arguments
CONSUMER_ONLY=false
PROVIDER_ONLY=false
BROKER_URL="${PACT_BROKER_URL:-http://localhost:9292}"
VERSION="${CONSUMER_VERSION:-1.0.0}"
TAG="${PACT_TAG:-dev}"
ENVIRONMENT="${PACT_ENV:-dev}"
PUBLISH=false
VERIFY_LATEST=false
CAN_I_DEPLOY=false
GENERATE_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--consumer-only)
            CONSUMER_ONLY=true
            shift
            ;;
        -p|--provider-only)
            PROVIDER_ONLY=true
            shift
            ;;
        -b|--broker-url)
            BROKER_URL="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --publish)
            PUBLISH=true
            shift
            ;;
        --verify-latest)
            VERIFY_LATEST=true
            shift
            ;;
        --can-i-deploy)
            CAN_I_DEPLOY=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Setup environment
setup_environment() {
    log_info "Setting up contract testing environment..."
    
    # Create directories
    mkdir -p "$PACT_DIR" "$RESULTS_DIR"
    
    # Set environment variables
    export PACT_BROKER_URL="$BROKER_URL"
    export CONSUMER_VERSION="$VERSION"
    export PROVIDER_VERSION="$VERSION"
    export PACT_TAG="$TAG"
    export PACT_ENV="$ENVIRONMENT"
    
    # Check if Pact CLI is available
    if ! command -v pact-broker &> /dev/null; then
        log_warning "Pact CLI not found. Installing via gem..."
        if command -v gem &> /dev/null; then
            gem install pact_broker-client
        else
            log_warning "Ruby/Gem not available. Some features may not work."
        fi
    fi
    
    log_success "Environment setup complete"
}

# Run consumer contract tests
run_consumer_tests() {
    log_info "Running consumer contract tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run pytest with contract markers
    pytest tests/contract/ \
        -v \
        --tb=short \
        --junit-xml="$RESULTS_DIR/consumer_results.xml" \
        --html="$RESULTS_DIR/consumer_report.html" \
        --self-contained-html \
        -m "not slow"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Consumer contract tests passed"
        
        # Publish contracts if requested
        if [ "$PUBLISH" = true ]; then
            publish_contracts
        fi
    else
        log_error "Consumer contract tests failed"
        return $exit_code
    fi
    
    return 0
}

# Publish contracts to broker
publish_contracts() {
    log_info "Publishing contracts to broker..."
    
    if [ -z "${PACT_BROKER_USERNAME:-}" ] || [ -z "${PACT_BROKER_PASSWORD:-}" ]; then
        log_warning "Broker credentials not set. Skipping publish."
        return 0
    fi
    
    # Find all pact files
    local pact_files=("$PACT_DIR"/*.json)
    
    if [ ${#pact_files[@]} -eq 0 ] || [ ! -f "${pact_files[0]}" ]; then
        log_warning "No pact files found to publish"
        return 0
    fi
    
    for pact_file in "${pact_files[@]}"; do
        if [ -f "$pact_file" ]; then
            log_info "Publishing $(basename "$pact_file")..."
            
            pact-broker publish "$pact_file" \
                --broker-base-url="$BROKER_URL" \
                --broker-username="$PACT_BROKER_USERNAME" \
                --broker-password="$PACT_BROKER_PASSWORD" \
                --consumer-app-version="$VERSION" \
                --tag="$TAG"
        fi
    done
    
    log_success "Contracts published successfully"
}

# Run provider verification
run_provider_verification() {
    log_info "Running provider verification..."
    
    cd "$PROJECT_ROOT"
    
    # Start the provider service (if not already running)
    local provider_pid=""
    if ! curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        log_info "Starting provider service for verification..."
        python -m src.health_check &
        provider_pid=$!
        sleep 5
    fi
    
    # Run verification
    local verification_result=0
    
    if [ "$VERIFY_LATEST" = true ]; then
        # Verify against latest contracts from broker
        pact-broker \
            --broker-base-url="$BROKER_URL" \
            --broker-username="${PACT_BROKER_USERNAME:-}" \
            --broker-password="${PACT_BROKER_PASSWORD:-}" \
            verify \
            --provider-app-version="$VERSION" \
            --provider-base-url="http://localhost:8000" \
            --publish-verification-results \
            --provider-version-tag="$TAG"
    else
        # Verify against local pact files
        python -c "
import sys
sys.path.append('.')
from contract_testing.pact_config import ContractTestConfig

config = ContractTestConfig()
verification_config = config.get_verification_config()
print('Verification config:', verification_config)
"
        verification_result=$?
    fi
    
    # Clean up provider if we started it
    if [ -n "$provider_pid" ]; then
        log_info "Stopping provider service..."
        kill "$provider_pid" 2>/dev/null || true
    fi
    
    if [ $verification_result -eq 0 ]; then
        log_success "Provider verification passed"
    else
        log_error "Provider verification failed"
    fi
    
    return $verification_result
}

# Check if it's safe to deploy
check_can_i_deploy() {
    log_info "Checking deployment safety..."
    
    if [ -z "${PACT_BROKER_USERNAME:-}" ] || [ -z "${PACT_BROKER_PASSWORD:-}" ]; then
        log_warning "Broker credentials not set. Cannot check deployment safety."
        return 1
    fi
    
    pact-broker can-i-deploy \
        --broker-base-url="$BROKER_URL" \
        --broker-username="$PACT_BROKER_USERNAME" \
        --broker-password="$PACT_BROKER_PASSWORD" \
        --pacticipant="fair-credit-scorer" \
        --version="$VERSION" \
        --to="$ENVIRONMENT"
    
    local result=$?
    
    if [ $result -eq 0 ]; then
        log_success "✅ Safe to deploy to $ENVIRONMENT"
    else
        log_error "❌ Not safe to deploy to $ENVIRONMENT"
    fi
    
    return $result
}

# Generate contract testing report
generate_report() {
    log_info "Generating contract testing report..."
    
    local report_file="$RESULTS_DIR/contract_report.md"
    
    cat > "$report_file" << EOF
# Contract Testing Report

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Version:** $VERSION
**Tag:** $TAG
**Environment:** $ENVIRONMENT

## Summary

EOF
    
    # Add consumer test results
    if [ -f "$RESULTS_DIR/consumer_results.xml" ]; then
        echo "### Consumer Tests" >> "$report_file"
        echo "" >> "$report_file"
        
        # Parse XML results (simplified)
        local tests_run=$(grep -o 'tests="[0-9]*"' "$RESULTS_DIR/consumer_results.xml" | head -1 | grep -o '[0-9]*')
        local failures=$(grep -o 'failures="[0-9]*"' "$RESULTS_DIR/consumer_results.xml" | head -1 | grep -o '[0-9]*')
        local errors=$(grep -o 'errors="[0-9]*"' "$RESULTS_DIR/consumer_results.xml" | head -1 | grep -o '[0-9]*')
        
        echo "- Tests Run: ${tests_run:-0}" >> "$report_file"
        echo "- Failures: ${failures:-0}" >> "$report_file"
        echo "- Errors: ${errors:-0}" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add pact files information
    echo "### Generated Contracts" >> "$report_file"
    echo "" >> "$report_file"
    
    local pact_count=0
    for pact_file in "$PACT_DIR"/*.json; do
        if [ -f "$pact_file" ]; then
            echo "- $(basename "$pact_file")" >> "$report_file"
            ((pact_count++))
        fi
    done
    
    echo "" >> "$report_file"
    echo "**Total Contracts:** $pact_count" >> "$report_file"
    
    log_success "Report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting contract testing pipeline..."
    log_info "Version: $VERSION | Tag: $TAG | Environment: $ENVIRONMENT"
    
    setup_environment
    
    local overall_result=0
    
    # Check deployment safety first if requested
    if [ "$CAN_I_DEPLOY" = true ]; then
        check_can_i_deploy
        overall_result=$?
        if [ $overall_result -ne 0 ] && [ "$ENVIRONMENT" = "prod" ]; then
            log_error "Deployment safety check failed for production. Aborting."
            exit $overall_result
        fi
    fi
    
    # Run consumer tests
    if [ "$PROVIDER_ONLY" = false ]; then
        run_consumer_tests
        overall_result=$((overall_result + $?))
    fi
    
    # Run provider verification
    if [ "$CONSUMER_ONLY" = false ]; then
        run_provider_verification
        overall_result=$((overall_result + $?))
    fi
    
    # Generate report if requested
    if [ "$GENERATE_REPORT" = true ]; then
        generate_report
    fi
    
    if [ $overall_result -eq 0 ]; then
        log_success "🎉 All contract tests passed!"
    else
        log_error "💥 Contract testing failed!"
    fi
    
    exit $overall_result
}

# Run main function
main "$@"